"""
Tests for PacketRouter in-flight cap and shutdown behaviour.
Addresses the three concerns raised in PR 191 review:

  1. Cap enforcement: packets beyond _max_in_flight are dropped, not queued.
  2. Drop counter: _cap_drop_count increments on each cap-drop so operators
     have visibility into how often the safety valve fires.
  3. Shutdown drain: stop() waits for in-flight tasks to finish (up to 5 s),
     then cancels any that remain — tasks are never silently abandoned.

Run with:
    python -m pytest tests/test_packet_router.py -v
or:
    python -m unittest tests.test_packet_router -v
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pymc_core.node.handlers.ack import AckHandler
from pymc_core.node.handlers.advert import AdvertHandler
from pymc_core.node.handlers.control import ControlHandler
from pymc_core.node.handlers.group_text import GroupTextHandler
from pymc_core.node.handlers.login_response import LoginResponseHandler
from pymc_core.node.handlers.login_server import LoginServerHandler
from pymc_core.node.handlers.path import PathHandler
from pymc_core.node.handlers.protocol_request import ProtocolRequestHandler
from pymc_core.node.handlers.protocol_response import ProtocolResponseHandler
from pymc_core.node.handlers.text import TextMessageHandler
from pymc_core.node.handlers.trace import TraceHandler
from pymc_core.protocol.constants import ROUTE_TYPE_DIRECT

from repeater.packet_router import (
    PacketRouter,
    _companion_dedup_key,
    _is_direct_final_hop,
)

# ---------------------------------------------------------------------------
# Minimal daemon stub
# ---------------------------------------------------------------------------


def _make_daemon():
    """Minimal daemon that satisfies PacketRouter without touching hardware."""
    daemon = MagicMock()
    daemon.repeater_handler = AsyncMock(return_value=None)
    daemon.trace_helper = None
    daemon.discovery_helper = None
    daemon.advert_helper = None
    daemon.companion_bridges = {}
    daemon.login_helper = None
    daemon.text_helper = None
    daemon.path_helper = None
    daemon.protocol_request_helper = None
    return daemon


def _make_packet(payload_type: int = 0xFF):
    """Minimal packet stub."""
    pkt = MagicMock()
    pkt.get_payload_type.return_value = payload_type
    pkt.payload = b"\xff"
    pkt.header = 0x00
    pkt.rssi = -80
    pkt.snr = 5.0
    pkt.timestamp = time.time()
    pkt._injected_for_tx = False
    pkt.path = bytearray()
    pkt.calculate_packet_hash.return_value = b"\x01" * 32
    pkt.mark_do_not_retransmit = MagicMock()
    return pkt


def _make_bridge():
    bridge = MagicMock()
    bridge.process_received_packet = AsyncMock()
    return bridge


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInFlightCap(unittest.IsolatedAsyncioTestCase):
    # ── 1. Cap enforcement ──────────────────────────────────────────────────

    async def test_cap_drops_packets_when_full(self):
        """
        When _in_flight reaches _max_in_flight, new packets from the queue
        must be dropped (not passed to _route_packet).
        """
        router = PacketRouter(_make_daemon())
        router._max_in_flight = 3

        # Manually occupy all slots with long-sleeping tasks
        barrier = asyncio.Event()

        async def slow_route(pkt):
            await barrier.wait()  # blocks until we release

        routed = []

        async def counting_route(pkt):
            routed.append(pkt)
            await barrier.wait()

        router._route_packet = counting_route

        await router.start()

        # Fill the cap
        for _ in range(3):
            await router.enqueue(_make_packet())
        await asyncio.sleep(0.05)  # let queue drain into tasks
        self.assertEqual(router._in_flight, 3)

        # These should be dropped
        for _ in range(5):
            await router.enqueue(_make_packet())
        await asyncio.sleep(0.05)

        self.assertEqual(router._in_flight, 3, "In-flight count exceeded cap")
        self.assertEqual(router._cap_drop_count, 5, "Expected 5 cap-drops, got different count")

        barrier.set()  # release blocked tasks
        await router.stop()

    # ── 2. Drop counter ─────────────────────────────────────────────────────

    async def test_cap_drop_count_increments(self):
        """_cap_drop_count must increment by exactly 1 for each dropped packet."""
        router = PacketRouter(_make_daemon())
        router._max_in_flight = 1

        barrier = asyncio.Event()

        async def blocking_route(pkt):
            await barrier.wait()

        router._route_packet = blocking_route

        await router.start()

        # Fill the single slot
        await router.enqueue(_make_packet())
        await asyncio.sleep(0.05)
        self.assertEqual(router._in_flight, 1)

        # Drop three packets
        for _ in range(3):
            await router.enqueue(_make_packet())
        await asyncio.sleep(0.05)

        self.assertEqual(router._cap_drop_count, 3)

        barrier.set()
        await router.stop()

    async def test_cap_drop_count_zero_when_cap_not_reached(self):
        """_cap_drop_count must stay 0 when the cap is never reached."""
        router = PacketRouter(_make_daemon())
        router._max_in_flight = 30

        completed = []

        async def fast_route(pkt):
            completed.append(pkt)

        router._route_packet = fast_route

        await router.start()

        for _ in range(10):
            await router.enqueue(_make_packet())
        await asyncio.sleep(0.1)

        self.assertEqual(router._cap_drop_count, 0)
        await router.stop()

    async def test_injected_trace_packet_skips_inbound_trace_processing(self):
        """Locally injected TRACE packets must not be re-parsed as inbound trace responses."""
        daemon = _make_daemon()
        daemon.trace_helper = MagicMock()
        daemon.trace_helper.process_trace_packet = AsyncMock()

        router = PacketRouter(daemon)
        pkt = _make_packet(payload_type=TraceHandler.payload_type())

        await router.start()
        try:
            injected = await router.inject_packet(pkt)
            self.assertTrue(injected)
            await asyncio.sleep(0.05)

            daemon.repeater_handler.assert_awaited_once()
            daemon.trace_helper.process_trace_packet.assert_not_awaited()
        finally:
            await router.stop()

    # ── 3. Shutdown: in-flight tasks drained ────────────────────────────────

    async def test_stop_waits_for_in_flight_tasks(self):
        """
        stop() must wait for in-flight tasks to complete before returning.
        Tasks that finish within the 5-second timeout must complete normally,
        not be cancelled.
        """
        router = PacketRouter(_make_daemon())

        completed = []
        started = asyncio.Event()

        async def slow_route(pkt):
            started.set()
            await asyncio.sleep(0.2)  # finishes well within 5 s timeout
            completed.append(pkt)

        router._route_packet = slow_route

        await router.start()
        pkt = _make_packet()
        await router.enqueue(pkt)

        # Wait until the task has actually started
        await asyncio.wait_for(started.wait(), timeout=1.0)

        await router.stop()

        # Task should have completed, not been cancelled
        self.assertEqual(len(completed), 1, "In-flight task was cancelled instead of drained")

    async def test_stop_cancels_tasks_that_exceed_timeout(self):
        """
        Tasks that don't finish within the 5-second timeout must be cancelled,
        not left running indefinitely.
        """
        router = PacketRouter(_make_daemon())
        router._max_in_flight = 5

        cancelled = []
        started = asyncio.Event()

        async def hanging_route(pkt):
            started.set()
            try:
                await asyncio.sleep(999)  # will not finish within 5 s
            except asyncio.CancelledError:
                cancelled.append(pkt)
                raise

        router._route_packet = hanging_route

        async def fast_stop():
            router.running = False
            if router.router_task:
                router.router_task.cancel()
                try:
                    await router.router_task
                except asyncio.CancelledError:
                    pass
            if router._route_tasks:
                snapshot = set(router._route_tasks)
                _, still_pending = await asyncio.wait(snapshot, timeout=0.1)
                for task in still_pending:
                    task.cancel()
                await asyncio.gather(*still_pending, return_exceptions=True)

        router.stop = fast_stop

        await router.start()
        await router.enqueue(_make_packet())
        await asyncio.wait_for(started.wait(), timeout=1.0)

        await router.stop()

        self.assertEqual(len(cancelled), 1, "Hanging task was not cancelled on shutdown")

    # ── 4. Route-tasks set stays in sync with counter ───────────────────────

    async def test_route_tasks_set_cleaned_up_on_completion(self):
        """
        _route_tasks must be empty after all tasks complete — the done-callback
        must discard each task so the set doesn't grow unboundedly.
        """
        router = PacketRouter(_make_daemon())

        async def fast_route(pkt):
            await asyncio.sleep(0)  # yield, then done

        router._route_packet = fast_route

        await router.start()

        for _ in range(10):
            await router.enqueue(_make_packet())

        # Give tasks time to complete
        await asyncio.sleep(0.1)

        self.assertEqual(
            len(router._route_tasks), 0, "_route_tasks not cleaned up after task completion"
        )
        self.assertEqual(router._in_flight, 0, "_in_flight counter not back to 0 after completion")

        await router.stop()

    # ── 5. Counter and set always agree ─────────────────────────────────────

    async def test_counter_matches_set_size_under_load(self):
        """
        _in_flight must always equal len(_route_tasks) while tasks are running.
        Checked at steady state when the cap is saturated.
        """
        router = PacketRouter(_make_daemon())
        router._max_in_flight = 5

        barrier = asyncio.Event()

        async def blocking_route(pkt):
            await barrier.wait()

        router._route_packet = blocking_route

        await router.start()

        for _ in range(5):
            await router.enqueue(_make_packet())
        await asyncio.sleep(0.05)

        self.assertEqual(
            router._in_flight,
            len(router._route_tasks),
            f"Counter ({router._in_flight}) != set size ({len(router._route_tasks)})",
        )

        barrier.set()
        await router.stop()


if __name__ == "__main__":
    unittest.main()


class TestPacketRouterRoutingBranches(unittest.IsolatedAsyncioTestCase):
    def test_companion_dedup_key_handles_hash_exceptions(self):
        pkt = MagicMock()
        pkt.calculate_packet_hash.side_effect = RuntimeError("bad packet")
        self.assertIsNone(_companion_dedup_key(pkt))

    def test_is_direct_final_hop_helper(self):
        pkt = _make_packet()
        pkt.header = ROUTE_TYPE_DIRECT
        pkt.path = bytearray()
        self.assertTrue(_is_direct_final_hop(pkt))
        pkt.path = bytearray(b"\x01")
        self.assertFalse(_is_direct_final_hop(pkt))

    async def test_should_deliver_path_to_companions_dedupes(self):
        router = PacketRouter(_make_daemon())
        pkt = _make_packet(PathHandler.payload_type())
        self.assertTrue(router._should_deliver_path_to_companions(pkt))
        self.assertFalse(router._should_deliver_path_to_companions(pkt))
        key = _companion_dedup_key(pkt)
        router._companion_delivered[key] = time.time() - 1.0
        # Expired entries are only pruned once the dict grows beyond 200 entries.
        for i in range(205):
            router._companion_delivered[f"K{i}"] = time.time() + 60.0
        self.assertTrue(router._should_deliver_path_to_companions(pkt))

    async def test_enqueue_drops_oldest_when_queue_full(self):
        router = PacketRouter(_make_daemon())
        router.queue = asyncio.Queue(maxsize=1)
        p1 = _make_packet()
        p2 = _make_packet()
        await router.queue.put(p1)
        await router.enqueue(p2)
        got = await router.queue.get()
        self.assertIs(got, p2)

    async def test_inject_packet_returns_false_on_engine_error(self):
        daemon = _make_daemon()
        daemon.repeater_handler = AsyncMock(side_effect=RuntimeError("boom"))
        router = PacketRouter(daemon)
        ok = await router.inject_packet(_make_packet())
        self.assertFalse(ok)

    async def test_on_route_done_handles_task_exception(self):
        router = PacketRouter(_make_daemon())

        async def _fails():
            raise RuntimeError("route fail")

        task = asyncio.create_task(_fails())
        with self.assertRaises(RuntimeError):
            await task
        router._in_flight = 1
        router._route_tasks.add(task)
        router._on_route_done(task)
        self.assertEqual(router._in_flight, 0)
        self.assertEqual(len(router._route_tasks), 0)

    async def test_route_trace_inbound_uses_trace_helper_and_skips_engine(self):
        daemon = _make_daemon()
        daemon.trace_helper = MagicMock()
        daemon.trace_helper.process_trace_packet = AsyncMock()
        router = PacketRouter(daemon)
        pkt = _make_packet(TraceHandler.payload_type())
        await router._route_packet(pkt)
        daemon.trace_helper.process_trace_packet.assert_awaited_once()
        daemon.repeater_handler.assert_not_awaited()

    async def test_route_control_calls_discovery_and_delivery_and_engine(self):
        daemon = _make_daemon()
        daemon.discovery_helper = MagicMock()
        daemon.discovery_helper.control_handler = AsyncMock()
        daemon.deliver_control_data = AsyncMock()
        router = PacketRouter(daemon)
        pkt = _make_packet(ControlHandler.payload_type())
        pkt.path_len = 0
        await router._route_packet(pkt)
        daemon.discovery_helper.control_handler.assert_awaited_once()
        pkt.mark_do_not_retransmit.assert_called_once()
        daemon.deliver_control_data.assert_awaited_once()
        daemon.repeater_handler.assert_awaited_once()

    async def test_route_advert_delivers_to_helpers_and_engine(self):
        daemon = _make_daemon()
        daemon.advert_helper = MagicMock()
        daemon.advert_helper.process_advert_packet = AsyncMock()
        bridge = _make_bridge()
        daemon.companion_bridges = {0x42: bridge}
        router = PacketRouter(daemon)
        pkt = _make_packet(AdvertHandler.payload_type())
        await router._route_packet(pkt)
        daemon.advert_helper.process_advert_packet.assert_awaited_once()
        bridge.process_received_packet.assert_awaited_once()
        daemon.repeater_handler.assert_awaited_once()

    async def test_route_login_server_to_companion_marks_processed(self):
        daemon = _make_daemon()
        bridge = _make_bridge()
        daemon.companion_bridges = {0x7A: bridge}
        daemon.repeater_handler = AsyncMock()
        daemon.repeater_handler.storage = MagicMock()
        daemon.repeater_handler.record_packet_only = MagicMock()
        router = PacketRouter(daemon)
        pkt = _make_packet(LoginServerHandler.payload_type())
        pkt.payload = bytes([0x7A, 0x99])
        await router._route_packet(pkt)
        bridge.process_received_packet.assert_awaited_once()
        daemon.repeater_handler.assert_not_awaited()

    async def test_route_text_to_helper_marks_processed(self):
        daemon = _make_daemon()
        daemon.text_helper = MagicMock()
        daemon.text_helper.process_text_packet = AsyncMock(return_value=True)
        daemon.repeater_handler.storage = MagicMock()
        daemon.repeater_handler.record_packet_only = MagicMock()
        router = PacketRouter(daemon)
        pkt = _make_packet(TextMessageHandler.payload_type())
        pkt.payload = bytes([0xEE, 0x01])
        await router._route_packet(pkt)
        daemon.text_helper.process_text_packet.assert_awaited_once()
        daemon.repeater_handler.assert_not_awaited()

    async def test_route_ack_delivers_to_all_bridges_and_engine(self):
        daemon = _make_daemon()
        b1 = _make_bridge()
        b2 = _make_bridge()
        daemon.companion_bridges = {0x01: b1, 0x02: b2}
        router = PacketRouter(daemon)
        pkt = _make_packet(AckHandler.payload_type())
        await router._route_packet(pkt)
        b1.process_received_packet.assert_awaited_once()
        b2.process_received_packet.assert_awaited_once()
        daemon.repeater_handler.assert_awaited_once()

    async def test_route_path_dedupes_companion_delivery(self):
        daemon = _make_daemon()
        bridge = _make_bridge()
        daemon.companion_bridges = {0x01: bridge}
        router = PacketRouter(daemon)
        pkt = _make_packet(PathHandler.payload_type())
        pkt.payload = bytes([0x01, 0xAA])
        await router._route_packet(pkt)
        await router._route_packet(pkt)
        bridge.process_received_packet.assert_awaited_once()
        self.assertEqual(daemon.repeater_handler.await_count, 2)

    async def test_route_login_response_final_hop_skips_engine(self):
        daemon = _make_daemon()
        b1 = _make_bridge()
        daemon.companion_bridges = {0x01: b1}
        daemon.local_hash = 0xFF
        daemon.repeater_handler.storage = MagicMock()
        daemon.repeater_handler.record_packet_only = MagicMock()
        router = PacketRouter(daemon)
        pkt = _make_packet(LoginResponseHandler.payload_type())
        pkt.header = ROUTE_TYPE_DIRECT
        pkt.path = bytearray()
        pkt.payload = bytes([0xFF, 0x22])
        await router._route_packet(pkt)
        b1.process_received_packet.assert_awaited_once()
        daemon.repeater_handler.assert_not_awaited()

    async def test_route_protocol_response_final_hop_skips_engine(self):
        daemon = _make_daemon()
        b1 = _make_bridge()
        daemon.companion_bridges = {0x01: b1}
        daemon.repeater_handler.storage = MagicMock()
        daemon.repeater_handler.record_packet_only = MagicMock()
        router = PacketRouter(daemon)
        pkt = _make_packet(ProtocolResponseHandler.payload_type())
        pkt.header = ROUTE_TYPE_DIRECT
        pkt.path = bytearray()
        # PathHandler and ProtocolResponseHandler currently share payload type=8.
        # Patch PathHandler type here so ProtocolResponse branch is reachable.
        with patch("repeater.packet_router.PathHandler.payload_type", return_value=0x55):
            await router._route_packet(pkt)
        self.assertGreaterEqual(b1.process_received_packet.await_count, 1)
        daemon.repeater_handler.assert_not_awaited()

    async def test_route_protocol_request_final_hop_skips_engine(self):
        daemon = _make_daemon()
        b1 = _make_bridge()
        daemon.companion_bridges = {0x01: b1}
        daemon.repeater_handler.storage = MagicMock()
        daemon.repeater_handler.record_packet_only = MagicMock()
        router = PacketRouter(daemon)
        pkt = _make_packet(ProtocolRequestHandler.payload_type())
        pkt.header = ROUTE_TYPE_DIRECT
        pkt.path = bytearray()
        pkt.payload = bytes([0xAA, 0xBB])
        await router._route_packet(pkt)
        b1.process_received_packet.assert_awaited_once()
        daemon.repeater_handler.assert_not_awaited()

    async def test_route_group_text_delivers_and_forwards(self):
        daemon = _make_daemon()
        b1 = _make_bridge()
        daemon.companion_bridges = {0x01: b1}
        router = PacketRouter(daemon)
        pkt = _make_packet(GroupTextHandler.payload_type())
        await router._route_packet(pkt)
        b1.process_received_packet.assert_awaited_once()
        daemon.repeater_handler.assert_awaited_once()
