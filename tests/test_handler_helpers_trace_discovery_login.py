import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pymc_core.protocol.constants import PAYLOAD_TYPE_ANON_REQ, ROUTE_TYPE_DIRECT
from repeater.handler_helpers.discovery import DiscoveryHelper
from repeater.handler_helpers.login import LoginHelper
from repeater.handler_helpers.trace import TraceHelper


class DummyPacket:
    def __init__(
        self, *, route=ROUTE_TYPE_DIRECT, path=b"", payload=b"\x01\x02", snr=2.5, rssi=-70
    ):
        self.header = route
        self.path = bytearray(path)
        self.path_len = len(self.path)
        self.payload = bytearray(payload)
        self.snr = snr
        self.rssi = rssi

    def get_route_type(self):
        return self.header

    def get_payload_type(self):
        return 0x09

    def get_snr(self):
        return self.snr

    def calculate_packet_hash(self):
        return bytes.fromhex("A1B2C3D4E5F6A7B8")

    def write_to(self):
        return b"\x01\x02\x03"


class FakeIdentity:
    def __init__(self, first_byte=0x42):
        self._pk = bytes([first_byte]) + bytes(range(1, 33))

    def get_public_key(self):
        return self._pk


@pytest.mark.asyncio
async def test_trace_helper_should_forward_matching_next_hop_only():
    repeater_handler = MagicMock()
    repeater_handler.is_duplicate.return_value = False
    helper = TraceHelper(
        local_hash=0x42,
        local_identity=FakeIdentity(0x42),
        repeater_handler=repeater_handler,
    )
    packet = DummyPacket(path=b"\x00")

    assert helper._should_forward_trace(packet, b"", flags=0, hash_width=1) is False
    assert helper._should_forward_trace(packet, b"\x01", flags=0, hash_width=0) is False

    # offset = len(path)=1 for hash_width=1, so this trace is complete and not forwarded
    assert helper._should_forward_trace(packet, b"\x42", flags=0, hash_width=1) is False

    # next hop mismatch
    packet.path = bytearray()
    assert helper._should_forward_trace(packet, b"\x99", flags=0, hash_width=1) is False

    # match + non-duplicate forwards
    assert helper._should_forward_trace(packet, b"\x42", flags=0, hash_width=1) is True

    repeater_handler.is_duplicate.return_value = True
    assert helper._should_forward_trace(packet, b"\x42", flags=0, hash_width=1) is False


@pytest.mark.asyncio
async def test_trace_helper_process_sets_pending_ping_and_forwards():
    repeater_handler = MagicMock()
    repeater_handler.is_duplicate.return_value = False
    repeater_handler.calculate_packet_score.return_value = 0.9
    helper = TraceHelper(
        local_hash=0x42,
        local_identity=FakeIdentity(0x42),
        repeater_handler=repeater_handler,
    )

    tag = 77
    evt = helper.register_ping(tag, 0x42)

    packet = DummyPacket(path=b"\x01", payload=b"\xaa\xbb\xcc")
    helper._forward_trace_packet = AsyncMock()
    helper._extract_path_info = MagicMock(return_value=([], []))
    helper._should_forward_trace = MagicMock(return_value=True)
    helper.trace_handler._parse_trace_payload = MagicMock(
        return_value={
            "valid": True,
            "trace_path_bytes": b"\x42",
            "flags": 0,
            "trace_hops": [b"\x42"],
            "trace_path": [0x42],
            "tag": tag,
        }
    )
    helper.trace_handler._format_trace_response = MagicMock(return_value="trace ok")

    await helper.process_trace_packet(packet)

    assert evt.is_set()
    assert helper.pending_pings[tag]["result"]["rssi"] == -70
    repeater_handler.log_trace_record.assert_called_once()
    helper._forward_trace_packet.assert_awaited_once()


@pytest.mark.asyncio
async def test_trace_helper_ignores_zero_rssi_pending_ping_response():
    helper = TraceHelper(
        local_hash=0x42, local_identity=FakeIdentity(0x42), repeater_handler=MagicMock()
    )
    tag = 9
    evt = helper.register_ping(tag, 0x42)

    packet = DummyPacket(path=b"\x01", rssi=0)
    helper.trace_handler._parse_trace_payload = MagicMock(
        return_value={
            "valid": True,
            "trace_path_bytes": b"\x42",
            "flags": 0,
            "trace_hops": [b"\x42"],
            "trace_path": [0x42],
            "tag": tag,
        }
    )

    await helper.process_trace_packet(packet)

    assert not evt.is_set()
    assert helper.pending_pings[tag]["result"] is None


@pytest.mark.asyncio
async def test_trace_helper_forward_trace_packet_updates_recent_record_and_injects():
    packet_injector = AsyncMock(return_value=True)
    repeater_handler = MagicMock()
    pkt = DummyPacket(path=b"", snr=3.5)
    pkt_hash = pkt.calculate_packet_hash().hex().upper()[:16]
    repeater_handler.recent_packets = [{"packet_hash": pkt_hash, "transmitted": False}]

    helper = TraceHelper(
        local_hash=0x42,
        local_identity=FakeIdentity(0x42),
        repeater_handler=repeater_handler,
        packet_injector=packet_injector,
    )

    await helper._forward_trace_packet(pkt, num_hops=1)

    assert repeater_handler.recent_packets[0]["transmitted"] is True
    assert repeater_handler.recent_packets[0]["drop_reason"] == "trace_forwarded"
    assert pkt.path_len == 1
    packet_injector.assert_awaited_once()


def test_trace_helper_cleanup_stale_pings():
    helper = TraceHelper(
        local_hash=0x42, local_identity=FakeIdentity(0x42), repeater_handler=MagicMock()
    )
    helper.pending_pings = {
        1: {"sent_at": time.time() - 100, "event": asyncio.Event(), "result": None, "target": 1},
        2: {"sent_at": time.time(), "event": asyncio.Event(), "result": None, "target": 2},
    }

    helper.cleanup_stale_pings(max_age_seconds=10)

    assert 1 not in helper.pending_pings
    assert 2 in helper.pending_pings


def test_discovery_request_filter_match_and_mismatch():
    helper = DiscoveryHelper(
        local_identity=FakeIdentity(0x42), packet_injector=AsyncMock(), node_type=2
    )
    helper._send_discovery_response = MagicMock()

    helper._on_discovery_request(
        {"tag": 1, "filter": 0x00, "prefix_only": False, "snr": 1.2, "rssi": -80}
    )
    helper._send_discovery_response.assert_not_called()

    helper._on_discovery_request(
        {"tag": 2, "filter": 0x04, "prefix_only": True, "snr": 2.3, "rssi": -70}
    )
    helper._send_discovery_response.assert_called_once_with(2, 2, 2.3, True)


def test_discovery_request_without_identity_does_not_send():
    helper = DiscoveryHelper(local_identity=None, packet_injector=AsyncMock(), node_type=2)
    helper._send_discovery_response = MagicMock()

    helper._on_discovery_request(
        {"tag": 7, "filter": 0x04, "prefix_only": False, "snr": 0.0, "rssi": -90}
    )

    helper._send_discovery_response.assert_not_called()


@pytest.mark.asyncio
async def test_discovery_send_packet_async_success_failure_and_exception():
    injector = AsyncMock(side_effect=[True, False, RuntimeError("send fail")])
    helper = DiscoveryHelper(local_identity=FakeIdentity(0x42), packet_injector=injector)

    await helper._send_packet_async(packet=object(), tag=0x11)
    await helper._send_packet_async(packet=object(), tag=0x12)
    await helper._send_packet_async(packet=object(), tag=0x13)

    assert injector.await_count == 3


def test_discovery_send_response_without_injector_is_safe():
    helper = DiscoveryHelper(local_identity=FakeIdentity(0x42), packet_injector=None)

    with patch(
        "pymc_core.protocol.packet_builder.PacketBuilder.create_discovery_response",
        return_value=object(),
    ):
        helper._send_discovery_response(tag=5, node_type=2, inbound_snr=1.0, prefix_only=False)


def test_login_register_identity_room_server_requires_passwords():
    helper = LoginHelper(identity_manager=MagicMock(), packet_injector=AsyncMock())
    identity = FakeIdentity(0x51)

    with (
        patch("repeater.handler_helpers.acl.ACL") as acl_cls,
        patch("repeater.handler_helpers.login.LoginServerHandler") as handler_cls,
    ):
        helper.register_identity(
            name="room-a",
            identity=identity,
            identity_type="room_server",
            config={"settings": {}},
        )

    acl_cls.assert_not_called()
    handler_cls.assert_not_called()
    assert 0x51 not in helper.handlers


def test_login_register_identity_repeater_creates_acl_and_handler():
    helper = LoginHelper(identity_manager=MagicMock(), packet_injector=AsyncMock())
    identity = FakeIdentity(0x52)
    acl_obj = MagicMock()
    handler_obj = MagicMock()

    with (
        patch("repeater.handler_helpers.acl.ACL", return_value=acl_obj) as acl_cls,
        patch(
            "repeater.handler_helpers.login.LoginServerHandler", return_value=handler_obj
        ) as handler_cls,
    ):
        helper.register_identity(
            name="repeater-main",
            identity=identity,
            identity_type="repeater",
            config={
                "repeater": {
                    "security": {"max_clients": 3, "admin_password": "a", "guest_password": "g"}
                }
            },
        )

    acl_cls.assert_called_once()
    handler_cls.assert_called_once()
    handler_obj.set_send_packet_callback.assert_called_once()
    assert helper.handlers[0x52] is handler_obj
    assert helper.acls[0x52] is acl_obj


@pytest.mark.asyncio
async def test_login_process_packet_routes_to_registered_handler_and_marks_no_retransmit():
    helper = LoginHelper(identity_manager=MagicMock(), packet_injector=AsyncMock())
    login_handler = AsyncMock()
    helper.handlers[0x62] = login_handler

    packet = SimpleNamespace(
        payload=bytearray([0x62, 0xAA]),
        get_payload_type=lambda: 0x01,
        mark_do_not_retransmit=MagicMock(),
    )

    handled = await helper.process_login_packet(packet)

    assert handled is True
    login_handler.assert_awaited_once_with(packet)
    packet.mark_do_not_retransmit.assert_called_once()


@pytest.mark.asyncio
async def test_login_process_packet_unknown_and_short_payload_are_not_handled():
    helper = LoginHelper(identity_manager=MagicMock(), packet_injector=AsyncMock())

    short_packet = SimpleNamespace(payload=bytearray())
    assert await helper.process_login_packet(short_packet) is False

    unknown_packet = SimpleNamespace(
        payload=bytearray([0x63]),
        get_payload_type=lambda: PAYLOAD_TYPE_ANON_REQ,
    )
    assert await helper.process_login_packet(unknown_packet) is False


@pytest.mark.asyncio
async def test_login_delayed_send_success_and_error_paths():
    injector = AsyncMock(side_effect=[True, RuntimeError("send failed")])
    helper = LoginHelper(identity_manager=MagicMock(), packet_injector=injector)

    with patch("repeater.handler_helpers.login.asyncio.sleep", new_callable=AsyncMock):
        await helper._delayed_send(packet=object(), delay_ms=10)
        await helper._delayed_send(packet=object(), delay_ms=10)

    assert injector.await_count == 2


def test_login_acl_access_and_client_listing():
    helper = LoginHelper(identity_manager=MagicMock(), packet_injector=AsyncMock())
    acl_a = MagicMock()
    acl_b = MagicMock()
    acl_a.get_all_clients.return_value = [{"id": "a1"}]
    acl_b.get_all_clients.return_value = [{"id": "b1"}, {"id": "b2"}]
    helper.acls = {0x70: acl_a, 0x71: acl_b}

    assert helper.get_acl_for_identity(0x70) is acl_a
    assert helper.get_acl_for_identity(0x99) is None
    assert helper.list_authenticated_clients(0x71) == [{"id": "b1"}, {"id": "b2"}]

    all_clients = helper.list_authenticated_clients()
    assert {c["id"] for c in all_clients} == {"a1", "b1", "b2"}
