import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from repeater.data_acquisition.storage_collector import StorageCollector

sys.modules.setdefault("psutil", types.ModuleType("psutil"))

nacl_module = types.ModuleType("nacl")
nacl_signing_module = types.ModuleType("nacl.signing")


class _SigningKeyStub:
    pass


nacl_signing_module.SigningKey = _SigningKeyStub
nacl_module.signing = nacl_signing_module

sys.modules.setdefault("nacl", nacl_module)
sys.modules.setdefault("nacl.signing", nacl_signing_module)


def _make_collector() -> StorageCollector:
    with (
        patch("repeater.data_acquisition.storage_collector.SQLiteHandler"),
        patch("repeater.data_acquisition.storage_collector.RRDToolHandler"),
        patch("repeater.data_acquisition.hardware_stats.HardwareStatsCollector"),
    ):
        collector = StorageCollector(
            config={"storage": {"storage_dir": "/tmp/openhop_repeater_test"}}
        )

    # Stop any real stats-broadcast thread started during construction so the tests
    # drive the loop deterministically.
    collector._stats_stop_event.set()
    if collector._stats_thread is not None:
        collector._stats_thread.join(timeout=1)
    collector._stats_stop_event = threading.Event()
    collector._stats_thread = None

    collector.sqlite_handler = MagicMock()
    collector.sqlite_handler.get_packet_stats.return_value = {"total_packets": 1}
    collector.websocket_available = True
    collector.websocket_broadcast_packet = MagicMock()
    collector.websocket_broadcast_stats = MagicMock()
    collector.websocket_has_connected_clients = MagicMock(return_value=True)
    collector.repeater_handler = SimpleNamespace(start_time=100.0)
    return collector


def test_publish_packet_sync_broadcasts_packet_event_not_stats():
    # The per-packet path must stay fast: it broadcasts the packet event but never
    # runs the heavy aggregate or the stats broadcast (those moved to the loop).
    collector = _make_collector()

    collector._publish_packet_sync({"type": 1, "transmitted": True}, skip_mqtt=False)

    assert collector.websocket_broadcast_packet.call_count == 1
    assert collector.sqlite_handler.get_packet_stats.call_count == 0
    assert collector.websocket_broadcast_stats.call_count == 0


def test_broadcast_stats_once_queries_and_broadcasts():
    collector = _make_collector()

    collector._broadcast_stats_once()

    collector.sqlite_handler.get_packet_stats.assert_called_once_with(hours=24)
    assert collector.websocket_broadcast_stats.call_count == 1
    payload = collector.websocket_broadcast_stats.call_args.args[0]
    assert payload["packet_stats"] == {"total_packets": 1}
    assert "uptime_seconds" in payload["system_stats"]


def test_stats_loop_broadcasts_when_clients_connected():
    collector = _make_collector()
    collector._broadcast_stats_once = MagicMock()
    collector.websocket_has_connected_clients = MagicMock(return_value=True)
    # Drive exactly one iteration: wait() returns False (proceed) then True (exit).
    collector._stats_stop_event = MagicMock()
    collector._stats_stop_event.wait.side_effect = [False, True]

    collector._stats_broadcast_loop()

    assert collector._broadcast_stats_once.call_count == 1


def test_stats_loop_skips_when_no_clients():
    collector = _make_collector()
    collector._broadcast_stats_once = MagicMock()
    collector.websocket_has_connected_clients = MagicMock(return_value=False)
    collector._stats_stop_event = MagicMock()
    collector._stats_stop_event.wait.side_effect = [False, True]

    collector._stats_broadcast_loop()

    assert collector._broadcast_stats_once.call_count == 0
