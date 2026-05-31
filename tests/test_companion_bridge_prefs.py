"""Tests for RepeaterCompanionBridge prefs JSON round-trip (bytes fields)."""

import pytest

from pymc_core import LocalIdentity

from repeater.companion.bridge import RepeaterCompanionBridge, _prefs_bytes_from_json


@pytest.fixture
def identity():
    return LocalIdentity()


def test_prefs_bytes_from_json_round_trip():
    assert _prefs_bytes_from_json("") == b""
    assert _prefs_bytes_from_json("00") == b"\x00"
    key = bytes(range(16))
    assert _prefs_bytes_from_json(key.hex()) == key
    assert _prefs_bytes_from_json(bytearray(key)) == key
    assert _prefs_bytes_from_json(key) == key
    assert _prefs_bytes_from_json("not-hex") == b""


def test_load_prefs_restores_default_scope_key_as_bytes(identity):
    """Hex strings from SQLite JSON must become bytes (not str) on NodePrefs."""

    class FakeSqlite:
        def companion_load_prefs(self, companion_hash: str):
            return {
                "default_scope_name": "region1",
                "default_scope_key": bytes(range(16)).hex(),
            }

        def companion_save_prefs(self, companion_hash: str, prefs: dict) -> bool:
            return True

    async def inject(pkt, wait_for_ack=False):
        return True

    bridge = RepeaterCompanionBridge(
        identity,
        inject,
        sqlite_handler=FakeSqlite(),
        companion_hash="testhash",
        node_name="bootname",
    )
    assert bridge.prefs.default_scope_name == "region1"
    assert isinstance(bridge.prefs.default_scope_key, bytes)
    assert bridge.prefs.default_scope_key == bytes(range(16))
    scope = bridge.get_default_flood_scope()
    assert scope is not None
    assert scope[0] == "region1"
    assert scope[1] == bytes(range(16))
