from types import SimpleNamespace
from unittest.mock import MagicMock, patch


from repeater.handler_helpers.mesh_cli import MeshCLI


def _base_config():
    return {
        "version": "3.2.1",
        "repeater": {
            "name": "node-a",
            "mode": "forward",
            "latitude": 1.2,
            "longitude": 3.4,
            "airtime_factor": 1.1,
            "advert_interval_minutes": 120,
            "flood_advert_interval_hours": 24,
            "max_flood_hops": 20,
            "rx_delay_base": 0.2,
            "tx_delay_factor": 1.3,
            "direct_tx_delay_factor": 0.6,
            "multi_acks": 2,
            "interference_threshold": -115,
            "agc_reset_interval": 8,
        },
        "radio": {
            "frequency": 915000000,
            "bandwidth": 125000,
            "spreading_factor": 7,
            "coding_rate": 5,
            "tx_power": 22,
        },
        "security": {"guest_password": "guest", "allow_read_only": True},
    }


def _cfg_mgr(save_ok=True, err=None):
    return SimpleNamespace(
        save_to_file=MagicMock(return_value=(save_ok, err)),
        live_update_daemon=MagicMock(),
    )


def test_handle_command_admin_and_prefix_behavior():
    cli = MeshCLI("/tmp/cfg.yaml", _base_config(), _cfg_mgr())

    assert cli.handle_command(b"a", "help", is_admin=False) == "Error: Admin permission required"
    assert cli.handle_command(b"a", "12|help set", is_admin=True).startswith("12|")


def test_help_routing_and_basic_unknown_paths():
    cli = MeshCLI("/tmp/cfg.yaml", _base_config(), _cfg_mgr(), enable_regions=False)

    assert "pyMC CLI Commands" in cli._route_command("help")
    assert "No detailed help" in cli._route_command("help nope")
    assert cli._route_command("start ota").startswith("Error:")
    assert cli._route_command("sensor read").startswith("Error:")
    assert cli._route_command("gps on").startswith("Error:")
    assert cli._route_command("stats-foo").startswith("Error:")
    assert cli._route_command("region load x").startswith("Error: Region commands not available")
    assert cli._route_command("unknown") == "Unknown command"


def test_cmd_advert_branches_and_success_schedule():
    cli = MeshCLI("/tmp/cfg.yaml", _base_config(), _cfg_mgr(), send_advert_callback=MagicMock())

    # No callback configured.
    cli_no_cb = MeshCLI("/tmp/cfg.yaml", _base_config(), _cfg_mgr(), send_advert_callback=None)
    assert cli_no_cb._cmd_advert().startswith("Error: Advert functionality")

    # Callback present but no event loop.
    cli._event_loop = None
    assert cli._cmd_advert() == "Error: Event loop not available"

    # Event loop available/running and schedule succeeds.
    fake_loop = SimpleNamespace(is_running=lambda: True)
    cli._event_loop = fake_loop

    with patch(
        "asyncio.run_coroutine_threadsafe", side_effect=lambda coro, _loop: coro.close()
    ) as run_ts:
        out = cli._cmd_advert()

    assert out == "OK - Advert sent"
    run_ts.assert_called_once()


def test_cmd_password_save_success_failure_and_exception():
    cfg = _base_config()
    ok_mgr = _cfg_mgr(save_ok=True)
    cli_ok = MeshCLI("/tmp/cfg.yaml", cfg, ok_mgr)

    assert cli_ok._cmd_password("password   ") == "Error: Password cannot be empty"
    assert cli_ok._cmd_password("password newpw") == "password now: newpw"
    ok_mgr.live_update_daemon.assert_called_once_with(["security"])

    bad_mgr = _cfg_mgr(save_ok=False, err="disk")
    cli_bad = MeshCLI("/tmp/cfg.yaml", _base_config(), bad_mgr)
    assert "Failed to save config" in cli_bad._cmd_password("password x")

    ex_mgr = SimpleNamespace(
        save_to_file=MagicMock(side_effect=RuntimeError("boom")),
        live_update_daemon=MagicMock(),
    )
    cli_ex = MeshCLI("/tmp/cfg.yaml", _base_config(), ex_mgr)
    assert cli_ex._cmd_password("password x") == "Error: Failed to save password"


def test_cmd_get_public_key_and_neighbor_branches():
    cli = MeshCLI("/tmp/cfg.yaml", _base_config(), _cfg_mgr())

    assert cli._cmd_get("public.key") == "Error: Identity not available"

    cli.identity = SimpleNamespace(get_public_key=lambda: b"\x01" * 32)
    assert cli._cmd_get("public.key") == "> " + (b"\x01" * 32).hex()

    cli.identity = SimpleNamespace(get_public_key=MagicMock(side_effect=RuntimeError("bad")))
    assert cli._cmd_get("public.key").startswith("Error:")

    # neighbors: no storage
    assert cli._cmd_neighbors() == "Error: Storage not available"

    # neighbors: empty, filtered empty, then formatted output
    storage = SimpleNamespace(get_neighbors=lambda: {})
    cli.storage_handler = storage
    assert cli._cmd_neighbors() == "No neighbors discovered yet"

    storage.get_neighbors = lambda: {
        "aa": {"is_repeater": False, "zero_hop": False, "last_seen": 1}
    }
    assert "No repeaters or zero hop" in cli._cmd_neighbors()

    storage.get_neighbors = lambda: {
        "abcdef12feed": {"is_repeater": True, "zero_hop": False, "last_seen": 10, "snr": 4.9},
        "11223344aabb": {"is_repeater": False, "zero_hop": True, "last_seen": 20, "snr": 1.2},
    }
    with patch("time.time", return_value=30):
        out = cli._cmd_neighbors()

    assert "abcdef12:20:4" in out
    assert "11223344:10:1" in out

    cli.storage_handler = SimpleNamespace(
        get_neighbors=MagicMock(side_effect=RuntimeError("db fail"))
    )
    assert cli._cmd_neighbors().startswith("Error:")


def test_cmd_set_updates_and_validation_errors():
    cfg = _base_config()
    mgr = _cfg_mgr()
    cli = MeshCLI("/tmp/cfg.yaml", cfg, mgr)

    assert cli._cmd_set("af 2.5") == "OK"
    assert cfg["repeater"]["airtime_factor"] == 2.5

    assert cli._cmd_set("name node-z") == "OK"
    assert cfg["repeater"]["node_name"] == "node-z"

    assert cli._cmd_set("repeat off").endswith("OFF")
    assert cfg["repeater"]["mode"] == "monitor"

    assert cli._cmd_set("radio 900000000 250000 9 6").startswith("OK")
    assert cfg["radio"]["frequency"] == 900000000.0

    assert cli._cmd_set("freq 868000000").startswith("OK")
    assert cli._cmd_set("tx 17") == "OK"
    assert cli._cmd_set("guest.password g") == "OK"
    assert cli._cmd_set("allow.read.only off") == "OK"

    assert cli._cmd_set("advert.interval 59").startswith("Error: interval range")
    assert cli._cmd_set("flood.advert.interval 2").startswith("Error: interval range")
    assert cli._cmd_set("flood.max 100") == "Error: max 64"
    assert cli._cmd_set("rxdelay -1") == "Error: cannot be negative"
    assert cli._cmd_set("txdelay -1") == "Error: cannot be negative"
    assert cli._cmd_set("direct.txdelay -1") == "Error: cannot be negative"

    assert cli._cmd_set("agc.reset.interval 10") == "OK - interval rounded to 8"
    assert cli._cmd_set("bad") == "Error: Missing value"
    assert cli._cmd_set("tx nope").startswith("Error: invalid value")
    assert cli._cmd_set("unknown.key 1") == "unknown config: unknown.key"


def test_misc_commands_and_routes():
    cli = MeshCLI("/tmp/cfg.yaml", _base_config(), _cfg_mgr(), enable_regions=True)

    assert cli._cmd_region("region").startswith("Error:")
    assert cli._cmd_region("region load us").startswith("Error:")
    assert cli._cmd_region("region save").startswith("Error:")
    assert cli._cmd_region("region remove x").startswith("Error:")
    assert cli._cmd_region("region unknown").startswith("Err -")

    assert cli._cmd_setperm("setperm") == "Err - bad params"
    assert cli._cmd_setperm("setperm abc zz") == "Err - invalid permissions"
    assert cli._cmd_setperm("setperm abc 2").startswith("Error:")

    assert cli._cmd_tempradio("tempradio 1 2 3").startswith("Error: Expected")
    assert cli._cmd_tempradio("tempradio 299 125 7 5 10") == "Error: invalid frequency"
    assert cli._cmd_tempradio("tempradio 915 6 7 5 10") == "Error: invalid bandwidth"
    assert cli._cmd_tempradio("tempradio 915 125 4 5 10") == "Error: invalid spreading factor"
    assert cli._cmd_tempradio("tempradio 915 125 7 9 10") == "Error: invalid coding rate"
    assert cli._cmd_tempradio("tempradio 915 125 7 5 0") == "Error: invalid timeout"
    assert cli._cmd_tempradio("tempradio 915 125 7 5 nope") == "Error, invalid params"
    assert cli._cmd_tempradio("tempradio 915 125 7 5 10").startswith("Error:")

    assert cli._cmd_neighbor_remove("neighbor.remove   ") == "ERR: Missing pubkey"
    assert cli._cmd_neighbor_remove("neighbor.remove abc").startswith("Error:")

    assert cli._cmd_log("log start").startswith("Error:")
    assert cli._cmd_log("log stop").startswith("Error:")
    assert cli._cmd_log("log erase").startswith("Error:")
    assert cli._cmd_log("log") == "Error: Use journalctl to view logs"
    assert cli._cmd_log("log whatever") == "Unknown log command"
