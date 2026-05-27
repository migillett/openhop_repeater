import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repeater.sensors import SensorBase, SensorManager, SensorRegistry
from repeater.sensors import ens210 as ens210_module
from repeater.sensors import ina219 as ina219_module
from repeater.sensors import lafvin_ups_3s as lafvin_ups_3s_module
from repeater.sensors import shtc3 as shtc3_module
from repeater.sensors import waveshare_ups_d as waveshare_ups_d_module
from repeater.sensors import waveshare_ups_e as waveshare_ups_e_module
from repeater.sensors.ens210 import ENS210Sensor
from repeater.sensors.ina219 import INA219Sensor
from repeater.sensors.lafvin_ups_3s import LafvinUps3sSensor
from repeater.sensors.shtc3 import SHTC3Sensor
from repeater.sensors.waveshare_ups_d import WaveshareUpsDSensor
from repeater.sensors.waveshare_ups_e import WaveshareUpsESensor


class _TestRegistry(SensorRegistry):
    _factories = {}


class _DummySensor(SensorBase):
    sensor_type = "dummy"

    def _read(self):
        return {"value": self.settings.get("value", 0)}


class _FailingSensor(SensorBase):
    sensor_type = "failing"

    def _read(self):
        raise RuntimeError("boom")


_TestRegistry.register("dummy", _DummySensor)
_TestRegistry.register("failing", _FailingSensor)


def _install_fake_smbus2(monkeypatch, bus_class, *, i2c_msg=None):
    module = types.ModuleType("smbus2")
    setattr(module, "SMBus", bus_class)
    if i2c_msg is not None:
        setattr(module, "i2c_msg", i2c_msg)
    monkeypatch.setitem(sys.modules, "smbus2", module)
    monkeypatch.setattr(SensorBase, "ensure_python_modules", lambda self, modules: True)
    return module


def _swap_word(value):
    return ((value & 0xFF) << 8) | ((value >> 8) & 0xFF)


def _load_hardware_stats_sensor_module(monkeypatch):
    fake_data_acquisition = types.ModuleType("repeater.data_acquisition")
    fake_data_acquisition.__path__ = []
    fake_hardware_stats = types.ModuleType("repeater.data_acquisition.hardware_stats")
    setattr(fake_hardware_stats, "HardwareStatsCollector", MagicMock)

    monkeypatch.setitem(sys.modules, "repeater.data_acquisition", fake_data_acquisition)
    monkeypatch.setitem(
        sys.modules, "repeater.data_acquisition.hardware_stats", fake_hardware_stats
    )

    module_name = "repeater.sensors._hardware_stats_test"
    module_path = Path(__file__).resolve().parents[1] / "repeater" / "sensors" / "hardware_stats.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_sensor_manager_summary_reflects_sensor_config():
    config = {
        "sensors": {
            "enabled": True,
            "poll_interval_seconds": 12,
            "auto_install_packages": True,
            "definitions": [
                {"name": "demo", "type": "dummy", "settings": {"value": 7}},
                {"name": "skip-me"},
                "not-a-dict",
            ],
        }
    }

    with patch.object(SensorManager, "_load_sensor_module", return_value=None):
        manager = SensorManager(config, registry=_TestRegistry)

    definitions = manager._get_sensor_definitions()
    summary = manager.get_summary()

    assert definitions[0]["auto_install_packages"] is True
    assert summary["enabled"] is True
    assert summary["poll_interval_seconds"] == 12.0
    assert summary["configured"] == 2
    assert summary["loaded"] == 1


def test_sensor_manager_loads_and_reads_sensors_without_stopping_on_failure():
    config = {
        "sensors": {
            "enabled": True,
            "definitions": [
                {"name": "good", "type": "dummy", "settings": {"value": 11}},
                {"name": "bad", "type": "failing"},
                {"name": "skipped", "type": "missing", "enabled": True},
            ],
        }
    }

    with patch.object(SensorManager, "_load_sensor_module", return_value=None):
        manager = SensorManager(config, registry=_TestRegistry)

    summary = manager.get_summary()
    assert summary["configured"] == 3
    assert summary["loaded"] == 2

    readings = manager.read_all()
    assert len(readings) == 2
    assert readings[0]["ok"] is True
    assert readings[0]["data"]["value"] == 11
    assert readings[1]["ok"] is False
    assert readings[1]["error"].startswith("RuntimeError:")


def test_hardware_stats_sensor_reads_from_collector(monkeypatch):
    hardware_stats_module = _load_hardware_stats_sensor_module(monkeypatch)
    collector = MagicMock()
    collector.get_stats.return_value = {"cpu": {"usage_percent": 42.0}}

    with patch.object(hardware_stats_module, "HardwareStatsCollector", return_value=collector):
        reading = hardware_stats_module.HardwareStatsSensor("host").read()

    assert reading["ok"] is True
    assert reading["data"] == {"cpu": {"usage_percent": 42.0}}


def test_ina219_sensor_reads_voltage_current_and_power(monkeypatch):
    class _Bus:
        def __init__(self, bus_number):
            self.bus_number = bus_number

        def write_word_data(self, addr, register, value):
            return None

        def read_word_data(self, addr, register):
            values = {
                ina219_module._REG_BUS_VOLTAGE: 0x5DC0,
                ina219_module._REG_SHUNT_VOLTAGE: 1234,
                ina219_module._REG_CURRENT: 1000,
                ina219_module._REG_POWER: 500,
            }
            return _swap_word(values[register])

        def close(self):
            return None

    _install_fake_smbus2(monkeypatch, _Bus)

    reading = INA219Sensor("power_monitor").read()

    assert reading["ok"] is True
    assert reading["data"]["bus_voltage_v"] == 12.0
    assert reading["data"]["shunt_voltage_v"] == 0.01234
    assert reading["data"]["current_ma"] == pytest.approx(61.04, abs=0.01)
    assert reading["data"]["power_mw"] == pytest.approx(610.35, abs=0.01)


def test_shtc3_sensor_reads_temperature_and_humidity(monkeypatch):
    class _Msg:
        def __init__(self, addr, payload, *, is_read=False):
            self.addr = addr
            self.payload = list(payload)
            self.is_read = is_read

        def __iter__(self):
            return iter(self.payload)

    class _I2CMsg:
        @staticmethod
        def write(addr, payload):
            return _Msg(addr, payload)

        @staticmethod
        def read(addr, length):
            return _Msg(addr, [0] * length, is_read=True)

    class _Bus:
        def __init__(self, bus_number):
            self.bus_number = bus_number

        def i2c_rdwr(self, msg):
            if getattr(msg, "is_read", False):
                msg.payload[:] = [0x66, 0x66, 0x00, 0x80, 0x00, 0x00]

        def close(self):
            return None

    _install_fake_smbus2(monkeypatch, _Bus, i2c_msg=_I2CMsg)
    monkeypatch.setattr(shtc3_module.time, "sleep", lambda *_args, **_kwargs: None)

    reading = SHTC3Sensor("ambient").read()

    assert reading["ok"] is True
    assert reading["data"] == {
        "temperature_c": 25.0,
        "temperature_f": 77.0,
        "humidity_pct": 50.0,
    }


def test_ens210_sensor_reads_temperature_and_humidity(monkeypatch):
    class _Bus:
        def __init__(self, bus_number):
            self.bus_number = bus_number

        def write_byte_data(self, addr, register, value):
            return None

        def read_i2c_block_data(self, addr, register, length):
            if register == ens210_module._REG_T_VAL:
                return [0x4A, 0x49, 0x01]
            if register == ens210_module._REG_H_VAL:
                return [0x00, 0x6E, 0x01]
            raise AssertionError(f"unexpected register: {register}")

        def close(self):
            return None

    _install_fake_smbus2(monkeypatch, _Bus)
    monkeypatch.setattr(ens210_module.time, "sleep", lambda *_args, **_kwargs: None)

    reading = ENS210Sensor("ambient").read()

    assert reading["ok"] is True
    assert reading["data"] == {
        "temperature_c": 20.01,
        "humidity_pct": 55.0,
    }


def test_waveshare_ups_d_sensor_reads_battery_state(monkeypatch):
    class _Bus:
        def __init__(self, bus_number):
            self.bus_number = bus_number

        def write_i2c_block_data(self, addr, register, data):
            return None

        def read_i2c_block_data(self, addr, register, length):
            values = {
                waveshare_ups_d_module._REG_BUS: [0x1F, 0x40],
                waveshare_ups_d_module._REG_SHUNT: [0x00, 0x64],
                waveshare_ups_d_module._REG_CURRENT: [0xFC, 0x18],
                waveshare_ups_d_module._REG_POWER: [0x00, 0x64],
            }
            return values[register]

        def close(self):
            return None

    _install_fake_smbus2(monkeypatch, _Bus)
    monkeypatch.setattr(waveshare_ups_d_module.time, "sleep", lambda *_args, **_kwargs: None)

    reading = WaveshareUpsDSensor("battery").read()

    assert reading["ok"] is True
    assert reading["data"]["bus_voltage_v"] == 4.0
    assert reading["data"]["battery_percent"] == 85
    assert reading["data"]["charge_state"] == "charging"
    assert reading["data"]["current_ma"] == pytest.approx(-152.4, abs=0.1)
    assert reading["data"]["power_mw"] == pytest.approx(304.8, abs=0.1)


def test_waveshare_ups_e_sensor_reads_pack_state(monkeypatch):
    class _Bus:
        def __init__(self, bus_number):
            self.bus_number = bus_number

        def read_i2c_block_data(self, addr, register, length):
            values = {
                waveshare_ups_e_module._REG_STATUS: [waveshare_ups_e_module._FLAG_CHARGING],
                waveshare_ups_e_module._REG_VBUS: [0xA0, 0x0F, 0x2C, 0x01, 0x58, 0x1B],
                waveshare_ups_e_module._REG_BATT: [
                    0x80,
                    0x3E,
                    0xFA,
                    0x00,
                    0x4E,
                    0x00,
                    0x98,
                    0x08,
                    0x2D,
                    0x00,
                    0x5A,
                    0x00,
                ],
                waveshare_ups_e_module._REG_CELLS: [0x80, 0x0C, 0x6C, 0x0C, 0x1C, 0x0C, 0x76, 0x0C],
            }
            return values[register]

        def close(self):
            return None

    _install_fake_smbus2(monkeypatch, _Bus)

    reading = WaveshareUpsESensor("battery").read()

    assert reading["ok"] is True
    assert reading["data"]["charge_state"] == "charging"
    assert reading["data"]["battery_voltage_mv"] == 16000
    assert reading["data"]["battery_percent"] == 78
    assert reading["data"]["remaining_capacity_mah"] == 2200
    assert reading["data"]["cell_voltages_mv"] == [3200, 3180, 3100, 3190]
    assert reading["data"]["low_cell_warning"] is True
    assert reading["data"]["time_to_full_min"] == 90
    assert "time_to_empty_min" not in reading["data"]


def test_lafvin_pack_voltage_to_percent_piecewise_bounds():
    assert lafvin_ups_3s_module._pack_voltage_to_percent(12.6) == 100
    assert lafvin_ups_3s_module._pack_voltage_to_percent(12.0) == 85
    assert lafvin_ups_3s_module._pack_voltage_to_percent(11.4) == 60
    assert lafvin_ups_3s_module._pack_voltage_to_percent(11.1) == 39
    assert lafvin_ups_3s_module._pack_voltage_to_percent(10.5) == 15
    assert lafvin_ups_3s_module._pack_voltage_to_percent(9.0) == 0
    assert lafvin_ups_3s_module._pack_voltage_to_percent(8.5) == 0


def test_lafvin_sensor_handles_missing_dependency(monkeypatch):
    monkeypatch.setattr(
        SensorBase,
        "ensure_python_modules",
        lambda self, modules: False,
    )

    sensor = LafvinUps3sSensor("battery")
    reading = sensor.read()

    assert sensor.available is False
    assert reading["ok"] is False
    assert "not available" in reading["error"]


def test_lafvin_sensor_reads_pack_state(monkeypatch):
    class _Bus:
        def __init__(self, bus_number):
            self.bus_number = bus_number

        def write_i2c_block_data(self, addr, register, data):
            return None

        def read_i2c_block_data(self, addr, register, length):
            values = {
                lafvin_ups_3s_module._REG_BUS: [0x1F, 0x40],
                lafvin_ups_3s_module._REG_SHUNT: [0x00, 0x64],
                lafvin_ups_3s_module._REG_CURRENT: [0xFC, 0x18],
                lafvin_ups_3s_module._REG_POWER: [0x00, 0x64],
            }
            return values[register]

        def close(self):
            return None

    _install_fake_smbus2(monkeypatch, _Bus)
    monkeypatch.setattr(lafvin_ups_3s_module.time, "sleep", lambda *_args, **_kwargs: None)

    reading = LafvinUps3sSensor("battery").read()

    assert reading["ok"] is True
    assert reading["data"]["bus_voltage_v"] == 4.0
    assert reading["data"]["battery_percent"] == 0
    assert reading["data"]["charge_state"] == "charging"
    assert reading["data"]["current_ma"] == pytest.approx(-152.6, abs=0.2)
    assert reading["data"]["power_mw"] == pytest.approx(305.2, abs=0.2)


def test_lafvin_sensor_read_wraps_bus_failures(monkeypatch):
    class _BrokenBus:
        def __init__(self, bus_number):
            self.bus_number = bus_number

        def write_i2c_block_data(self, addr, register, data):
            return None

        def read_i2c_block_data(self, addr, register, length):
            raise RuntimeError("i2c broken")

        def close(self):
            return None

    _install_fake_smbus2(monkeypatch, _BrokenBus)
    monkeypatch.setattr(lafvin_ups_3s_module.time, "sleep", lambda *_args, **_kwargs: None)

    reading = LafvinUps3sSensor("battery").read()
    assert reading["ok"] is False
    assert "LAFVIN UPS 3S read failed" in reading["error"]
