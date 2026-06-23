from types import SimpleNamespace

from repeater.config import BaselineCrcCounterRadio


def test_baseline_crc_counter_radio_reports_delta_from_initial_raw_count():
    raw = SimpleNamespace(crc_error_count=20_000, frequency=869_618_000)
    radio = BaselineCrcCounterRadio(raw)

    assert radio.frequency == 869_618_000
    assert radio.crc_error_count == 0

    raw.crc_error_count = 20_003

    assert radio.crc_error_count == 3


def test_baseline_crc_counter_radio_handles_delayed_modem_counter():
    raw = SimpleNamespace(crc_error_count=0)
    radio = BaselineCrcCounterRadio(raw)

    assert radio.crc_error_count == 0

    raw.crc_error_count = 145
    assert radio.crc_error_count == 0

    raw.crc_error_count = 148
    assert radio.crc_error_count == 3
