from __future__ import annotations

from src.calibration.calibration_metrics import signal_family_from_key


def test_va_mag_voltage_mag() -> None:
    assert signal_family_from_key("VA_mag") == "voltage_mag"


def test_ia_mag_current_mag() -> None:
    assert signal_family_from_key("IA_mag") == "current_mag"


def test_frequency_family() -> None:
    assert signal_family_from_key("Frequency") == "frequency"


def test_rocof_family() -> None:
    assert signal_family_from_key("ROCOF") == "frequency"


def test_va_ang_delta_angle_voltage() -> None:
    assert signal_family_from_key("VA_ang_delta") == "angle_voltage"


def test_ia_ang_angle_current() -> None:
    assert signal_family_from_key("IA_ang") == "angle_current"


def test_unknown_other() -> None:
    assert signal_family_from_key("mystery") == "other"
