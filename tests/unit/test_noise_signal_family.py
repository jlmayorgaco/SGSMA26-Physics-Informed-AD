from __future__ import annotations

from src.calibration.noise_profiler import signal_family


def test_voltage_mag_family_detection() -> None:
    assert signal_family("BUS10_VA_MAG") == "voltage_mag"


def test_current_mag_family_detection() -> None:
    assert signal_family("BUS10_IA_MAG") == "current_mag"


def test_angle_voltage_family_detection() -> None:
    assert signal_family("BUS10_VA_ANG") == "angle_voltage"


def test_angle_current_family_detection() -> None:
    assert signal_family("BUS10_IA_ANG") == "angle_current"


def test_frequency_family_detection() -> None:
    assert signal_family("BUS10_Freq") == "frequency"


def test_rocof_family_detection() -> None:
    assert signal_family("BUS10_ROCOF") == "rocof"


def test_unknown_family_detection() -> None:
    assert signal_family("BUS10_MISC") == "other"
