from __future__ import annotations

from src.simulation.event0_artifacts import suffix_to_family


def test_suffix_to_family_maps_expected_values() -> None:
    assert suffix_to_family("VA_MAG") == "voltage_mag"
    assert suffix_to_family("IA_MAG") == "current_mag"
    assert suffix_to_family("Freq") == "frequency"
    assert suffix_to_family("ROCOF") == "rocof"
    assert suffix_to_family("VA_ANG") == "angle_voltage"
    assert suffix_to_family("IA_ANG") == "angle_current"


def test_suffix_to_family_unknown() -> None:
    assert suffix_to_family("XYZ") == "other"
