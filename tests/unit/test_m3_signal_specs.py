from __future__ import annotations

from src.calibration.signal_specs import build_signal_specs


def test_signal_spec_count_gt_zero() -> None:
    assert len(build_signal_specs()) > 0


def test_expected_keys_present_in_every_spec() -> None:
    required = {"signal_key", "raw_suffix", "sim_source", "support_status", "raw_representation", "recommended", "notes"}
    for spec in build_signal_specs():
        assert required.issubset(set(spec.keys()))


def test_voltage_mag_specs_supported_direct() -> None:
    specs = [s for s in build_signal_specs() if s["signal_key"].startswith("V") and s["signal_key"].endswith("_mag")]
    assert specs and all(s["support_status"] == "supported_direct" for s in specs)


def test_current_mag_specs_current_mag_selected() -> None:
    specs = [s for s in build_signal_specs() if s["signal_key"].startswith("I") and s["signal_key"].endswith("_mag")]
    assert specs and all(s["sim_source"] == "current_mag_selected" for s in specs)


def test_frequency_spec_supported_derived() -> None:
    spec = next(s for s in build_signal_specs() if s["signal_key"] == "Frequency")
    assert spec["support_status"] == "supported_derived"


def test_rocof_spec_supported_derived() -> None:
    spec = next(s for s in build_signal_specs() if s["signal_key"] == "ROCOF")
    assert spec["support_status"] == "supported_derived"


def test_voltage_angle_delta_experimental_relative_only() -> None:
    specs = [s for s in build_signal_specs() if s["signal_key"].endswith("_ang_delta")]
    assert specs and all(s["support_status"] == "experimental_relative_only" for s in specs)


def test_current_angle_unsupported_raw_absolute() -> None:
    specs = [s for s in build_signal_specs() if s["signal_key"].startswith("I") and s["signal_key"].endswith("_ang")]
    assert specs and all(s["support_status"] == "unsupported_raw_absolute" for s in specs)
