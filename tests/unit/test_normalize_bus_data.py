from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_engineering.normalization import normalize_bus_data


def _build_bus_df(event_values: list[int]) -> pd.DataFrame:
    idx = pd.Index([0.0, 0.1, 0.2, 0.3], name="TIMESTAMP")
    return pd.DataFrame(
        {
            "BUS10_VA_ANG": [-60.0, -60.2, -60.4, -60.6],
            "BUS10_IA_ANG": [-70.0, -70.2, -70.4, -70.6],
            "BUS10_VA_MAG": [100.0, 100.0, 120.0, 140.0],
            "BUS10_IA_MAG": [10.0, 10.0, 15.0, 20.0],
            "BUS10_Freq": [60.0, 60.0, 59.8, 59.7],
            "BUS10_ROCOF": [0.1, 0.1, 0.4, 0.5],
            "BUS10_MISC": [1.0, 2.0, 3.0, 4.0],
            "DATA_PRESENT": [1, 1, 1, 1],
            "Event": event_values,
        },
        index=idx,
    )


def _run_normalize(event_values: list[int], global_event_values: list[int] | None = None):
    bus_df = _build_bus_df(event_values)
    bus_data = {"Bus10": bus_df}
    event_df = pd.DataFrame(
        {"Bus10": global_event_values if global_event_values is not None else event_values},
        index=bus_df.index,
    )
    return normalize_bus_data(bus_data, event_df)


def test_normalize_bus_data_preserves_data_present_and_event() -> None:
    normalized, _ = _run_normalize([0, 0, 1, 1])
    out = normalized["Bus10"]
    assert "DATA_PRESENT" in out.columns
    assert "Event" in out.columns
    assert out["DATA_PRESENT"].tolist() == [1, 1, 1, 1]
    assert out["Event"].tolist() == [0, 0, 1, 1]


def test_voltage_angles_become_ang_speed() -> None:
    normalized, _ = _run_normalize([0, 0, 1, 1])
    out = normalized["Bus10"]
    assert "BUS10_VA_ANG_SPEED_RAD_S" in out.columns
    assert "BUS10_VA_ANG" not in out.columns


def test_current_angles_become_ang_speed() -> None:
    normalized, _ = _run_normalize([0, 0, 1, 1])
    out = normalized["Bus10"]
    assert "BUS10_IA_ANG_SPEED_RAD_S" in out.columns
    assert "BUS10_IA_ANG" not in out.columns


def test_voltage_magnitude_divides_by_baseline() -> None:
    normalized, _ = _run_normalize([0, 0, 1, 1])
    va = normalized["Bus10"]["BUS10_VA_MAG"].to_numpy(dtype=float)
    assert np.isclose(va[0], 1.0)
    assert np.isclose(va[2], 1.2)


def test_current_magnitude_divides_by_baseline() -> None:
    normalized, _ = _run_normalize([0, 0, 1, 1])
    ia = normalized["Bus10"]["BUS10_IA_MAG"].to_numpy(dtype=float)
    assert np.isclose(ia[0], 1.0)
    assert np.isclose(ia[2], 1.5)


def test_frequency_divides_by_baseline() -> None:
    normalized, _ = _run_normalize([0, 0, 1, 1])
    freq = normalized["Bus10"]["BUS10_Freq"].to_numpy(dtype=float)
    assert np.isclose(freq[0], 1.0)
    assert np.isclose(freq[2], 59.8 / 60.0)


def test_rocof_is_centered() -> None:
    normalized, _ = _run_normalize([0, 0, 1, 1])
    rocof = normalized["Bus10"]["BUS10_ROCOF"].to_numpy(dtype=float)
    assert np.isclose(rocof[0], 0.0)
    assert np.isclose(rocof[2], 0.3)


def test_fallback_to_full_series_when_no_event0_exists() -> None:
    _, baselines = _run_normalize([1, 1, 1, 1], global_event_values=[1, 1, 1, 1])
    assert (baselines["baseline_source"] == "full_series_fallback").all()


def test_baseline_df_contains_expected_columns() -> None:
    _, baselines = _run_normalize([0, 0, 1, 1])
    expected = {
        "bus_id",
        "raw_signal",
        "output_signal",
        "signal_family",
        "transform",
        "baseline_method",
        "baseline_value",
        "n_normal_samples",
        "baseline_source",
        "notes",
    }
    assert expected.issubset(set(baselines.columns))
