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
            "DATA_PRESENT": [1, 1, 1, 1],
            "Event": event_values,
        },
        index=idx,
    )


def _run_normalize(
    event_values: list[int],
    feature_mode: str,
    global_event_values: list[int] | None = None,
):
    bus_df = _build_bus_df(event_values)
    bus_data = {"Bus10": bus_df}
    event_df = pd.DataFrame(
        {"Bus10": global_event_values if global_event_values is not None else event_values},
        index=bus_df.index,
    )
    return normalize_bus_data(bus_data, event_df, feature_mode=feature_mode), bus_df


def test_augment_mode_preserves_raw_angle_columns() -> None:
    (normalized, _), _ = _run_normalize([0, 0, 1, 1], "augment_angles")
    out = normalized["Bus10"]
    assert "BUS10_VA_ANG" in out.columns
    assert "BUS10_IA_ANG" in out.columns


def test_augment_mode_preserves_raw_magnitude_columns() -> None:
    (normalized, _), _ = _run_normalize([0, 0, 1, 1], "augment_angles")
    out = normalized["Bus10"]
    assert "BUS10_VA_MAG" in out.columns
    assert "BUS10_IA_MAG" in out.columns


def test_augment_mode_preserves_raw_frequency_column() -> None:
    (normalized, _), _ = _run_normalize([0, 0, 1, 1], "augment_angles")
    assert "BUS10_Freq" in normalized["Bus10"].columns


def test_augment_mode_preserves_raw_rocof_column() -> None:
    (normalized, _), _ = _run_normalize([0, 0, 1, 1], "augment_angles")
    assert "BUS10_ROCOF" in normalized["Bus10"].columns


def test_augment_mode_adds_angle_feature_columns() -> None:
    (normalized, _), _ = _run_normalize([0, 0, 1, 1], "augment_angles")
    cols = set(normalized["Bus10"].columns)
    assert {"BUS10_VA_ANG_SPEED_RAD_S", "BUS10_VA_ANG_SIN", "BUS10_VA_ANG_COS", "BUS10_VA_ANG_DEV_DEG"}.issubset(cols)


def test_augment_mode_adds_mag_pu_columns() -> None:
    (normalized, _), _ = _run_normalize([0, 0, 1, 1], "augment_angles")
    cols = set(normalized["Bus10"].columns)
    assert {"BUS10_VA_MAG_PU", "BUS10_IA_MAG_PU"}.issubset(cols)


def test_augment_mode_adds_mag_dev_pu_columns() -> None:
    (normalized, _), _ = _run_normalize([0, 0, 1, 1], "augment_angles")
    cols = set(normalized["Bus10"].columns)
    assert {"BUS10_VA_MAG_DEV_PU", "BUS10_IA_MAG_DEV_PU"}.issubset(cols)


def test_augment_mode_adds_freq_pu_column() -> None:
    (normalized, _), _ = _run_normalize([0, 0, 1, 1], "augment_angles")
    assert "BUS10_Freq_PU" in normalized["Bus10"].columns


def test_augment_mode_adds_freq_dev_hz_column() -> None:
    (normalized, _), _ = _run_normalize([0, 0, 1, 1], "augment_angles")
    assert "BUS10_Freq_DEV_HZ" in normalized["Bus10"].columns


def test_augment_mode_adds_rocof_centered_column() -> None:
    (normalized, _), _ = _run_normalize([0, 0, 1, 1], "augment_angles")
    assert "BUS10_ROCOF_CENTERED" in normalized["Bus10"].columns


def test_augment_mode_does_not_overwrite_raw_values() -> None:
    (normalized, _), raw_df = _run_normalize([0, 0, 1, 1], "augment_angles")
    out = normalized["Bus10"]
    for col in ["BUS10_VA_ANG", "BUS10_VA_MAG", "BUS10_IA_MAG", "BUS10_Freq", "BUS10_ROCOF"]:
        assert np.allclose(out[col].to_numpy(dtype=float), raw_df[col].to_numpy(dtype=float), equal_nan=True)


def test_legacy_mode_still_matches_expected_behavior() -> None:
    (normalized, _), _ = _run_normalize([0, 0, 1, 1], "legacy_replace_angles")
    out = normalized["Bus10"]
    assert "BUS10_VA_ANG" not in out.columns
    assert "BUS10_VA_ANG_SPEED_RAD_S" in out.columns
    assert np.isclose(out["BUS10_VA_MAG"].iloc[0], 1.0)


def test_fallback_to_full_series_when_no_event0_exists() -> None:
    ( _, baselines), _ = _run_normalize([1, 1, 1, 1], "legacy_replace_angles", global_event_values=[1, 1, 1, 1])
    assert (baselines["baseline_source"] == "full_series_fallback").all()
