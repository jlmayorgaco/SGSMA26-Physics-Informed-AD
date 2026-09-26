from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_engineering.angle_features import build_angle_feature_block


def _angle_series() -> tuple[pd.Series, pd.Series, pd.Index]:
    idx = pd.Index([0.0, 0.1, 0.2, 0.3], name="TIMESTAMP")
    series = pd.Series([-60.0, -60.2, -60.4, -60.6], index=idx)
    normal_series = pd.Series([-60.0, -60.2], index=idx[:2])
    return series, normal_series, idx


def test_preserve_raw_angle_output_exists() -> None:
    series, normal_series, idx = _angle_series()
    block, _ = build_angle_feature_block(series, normal_series, "BUS10_VA_ANG", idx, "augment_angles")
    assert "BUS10_VA_ANG" in block.columns
    assert np.allclose(block["BUS10_VA_ANG"].to_numpy(dtype=float), series.to_numpy(dtype=float))


def test_add_speed_sin_cos_dev_columns() -> None:
    series, normal_series, idx = _angle_series()
    block, _ = build_angle_feature_block(series, normal_series, "BUS10_VA_ANG", idx, "augment_angles")
    assert "BUS10_VA_ANG_SPEED_RAD_S" in block.columns
    assert "BUS10_VA_ANG_SIN" in block.columns
    assert "BUS10_VA_ANG_COS" in block.columns
    assert "BUS10_VA_ANG_DEV_DEG" in block.columns


def test_dev_deg_is_wrapped() -> None:
    series, normal_series, idx = _angle_series()
    block, _ = build_angle_feature_block(series, normal_series, "BUS10_VA_ANG", idx, "augment_angles")
    vals = block["BUS10_VA_ANG_DEV_DEG"].to_numpy(dtype=float)
    assert np.all(vals >= -180.0)
    assert np.all(vals < 180.0)


def test_baseline_rows_include_preserve_raw_and_derived_outputs() -> None:
    series, normal_series, idx = _angle_series()
    _, rows = build_angle_feature_block(series, normal_series, "BUS10_VA_ANG", idx, "augment_angles")
    by_output = {row["output_signal"]: row["transform"] for row in rows}
    assert by_output["BUS10_VA_ANG"] == "preserve_raw"
    assert by_output["BUS10_VA_ANG_SPEED_RAD_S"] == "angle_to_angular_speed_rad_s"
    assert by_output["BUS10_VA_ANG_SIN"] == "sin_deg"
    assert by_output["BUS10_VA_ANG_COS"] == "cos_deg"
    assert by_output["BUS10_VA_ANG_DEV_DEG"] == "wrap_relative_to_circular_mean"
