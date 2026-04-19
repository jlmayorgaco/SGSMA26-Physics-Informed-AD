from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_engineering.normalization import calculate_angular_speed, detect_signal_family
from src.data_engineering.normalization_baselines import robust_center, robust_trimmed_mean


def test_calculate_angular_speed_returns_same_length() -> None:
    idx = pd.Index([0.0, 0.1, 0.2, 0.3], name="TIMESTAMP")
    angles = pd.Series([0.0, 10.0, 20.0, 30.0], index=idx)
    speed = calculate_angular_speed(angles, idx)
    assert len(speed) == len(angles)


def test_calculate_angular_speed_handles_nan_gap() -> None:
    idx = pd.Index([0.0, 0.1, 0.2, 0.3], name="TIMESTAMP")
    angles = pd.Series([0.0, np.nan, 20.0, 30.0], index=idx)
    speed = calculate_angular_speed(angles, idx)
    assert len(speed) == len(angles)
    assert np.isfinite(speed).all()


def test_calculate_angular_speed_empty_input() -> None:
    idx = pd.Index([], dtype=float, name="TIMESTAMP")
    angles = pd.Series([], dtype=float, index=idx)
    speed = calculate_angular_speed(angles, idx)
    assert speed.size == 0


def test_detect_signal_family_voltage_angle() -> None:
    assert detect_signal_family("BUS10_VA_ANG") == "voltage_angle"


def test_detect_signal_family_current_angle() -> None:
    assert detect_signal_family("BUS10_IA_ANG") == "current_angle"


def test_detect_signal_family_voltage_mag() -> None:
    assert detect_signal_family("BUS10_VA_MAG") == "voltage_mag"


def test_detect_signal_family_current_mag() -> None:
    assert detect_signal_family("BUS10_IA_MAG") == "current_mag"


def test_detect_signal_family_frequency() -> None:
    assert detect_signal_family("BUS10_Freq") == "frequency"


def test_detect_signal_family_rocof() -> None:
    assert detect_signal_family("BUS10_ROCOF") == "rocof"


def test_robust_trimmed_mean_empty_returns_fallback() -> None:
    value, method = robust_trimmed_mean(pd.Series([], dtype=float))
    assert value == 1.0
    assert method == "fallback_constant"


def test_robust_center_empty_returns_fallback_zero() -> None:
    value, method = robust_center(pd.Series([], dtype=float))
    assert value == 0.0
    assert method == "fallback_zero"


def test_robust_center_prefers_median_for_long_series() -> None:
    values = pd.Series(np.arange(60, dtype=float))
    value, method = robust_center(values)
    assert method == "median"
    assert np.isclose(value, np.median(values.to_numpy(dtype=float)))
