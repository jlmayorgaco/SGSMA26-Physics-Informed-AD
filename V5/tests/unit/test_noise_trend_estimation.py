from __future__ import annotations

import numpy as np

from src.calibration.trend_estimation import estimate_trend, safe_odd_window


def test_safe_odd_window_small_series() -> None:
    assert safe_odd_window(3, 101) == 3


def test_safe_odd_window_even_target() -> None:
    assert safe_odd_window(20, 10) == 9


def test_estimate_trend_short_series_returns_median_short_series() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    trend, method = estimate_trend(y, "voltage_mag")
    assert method == "median_short_series"
    assert np.allclose(trend, np.nanmedian(y))


def test_estimate_trend_magnitude_family_returns_rolling_median() -> None:
    y = np.linspace(1.0, 2.0, 40)
    _, method = estimate_trend(y, "voltage_mag")
    assert method.startswith("rolling_median_w")


def test_estimate_trend_other_family_returns_savgol() -> None:
    y = np.linspace(1.0, 2.0, 40)
    _, method = estimate_trend(y, "angle_voltage")
    assert method.startswith("savgol_w")


def test_estimate_trend_preserves_length() -> None:
    y = np.random.RandomState(0).normal(size=50)
    trend, _ = estimate_trend(y, "rocof")
    assert len(trend) == len(y)
