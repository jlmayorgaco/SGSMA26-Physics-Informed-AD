from __future__ import annotations

import math

import numpy as np

from src.estimation.metrics import mae, rel_rmse, rmse, safe_corr


def test_rmse_zero_for_identical_arrays() -> None:
    arr = np.array([1.0, 2.0, 3.0])
    assert rmse(arr, arr) == 0.0


def test_mae_zero_for_identical_arrays() -> None:
    arr = np.array([5.0, -3.0, 0.0])
    assert mae(arr, arr) == 0.0


def test_rel_rmse_safe_for_near_constant_reference() -> None:
    ref = np.array([10.0, 10.0, 10.0 + 1e-9])
    pred = np.array([10.0, 10.0, 10.0])
    value = rel_rmse(ref, pred)
    assert math.isfinite(value)
    assert value >= 0.0


def test_safe_corr_nan_for_constant_vector() -> None:
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 2.0, 3.0])
    assert math.isnan(safe_corr(a, b))
