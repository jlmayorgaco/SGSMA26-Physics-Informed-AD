from __future__ import annotations

import numpy as np
import pandas as pd

from src.estimation.estimation_metrics import (
    compute_estimation_metrics,
    mae,
    rel_rmse,
    rmse,
    safe_corr,
)


def test_rmse_zero_for_equal_vectors() -> None:
    x = np.array([1.0, 2.0])
    assert np.isclose(rmse(x, x), 0.0)


def test_mae_zero_for_equal_vectors() -> None:
    x = np.array([1.0, 2.0])
    assert np.isclose(mae(x, x), 0.0)


def test_rel_rmse_finite() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.1, 2.1, 3.1])
    assert np.isfinite(rel_rmse(a, b))


def test_safe_corr_nan_on_constant_inputs() -> None:
    x = np.array([1.0, 1.0, 1.0])
    assert np.isnan(safe_corr(x, x))


def test_compute_estimation_metrics_returns_expected_columns() -> None:
    bus = "10"
    cols = {
        "TIMESTAMP": [0.0],
        "DATA_PRESENT": [1],
        "Event": [0],
    }
    for suffix in ["VA_ANG", "VA_MAG", "VB_ANG", "VB_MAG", "VC_ANG", "VC_MAG", "IA_ANG", "IA_MAG", "IB_ANG", "IB_MAG", "IC_ANG", "IC_MAG", "Freq", "ROCOF"]:
        cols[f"BUS{bus}_{suffix}"] = [1.0]
    sim_df = pd.DataFrame(cols)
    est_df = sim_df.copy()
    long_df, _, _ = compute_estimation_metrics({bus: sim_df}, {bus: est_df})
    assert {"bus_id", "signal", "rmse", "mae", "corr"}.issubset(long_df.columns)
