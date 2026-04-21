"""Estimation metrics for m4 estimated vs simulation signals."""

from __future__ import annotations

import numpy as np
import pandas as pd


PLOT_SUFFIXES = [
    "VA_ANG",
    "VA_MAG",
    "VB_ANG",
    "VB_MAG",
    "VC_ANG",
    "VC_MAG",
    "IA_ANG",
    "IA_MAG",
    "IB_ANG",
    "IB_MAG",
    "IC_ANG",
    "IC_MAG",
    "Freq",
    "ROCOF",
]
PMU_BUSES = ["39", "29", "10", "22", "19", "2", "5", "6"]


def raw_col(bus_id: str, suffix: str) -> str:
    return f"BUS{bus_id}_{suffix}"


def wrap_deg(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    return ((values + 180.0) % 360.0) - 180.0


def complex_phase_deg(z: np.ndarray) -> np.ndarray:
    return wrap_deg(np.rad2deg(np.angle(np.asarray(z, dtype=complex))))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    if len(av) < 2 or len(bv) < 2:
        return np.nan
    if np.std(av) < 1e-12 or np.std(bv) < 1e-12:
        return np.nan
    return float(np.corrcoef(av, bv)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((av - bv) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(av - bv)))


def rel_rmse(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.std(np.asarray(a, dtype=float)) + 1e-12)
    return float(rmse(a, b) / denom)


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return wrap_deg(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def _is_angle_suffix(raw_suffix: str) -> bool:
    return raw_suffix.endswith("_ANG")


def compute_estimation_metrics(simulation_dfs: dict[str, pd.DataFrame], estimated_dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for bus_id in sorted(simulation_dfs.keys(), key=lambda x: int(x)):
        sim_df = simulation_dfs[bus_id]
        est_df = estimated_dfs[bus_id]
        for suffix in PLOT_SUFFIXES:
            col = raw_col(bus_id, suffix)
            if col not in sim_df.columns or col not in est_df.columns:
                continue
            sim = sim_df[col].to_numpy(dtype=float)
            est = est_df[col].to_numpy(dtype=float)
            if _is_angle_suffix(suffix):
                diff = angle_diff_deg(sim, est)
                rows.append(
                    {
                        "bus_id": bus_id,
                        "signal": suffix,
                        "source": "pmu_passthrough" if bus_id in PMU_BUSES else "ybus_estimated",
                        "rmse": float(np.sqrt(np.mean(diff**2))),
                        "mae": float(np.mean(np.abs(diff))),
                        "relative_rmse": np.nan,
                        "corr": safe_corr(sim, est),
                        "max_abs_error": float(np.max(np.abs(diff))),
                    }
                )
            else:
                rows.append(
                    {
                        "bus_id": bus_id,
                        "signal": suffix,
                        "source": "pmu_passthrough" if bus_id in PMU_BUSES else "ybus_estimated",
                        "rmse": rmse(sim, est),
                        "mae": mae(sim, est),
                        "relative_rmse": rel_rmse(sim, est),
                        "corr": safe_corr(sim, est),
                        "max_abs_error": float(np.max(np.abs(sim - est))),
                    }
                )
    metrics_long = pd.DataFrame(rows)
    summary_bus = (
        metrics_long.groupby("bus_id")[["rmse", "mae", "relative_rmse", "corr"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("bus_id", key=lambda s: s.astype(int))
    )
    summary_signal = (
        metrics_long.groupby("signal")[["rmse", "mae", "relative_rmse", "corr"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("signal")
    )
    return metrics_long, summary_bus, summary_signal
