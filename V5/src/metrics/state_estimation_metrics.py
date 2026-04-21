"""Metrics for topology-aware PMU state estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wrap_deg(values: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=float) + 180.0) % 360.0) - 180.0


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def compute_state_estimation_metrics(
    timestamps: np.ndarray,
    bus_order: list[str],
    v_est_pu: np.ndarray,
    v_true_pu: np.ndarray,
    pmu_buses: set[str],
) -> tuple[pd.DataFrame, dict]:
    """Compute per-bus and aggregate metrics for |V| and angle."""
    t = np.asarray(timestamps, dtype=float)
    est = np.asarray(v_est_pu, dtype=complex)
    true = np.asarray(v_true_pu, dtype=complex)
    if est.shape != true.shape:
        raise ValueError(f"Shape mismatch est={est.shape} true={true.shape}")

    complex_err = np.abs(est - true)
    rows: list[dict] = []
    for bi, bus in enumerate(bus_order):
        est_mag = np.abs(est[:, bi])
        true_mag = np.abs(true[:, bi])
        est_ang = np.rad2deg(np.angle(est[:, bi]))
        true_ang = np.rad2deg(np.angle(true[:, bi]))
        diff_ang = _wrap_deg(est_ang - true_ang)
        rows.append(
            {
                "BUS": bus,
                "IS_PMU_BUS": bus in pmu_buses,
                "RMSE_V_MAG": _rmse(est_mag, true_mag),
                "MAE_V_MAG": _mae(est_mag, true_mag),
                "RMSE_ANG_DEG": float(np.sqrt(np.mean(diff_ang**2))),
                "MAE_ANG_DEG": float(np.mean(np.abs(diff_ang))),
                "MEAN_ABS_COMPLEX_ERROR": float(np.mean(complex_err[:, bi])),
                "NOTES": "",
            }
        )
    per_bus = pd.DataFrame(rows).sort_values("BUS", key=lambda s: s.str.extract(r"(\d+)").fillna(0).astype(int)[0])

    mag_err = np.abs(np.abs(est) - np.abs(true))
    ang_err = np.abs(_wrap_deg(np.rad2deg(np.angle(est)) - np.rad2deg(np.angle(true))))
    pmu_mask = per_bus["IS_PMU_BUS"]
    global_metrics = {
        "timestamp_count": int(len(t)),
        "bus_count": int(len(bus_order)),
        "rmse_v_mag_all": float(np.sqrt(np.mean(mag_err**2))),
        "mae_v_mag_all": float(np.mean(mag_err)),
        "rmse_ang_all": float(np.sqrt(np.mean(ang_err**2))),
        "mae_ang_all": float(np.mean(ang_err)),
        "rmse_v_mag_pmu_buses": float(per_bus.loc[pmu_mask, "RMSE_V_MAG"].mean() if pmu_mask.any() else np.nan),
        "rmse_v_mag_nonpmu_buses": float(per_bus.loc[~pmu_mask, "RMSE_V_MAG"].mean() if (~pmu_mask).any() else np.nan),
        "mae_v_mag_pmu_buses": float(per_bus.loc[pmu_mask, "MAE_V_MAG"].mean() if pmu_mask.any() else np.nan),
        "mae_v_mag_nonpmu_buses": float(per_bus.loc[~pmu_mask, "MAE_V_MAG"].mean() if (~pmu_mask).any() else np.nan),
        "rmse_ang_pmu_buses": float(per_bus.loc[pmu_mask, "RMSE_ANG_DEG"].mean() if pmu_mask.any() else np.nan),
        "rmse_ang_nonpmu_buses": float(per_bus.loc[~pmu_mask, "RMSE_ANG_DEG"].mean() if (~pmu_mask).any() else np.nan),
    }
    global_metrics["rmse_mag_pu_global"] = global_metrics["rmse_v_mag_all"]
    global_metrics["mae_mag_pu_global"] = global_metrics["mae_v_mag_all"]
    global_metrics["rmse_ang_deg_global"] = global_metrics["rmse_ang_all"]
    global_metrics["mae_ang_deg_global"] = global_metrics["mae_ang_all"]
    return per_bus, global_metrics


def residual_diagnostics(diagnostics: list[dict]) -> pd.DataFrame:
    """Convert diagnostics rows into DataFrame."""
    if not diagnostics:
        return pd.DataFrame(columns=["timestamp", "measurement_count", "residual_norm"])
    return pd.DataFrame(diagnostics)
