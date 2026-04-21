"""Metrics helpers for M8 dynamic benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.topology import bus_sort_key


def _wrap_deg(values: np.ndarray) -> np.ndarray:
    return ((np.asarray(values, dtype=float) + 180.0) % 360.0) - 180.0


def compute_dynamic_tracking_metrics(
    timestamps: np.ndarray,
    est: np.ndarray,
    truth: np.ndarray,
    selected_bus_indices: list[int],
) -> dict[str, float]:
    """Compute simple dynamic-tracking scores from derivatives."""
    if len(selected_bus_indices) == 0 or len(timestamps) < 3:
        return {
            "dvdt_corr_mean": float("nan"),
            "dangdt_corr_mean": float("nan"),
            "event_tracking_lag_frames": float("nan"),
        }
    t = np.asarray(timestamps, dtype=float)
    corr_v: list[float] = []
    corr_a: list[float] = []
    lag_scores: list[float] = []
    for bi in selected_bus_indices:
        e_mag = np.abs(est[:, bi])
        r_mag = np.abs(truth[:, bi])
        e_ang = np.rad2deg(np.angle(est[:, bi]))
        r_ang = np.rad2deg(np.angle(truth[:, bi]))
        e_dmag = np.gradient(e_mag, t)
        r_dmag = np.gradient(r_mag, t)
        e_dang = np.gradient(_wrap_deg(e_ang), t)
        r_dang = np.gradient(_wrap_deg(r_ang), t)
        if np.std(e_dmag) > 1e-12 and np.std(r_dmag) > 1e-12:
            corr_v.append(float(np.corrcoef(e_dmag, r_dmag)[0, 1]))
        if np.std(e_dang) > 1e-12 and np.std(r_dang) > 1e-12:
            corr_a.append(float(np.corrcoef(e_dang, r_dang)[0, 1]))
        # event lag via cross-correlation on d|V|/dt
        x = e_dmag - np.mean(e_dmag)
        y = r_dmag - np.mean(r_dmag)
        if np.std(x) > 1e-12 and np.std(y) > 1e-12:
            cc = np.correlate(x, y, mode="full")
            lag_scores.append(float(abs(np.argmax(cc) - (len(x) - 1))))
    return {
        "dvdt_corr_mean": float(np.nanmean(corr_v)) if corr_v else float("nan"),
        "dangdt_corr_mean": float(np.nanmean(corr_a)) if corr_a else float("nan"),
        "event_tracking_lag_frames": float(np.nanmean(lag_scores)) if lag_scores else float("nan"),
    }


def compute_extended_per_bus_metrics(
    bus_order: list[str],
    est: np.ndarray,
    truth: np.ndarray,
    pmu_buses: set[str],
    generator_buses: set[str],
    electrical_distance_to_nearest_pmu: dict[str, float],
) -> pd.DataFrame:
    """Compute enriched per-bus metrics for M8 reporting."""
    rows: list[dict] = []
    for i, bus in enumerate(bus_order):
        e = np.asarray(est[:, i], dtype=complex)
        r = np.asarray(truth[:, i], dtype=complex)
        em = np.abs(e)
        rm = np.abs(r)
        ea = np.rad2deg(np.angle(e))
        ra = np.rad2deg(np.angle(r))
        dmag = em - rm
        dang = _wrap_deg(ea - ra)
        rows.append(
            {
                "BUS": bus,
                "PMU_BUS_FLAG": bool(bus in pmu_buses),
                "GENERATOR_BUS_FLAG": bool(bus in generator_buses),
                "RMSE_V_MAG": float(np.sqrt(np.mean(dmag**2))),
                "MAE_V_MAG": float(np.mean(np.abs(dmag))),
                "RMSE_ANG_DEG": float(np.sqrt(np.mean(dang**2))),
                "MAE_ANG_DEG": float(np.mean(np.abs(dang))),
                "MEAN_ABS_COMPLEX_ERROR": float(np.mean(np.abs(e - r))),
                "MAX_ABS_COMPLEX_ERROR": float(np.max(np.abs(e - r))),
                "STD_EST_V_MAG": float(np.std(em)),
                "STD_TRUE_V_MAG": float(np.std(rm)),
                "FROZEN_BUS_FLAG": bool(np.std(em) < 1e-5),
                "ELECTRICAL_DISTANCE_TO_NEAREST_PMU": float(electrical_distance_to_nearest_pmu.get(bus, np.nan)),
                "NOTES": "",
            }
        )
    return pd.DataFrame(rows).sort_values("BUS", key=lambda s: s.map(bus_sort_key))


def compute_group_metrics(
    per_bus_df: pd.DataFrame,
    near_distance_quantile: float = 0.35,
) -> dict[str, float]:
    """Aggregate group metrics from enriched per-bus table."""
    df = per_bus_df.copy()
    near_thr = float(df["ELECTRICAL_DISTANCE_TO_NEAREST_PMU"].quantile(near_distance_quantile))
    near = df["ELECTRICAL_DISTANCE_TO_NEAREST_PMU"] <= near_thr
    out = {
        "rmse_v_mag_pmu": float(df.loc[df["PMU_BUS_FLAG"], "RMSE_V_MAG"].mean()),
        "rmse_v_mag_nonpmu": float(df.loc[~df["PMU_BUS_FLAG"], "RMSE_V_MAG"].mean()),
        "rmse_ang_pmu": float(df.loc[df["PMU_BUS_FLAG"], "RMSE_ANG_DEG"].mean()),
        "rmse_ang_nonpmu": float(df.loc[~df["PMU_BUS_FLAG"], "RMSE_ANG_DEG"].mean()),
        "rmse_v_mag_generator": float(df.loc[df["GENERATOR_BUS_FLAG"], "RMSE_V_MAG"].mean()),
        "rmse_v_mag_load": float(df.loc[~df["GENERATOR_BUS_FLAG"], "RMSE_V_MAG"].mean()),
        "rmse_v_mag_near_pmu": float(df.loc[near, "RMSE_V_MAG"].mean()),
        "rmse_v_mag_far_pmu": float(df.loc[~near, "RMSE_V_MAG"].mean()),
        "frozen_bus_fraction": float(df["FROZEN_BUS_FLAG"].mean()),
    }
    return out
