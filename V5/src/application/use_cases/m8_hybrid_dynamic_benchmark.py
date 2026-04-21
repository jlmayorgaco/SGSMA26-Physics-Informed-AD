"""M8 hybrid dynamic state-estimation benchmark use case."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.domain.topology import bus_sort_key, canonical_bus_name
from src.estimation.dynamic_state_estimation.dynamic_wls_estimator import run_graph_regularized_dynamic_wls
from src.estimation.dynamic_state_estimation.ekf_hybrid_estimator import run_hybrid_ekf
from src.estimation.dynamic_state_estimation.electrical_distance import build_electrical_distance_from_ybus
from src.estimation.dynamic_state_estimation.graph_regularization import build_distance_weights, build_graph_laplacian
from src.estimation.dynamic_state_estimation.hybrid_state_definition import build_hybrid_state_layout
from src.estimation.dynamic_state_estimation.ukf_hybrid_estimator import run_hybrid_ukf
from src.estimation.state_estimation.estimator_variants import build_estimator_registry
from src.estimation.state_estimation.measurement_model import FrameMeasurements
from src.estimation.state_estimation.priors import build_loadflow_prior
from src.infrastructure.io.metadata_loader import load_pmu_metadata
from src.infrastructure.io.raw_network_loader import load_network_model_from_raw
from src.metrics.dynamic_state_estimation_metrics import (
    compute_dynamic_tracking_metrics,
    compute_extended_per_bus_metrics,
    compute_group_metrics,
)
from src.metrics.state_estimation_metrics import compute_state_estimation_metrics
from src.reports.dynamic_benchmark_report import write_m8_reports
from src.simulation.andes_ieee39_runner import run_andes_ieee39_truth
from src.visualization.dynamic_state_estimation_plots import generate_m8_benchmark_plots


LOGGER = logging.getLogger(__name__)


def _layout(out: Path) -> dict[str, Path]:
    paths = {
        "root": out,
        "config": out / "config",
        "metadata": out / "metadata",
        "truth": out / "truth",
        "estimated": out / "estimated",
        "metrics": out / "metrics",
        "plots": out / "plots",
        "report": out / "report",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _align_truth(truth_bus_ids: list[str], truth_v: np.ndarray, bus_order: list[str]) -> np.ndarray:
    pos = {canonical_bus_name(b): i for i, b in enumerate(truth_bus_ids)}
    by_num: dict[str, int] = {}
    for k, i in pos.items():
        num = "".join(ch for ch in k if ch.isdigit())
        if num and num not in by_num:
            by_num[num] = i
    out = np.zeros((truth_v.shape[0], len(bus_order)), dtype=complex)
    for j, bus in enumerate(bus_order):
        if bus in pos:
            out[:, j] = truth_v[:, pos[bus]]
            continue
        num = "".join(ch for ch in bus if ch.isdigit())
        out[:, j] = truth_v[:, by_num[num]] if num in by_num else (1.0 + 0.0j)
    return out


def _window_types(t: np.ndarray, truth_v: np.ndarray, pmu_idx: list[int]) -> list[str]:
    mag = np.abs(truth_v[:, pmu_idx]).mean(axis=1)
    d = np.abs(np.gradient(mag, np.asarray(t, dtype=float)))
    thr = float(np.quantile(d, 0.85))
    return ["event" if x >= thr else "quiet" for x in d]


def _build_frames(
    timestamps: np.ndarray,
    truth_v: np.ndarray,
    ybus: np.ndarray,
    bus_order: list[str],
    pmu_buses: list[str],
    missing_mode: bool = False,
) -> list[FrameMeasurements]:
    pos = {b: i for i, b in enumerate(bus_order)}
    i_truth = (ybus @ truth_v.T).T
    frames: list[FrameMeasurements] = []
    for i, ts in enumerate(np.asarray(timestamps, dtype=float)):
        v = {b: complex(truth_v[i, pos[b]]) for b in pmu_buses}
        cur = {b: complex(i_truth[i, pos[b]]) for b in pmu_buses}
        dropped: dict[str, str] = {}
        if missing_mode and len(pmu_buses) > 0:
            drop_bus = pmu_buses[i % len(pmu_buses)]
            v.pop(drop_bus, None)
            cur.pop(drop_bus, None)
            dropped[drop_bus] = "synthetic_missing"
        frames.append(
            FrameMeasurements(
                timestamp=float(ts),
                voltage_by_bus=v,
                current_by_bus=cur,
                data_present_count=len(v),
                expected_pmu_buses=pmu_buses,
                valid_pmu_buses=sorted(v.keys(), key=bus_sort_key),
                dropped_reasons=dropped,
            )
        )
    return frames


def _subset_frames(frames: list[FrameMeasurements], idx: np.ndarray) -> list[FrameMeasurements]:
    keep = set(int(i) for i in idx.tolist())
    return [f for i, f in enumerate(frames) if i in keep]


def _states_to_long(t: np.ndarray, bus_order: list[str], states: np.ndarray, pmu_buses: set[str], scenario_id: str, window_types: list[str]) -> pd.DataFrame:
    rows = []
    for i, ts in enumerate(np.asarray(t, dtype=float)):
        for j, bus in enumerate(bus_order):
            z = complex(states[i, j])
            rows.append(
                {
                    "TIMESTAMP": float(ts),
                    "BUS": bus,
                    "SCENARIO_ID": scenario_id,
                    "WINDOW_TYPE": window_types[i],
                    "IS_PMU_BUS": bus in pmu_buses,
                    "V_EST_REAL_PU": float(np.real(z)),
                    "V_EST_IMAG_PU": float(np.imag(z)),
                    "V_EST_MAG_PU": float(np.abs(z)),
                    "V_EST_ANG_DEG": float(np.rad2deg(np.angle(z))),
                }
            )
    return pd.DataFrame(rows)


def _truth_to_long(t: np.ndarray, bus_order: list[str], truth: np.ndarray, scenario_id: str, window_types: list[str]) -> pd.DataFrame:
    rows = []
    for i, ts in enumerate(np.asarray(t, dtype=float)):
        for j, bus in enumerate(bus_order):
            z = complex(truth[i, j])
            rows.append(
                {
                    "TIMESTAMP": float(ts),
                    "BUS": bus,
                    "SCENARIO_ID": scenario_id,
                    "WINDOW_TYPE": window_types[i],
                    "V_TRUE_REAL_PU": float(np.real(z)),
                    "V_TRUE_IMAG_PU": float(np.imag(z)),
                    "V_TRUE_MAG_PU": float(np.abs(z)),
                    "V_TRUE_ANG_DEG": float(np.rad2deg(np.angle(z))),
                }
            )
    return pd.DataFrame(rows)


def _load_m7_best_name(m7_results_path: Path | None) -> str:
    if m7_results_path is None or not m7_results_path.exists():
        return "PMU_VOLTAGE_CURRENT_SMOOTHED"
    try:
        payload = json.loads(m7_results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "PMU_VOLTAGE_CURRENT_SMOOTHED"
    return str(payload.get("best_estimator", {}).get("name", "PMU_VOLTAGE_CURRENT_SMOOTHED"))


def _overall_score(payload: dict[str, Any]) -> float:
    def _f(v: float, fallback: float) -> float:
        return float(v) if np.isfinite(float(v)) else float(fallback)

    g = payload["global_metrics"]
    r = payload["robustness_metrics"]
    d = payload["dynamic_metrics"]
    return (
        0.38 * _f(g.get("RMSE_V_MAG_NONPMU", np.nan), 1.0)
        + 0.22 * _f(g.get("RMSE_ANG_NONPMU", np.nan), 45.0) / 30.0
        + 0.12 * _f(r.get("pmu_voltage_fit_rmse", np.nan), 1.0)
        + 0.10 * _f(r.get("missing_data_degradation_ratio", np.nan), 5.0)
        + 0.10 * (1.0 - _f(d.get("dvdt_corr_mean", np.nan), 0.0))
        + 0.08 * _f(payload["runtime"].get("mean_runtime_per_frame_s", np.nan), 1.0)
    )


def _verdict(payload: dict[str, Any], thresholds: dict[str, float], m7_reference_nonpmu_rmse: float) -> dict[str, bool]:
    g = payload["global_metrics"]
    r = payload["robustness_metrics"]
    d = payload["dynamic_metrics"]
    return {
        "passes_truth_validation": bool(np.isfinite(g.get("RMSE_V_MAG_ALL", np.nan))),
        "uses_all_valid_pmus": bool(r.get("mean_pmus_used_clean", 0.0) >= thresholds["min_clean_pmu_usage"]),
        "nonpmu_buses_not_overfrozen": bool(r.get("frozen_nonpmu_fraction", 1.0) <= thresholds["max_frozen_nonpmu_fraction"]),
        "pmu_fit_is_good": bool(r.get("pmu_voltage_fit_rmse", np.inf) <= thresholds["max_pmu_fit_rmse"]),
        "dynamic_tracking_is_good": bool(d.get("dvdt_corr_mean", -1.0) >= thresholds["min_dvdt_corr"]),
        "stable_under_missing_data": bool(r.get("missing_data_degradation_ratio", np.inf) <= thresholds["max_missing_data_degradation"]),
        "numerically_stable": bool(r.get("solver_failure_rate", 1.0) <= thresholds["max_solver_failure_rate"]),
        "better_than_m7_reference": bool(g.get("RMSE_V_MAG_NONPMU", np.inf) < m7_reference_nonpmu_rmse),
        "better_than_prior_baseline": bool(g.get("RMSE_V_MAG_NONPMU", np.inf) < thresholds["prior_baseline_rmse_nonpmu"]),
        "overall_status": False,
    }


def run_m8_hybrid_dynamic_benchmark_use_case(
    raw_path: str | Path,
    pmu_location_path: str | Path,
    output_dir: str | Path,
    use_andes_truth: bool = True,
    scenario_set: str = "default",
    start_time: float | None = None,
    end_time: float | None = 20.0,
    stride: int = 1,
    estimators: list[str] | None = None,
    default_estimator: str = "M8_HYBRID_EKF",
    m7_results_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run M8 dynamic benchmark and export canonical artifact package."""
    if not use_andes_truth:
        raise ValueError("M8 benchmark requires ANDES truth (use_andes_truth=True).")
    out = Path(output_dir)
    paths = _layout(out)

    pmu_meta = load_pmu_metadata(pmu_location_path)
    network = load_network_model_from_raw(raw_path)
    pmu_buses = sorted({canonical_bus_name(e.get("bus_label_canonical", "")) for e in pmu_meta.get("pmu_map", [])}, key=bus_sort_key)
    generator_buses = sorted({canonical_bus_name(g.get("bus_label_canonical", "")) for g in pmu_meta.get("buses", []) if str(g.get("bus_type_name", "")).upper() in {"PV", "SWING"}}, key=bus_sort_key)
    # fallback to generator records in RAW-derived metadata if PV/SWING list is unexpectedly empty
    if not generator_buses:
        generator_buses = [b for b in network.bus_order if b in {"BUS30X1", "BUS31X1", "BUS32X1", "BUS33X1", "BUS34X1", "BUS35X1", "BUS36X1", "BUS37X1", "BUS38X1", "BUS39"}]

    prior = np.asarray(build_loadflow_prior(network.bus_order, pmu_meta), dtype=complex)
    layout = build_hybrid_state_layout(network.bus_order, generator_buses)
    dist, dist_meta = build_electrical_distance_from_ybus(network.ybus)
    weights = build_distance_weights(dist)
    graph_l = build_graph_laplacian(weights)

    truth = run_andes_ieee39_truth(tf=float(end_time or 20.0), tstep=1.0 / 30.0, stride=max(1, int(stride)))
    t = np.asarray(truth.timestamps, dtype=float)
    tv = np.asarray(truth.voltage_complex_pu, dtype=complex)
    if start_time is not None:
        keep = t >= float(start_time)
        t = t[keep]
        tv = tv[keep, :]
    truth_v = _align_truth(truth.bus_ids, tv, network.bus_order)

    pmu_idx = [network.bus_order.index(b) for b in pmu_buses]
    wtypes = _window_types(t, truth_v, pmu_idx)
    frames_all = _build_frames(t, truth_v, network.ybus, network.bus_order, pmu_buses, missing_mode=False)
    frames_missing = _build_frames(t, truth_v, network.ybus, network.bus_order, pmu_buses, missing_mode=True)

    idx_event = np.asarray([i for i, x in enumerate(wtypes) if x == "event"], dtype=int)
    idx_quiet = np.asarray([i for i, x in enumerate(wtypes) if x == "quiet"], dtype=int)
    scenarios = [
        {"id": "QUIET", "idx": idx_quiet if len(idx_quiet) > 0 else np.arange(len(t)), "frames": frames_all, "window_types": wtypes},
        {"id": "EVENT", "idx": idx_event if len(idx_event) > 0 else np.arange(len(t)), "frames": frames_all, "window_types": wtypes},
        {"id": "MISSING_DATA", "idx": np.arange(len(t)), "frames": frames_missing, "window_types": ["missing_data"] * len(t)},
        {"id": "MIXED", "idx": np.arange(len(t)), "frames": frames_all, "window_types": ["mixed"] * len(t)},
    ]

    truth_long = pd.concat(
        [
            _truth_to_long(t[s["idx"]], network.bus_order, truth_v[s["idx"], :], s["id"], [s["window_types"][i] for i in s["idx"]])
            for s in scenarios
        ],
        ignore_index=True,
    )
    truth_long.to_csv(paths["truth"] / "andes_truth_bus_states.csv", index=False)

    m7_best_name = _load_m7_best_name(Path(m7_results_path) if m7_results_path else None)
    m7_registry = build_estimator_registry()
    m7_best_name = m7_best_name if m7_best_name in m7_registry else "PMU_VOLTAGE_CURRENT_SMOOTHED"

    runner_map = {
        "M8_HYBRID_EKF": lambda fr, wt: run_hybrid_ekf(
            network=network,
            prior=prior,
            frames=fr,
            window_types=wt,
            layout=layout,
            graph_laplacian=graph_l,
        ),
        "M8_HYBRID_UKF": lambda fr, wt: run_hybrid_ukf(
            network=network,
            prior=prior,
            frames=fr,
            window_types=wt,
            layout=layout,
            graph_laplacian=graph_l,
        ),
        "M8_GRAPH_REGULARIZED_WLS_DYNAMIC_PRIOR": lambda fr, wt: run_graph_regularized_dynamic_wls(
            network=network,
            prior=prior,
            frames=fr,
            window_types=wt,
            graph_laplacian=graph_l,
        ),
        "M8_M7_BEST_REFERENCE": lambda fr, wt: m7_registry[m7_best_name].runner(network=network, prior=prior, frames=fr, window_types=wt),
    }
    if estimators:
        chosen = [e for e in estimators if e in runner_map]
    else:
        chosen = list(runner_map.keys())
    if not chosen:
        raise ValueError("No valid M8 estimators selected.")

    # distance to nearest PMU
    pmu_set = set(pmu_buses)
    nearest_dist = {}
    for i, bus in enumerate(network.bus_order):
        dvals = [float(dist[i, network.bus_order.index(pb)]) for pb in pmu_buses]
        nearest_dist[bus] = float(min(dvals)) if dvals else float("nan")

    all_global_rows: list[dict[str, Any]] = []
    all_per_bus: list[pd.DataFrame] = []
    all_scenario_rows: list[dict[str, Any]] = []
    all_dynamic_rows: list[dict[str, Any]] = []
    all_robust_rows: list[dict[str, Any]] = []
    estimator_payloads: dict[str, Any] = {}
    prior_baseline_nonpmu = np.nan
    m7_ref_nonpmu = np.nan

    for est_name in chosen:
        est_dir = paths["estimated"] / est_name
        est_dir.mkdir(parents=True, exist_ok=True)
        scenario_metrics: dict[str, Any] = {}
        diag_frames: list[pd.DataFrame] = []
        est_frames: list[pd.DataFrame] = []
        pmu_fit_rows: list[dict[str, Any]] = []
        per_bus_rows: list[pd.DataFrame] = []
        missing_nonpmu = np.nan
        run_times: list[float] = []

        for s in scenarios:
            idx = s["idx"]
            if len(idx) == 0:
                continue
            frames_sub = _subset_frames(s["frames"], idx)
            wsub = [s["window_types"][i] for i in idx]
            run = runner_map[est_name](frames_sub, wsub)
            states = np.asarray(run["states"], dtype=complex)
            truth_sub = truth_v[idx, :]

            per_bus_basic, gm_basic = compute_state_estimation_metrics(
                timestamps=t[idx],
                bus_order=network.bus_order,
                v_est_pu=states,
                v_true_pu=truth_sub,
                pmu_buses=pmu_set,
            )
            per_bus_ext = compute_extended_per_bus_metrics(
                bus_order=network.bus_order,
                est=states,
                truth=truth_sub,
                pmu_buses=pmu_set,
                generator_buses=set(generator_buses),
                electrical_distance_to_nearest_pmu=nearest_dist,
            )
            per_bus_ext["scenario_id"] = s["id"]
            per_bus_ext["estimator"] = est_name
            per_bus_rows.append(per_bus_ext)

            group = compute_group_metrics(per_bus_ext)
            dyn = compute_dynamic_tracking_metrics(timestamps=t[idx], est=states, truth=truth_sub, selected_bus_indices=[0, 4, 9, 18, 28, 38])
            gm = {
                "RMSE_V_MAG_ALL": float(gm_basic["rmse_v_mag_all"]),
                "MAE_V_MAG_ALL": float(gm_basic["mae_v_mag_all"]),
                "RMSE_ANG_ALL": float(gm_basic["rmse_ang_all"]),
                "MAE_ANG_ALL": float(gm_basic["mae_ang_all"]),
                "RMSE_V_MAG_NONPMU": float(gm_basic["rmse_v_mag_nonpmu_buses"]),
                "MAE_V_MAG_NONPMU": float(gm_basic["mae_v_mag_nonpmu_buses"]),
                "RMSE_ANG_NONPMU": float(gm_basic["rmse_ang_nonpmu_buses"]),
                "MAE_ANG_NONPMU": float(per_bus_ext.loc[~per_bus_ext["PMU_BUS_FLAG"], "MAE_ANG_DEG"].mean()),
                "MEAN_ABS_COMPLEX_ERROR": float(per_bus_ext["MEAN_ABS_COMPLEX_ERROR"].mean()),
                "MAX_ABS_COMPLEX_ERROR": float(per_bus_ext["MAX_ABS_COMPLEX_ERROR"].max()),
            }
            gm.update(group)
            gm.update(dyn)
            gm["scenario_id"] = s["id"]
            gm["estimator"] = est_name
            scenario_metrics[s["id"]] = gm
            all_scenario_rows.append(gm)
            run_times.append(float(run["runtime_s"]))
            if s["id"] == "MISSING_DATA":
                missing_nonpmu = float(gm["RMSE_V_MAG_NONPMU"])

            est_frames.append(_states_to_long(t[idx], network.bus_order, states, pmu_set, s["id"], wsub))
            ddf = pd.DataFrame(run["diagnostics"])
            if not ddf.empty:
                ddf["scenario_id"] = s["id"]
                ddf["estimator"] = est_name
            diag_frames.append(ddf)

            for ii, ts in enumerate(t[idx]):
                for bus in pmu_buses:
                    bj = network.bus_order.index(bus)
                    z_est = states[ii, bj]
                    z_true = truth_sub[ii, bj]
                    pmu_fit_rows.append(
                        {
                            "TIMESTAMP": float(ts),
                            "PMU_BUS": bus,
                            "scenario_id": s["id"],
                            "estimator": est_name,
                            "V_MEAS_MAG": float(np.abs(z_true)),
                            "V_EST_MAG": float(np.abs(z_est)),
                            "V_MEAS_ANG_DEG": float(np.rad2deg(np.angle(z_true))),
                            "V_EST_ANG_DEG": float(np.rad2deg(np.angle(z_est))),
                            "ABS_COMPLEX_ERROR": float(np.abs(z_est - z_true)),
                            "MAG_ERROR": float(abs(abs(z_est) - abs(z_true))),
                            "ANGLE_ERROR_DEG": float(abs(((np.rad2deg(np.angle(z_est)) - np.rad2deg(np.angle(z_true)) + 180.0) % 360.0) - 180.0)),
                        }
                    )

        est_long = pd.concat(est_frames, ignore_index=True)
        diag_df = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
        pmu_fit_df = pd.DataFrame(pmu_fit_rows)
        per_bus_df = pd.concat(per_bus_rows, ignore_index=True)
        est_long.to_csv(est_dir / "estimated_bus_states.csv", index=False)
        diag_df.to_csv(est_dir / "frame_diagnostics.csv", index=False)
        pmu_fit_df.to_csv(est_dir / "pmu_fit_diagnostics.csv", index=False)
        if not diag_df.empty:
            diag_df.to_csv(est_dir / "residual_diagnostics.csv", index=False)

        gdf = pd.DataFrame(list(scenario_metrics.values()))
        global_metrics = gdf.drop(columns=["scenario_id", "estimator"]).mean(numeric_only=True).to_dict()
        global_metrics["timestamp_count"] = int(len(t))
        global_metrics["bus_count"] = int(len(network.bus_order))
        global_metrics["pmu_bus_count"] = int(len(pmu_buses))

        mean_pmu_used = float(diag_df["n_pmus_used"].mean()) if ("n_pmus_used" in diag_df.columns and not diag_df.empty) else np.nan
        min_pmu_used = float(diag_df["n_pmus_used"].min()) if ("n_pmus_used" in diag_df.columns and not diag_df.empty) else np.nan
        fail_rate = float((diag_df.get("solver_status", pd.Series(dtype=str)) != "ok").mean()) if (not diag_df.empty and "solver_status" in diag_df.columns) else 0.0
        frozen_nonpmu = float(per_bus_df.loc[~per_bus_df["PMU_BUS_FLAG"], "FROZEN_BUS_FLAG"].mean())
        robust = {
            "pmu_voltage_fit_rmse": float(np.sqrt(np.mean(np.square(pmu_fit_df["MAG_ERROR"].to_numpy(float))))) if not pmu_fit_df.empty else np.nan,
            "pmu_angle_fit_rmse": float(np.sqrt(np.mean(np.square(pmu_fit_df["ANGLE_ERROR_DEG"].to_numpy(float))))) if not pmu_fit_df.empty else np.nan,
            "mean_pmus_used_clean": mean_pmu_used,
            "min_pmus_used_clean": min_pmu_used,
            "missing_data_degradation_ratio": float(missing_nonpmu / max(float(global_metrics.get("RMSE_V_MAG_NONPMU", np.nan)), 1e-9)),
            "solver_failure_rate": fail_rate,
            "divergence_count": int((diag_df.get("residual_norm", pd.Series(dtype=float)) > 0.25).sum()) if ("residual_norm" in diag_df.columns and not diag_df.empty) else 0,
            "ill_conditioned_frame_count": int((diag_df.get("matrix_condition_number", pd.Series(dtype=float)) > 1e12).sum()) if ("matrix_condition_number" in diag_df.columns and not diag_df.empty) else 0,
            "frozen_nonpmu_fraction": frozen_nonpmu,
        }
        dyn_df = gdf[["dvdt_corr_mean", "dangdt_corr_mean", "event_tracking_lag_frames"]].mean(numeric_only=True).to_dict()
        runtime = {
            "total_runtime_s": float(sum(run_times)),
            "mean_runtime_per_frame_s": float(sum(run_times) / max(len(t) * len(scenarios), 1)),
            "state_dimension": int(2 * len(network.bus_order) + 2 * len(generator_buses)),
            "measurement_dimension_mean": float(diag_df.get("measurement_count", pd.Series(dtype=float)).mean()) if ("measurement_count" in diag_df.columns and not diag_df.empty) else np.nan,
        }
        payload = {
            "config": {
                "estimator_name": est_name,
                "lambda_reg": 5e-2,
                "mu_reg": 1e-3,
                "alpha_graph": 3e-2,
                "scenario_set": scenario_set,
            },
            "runtime": runtime,
            "solver_stats": {
                "solver_failure_rate": robust["solver_failure_rate"],
                "divergence_count": robust["divergence_count"],
                "ill_conditioned_frame_count": robust["ill_conditioned_frame_count"],
            },
            "global_metrics": global_metrics,
            "group_metrics": compute_group_metrics(
                per_bus_df.sort_values("scenario_id")
                .groupby("BUS", as_index=False)
                .agg(
                    {
                        "PMU_BUS_FLAG": "first",
                        "GENERATOR_BUS_FLAG": "first",
                        "ELECTRICAL_DISTANCE_TO_NEAREST_PMU": "first",
                        "RMSE_V_MAG": "mean",
                        "RMSE_ANG_DEG": "mean",
                        "FROZEN_BUS_FLAG": "mean",
                    }
                )
            ),
            "per_bus_summary": {
                "row_count": int(len(per_bus_df)),
                "frozen_bus_count": int(per_bus_df["FROZEN_BUS_FLAG"].sum()),
            },
            "scenario_metrics": scenario_metrics,
            "pmu_fit_metrics": {
                "PMU_VOLTAGE_FIT_RMSE": robust["pmu_voltage_fit_rmse"],
                "PMU_VOLTAGE_FIT_MAE": float(pmu_fit_df["MAG_ERROR"].mean()) if not pmu_fit_df.empty else np.nan,
                "PMU_ANGLE_FIT_RMSE": robust["pmu_angle_fit_rmse"],
                "mean_pmus_used_per_frame": robust["mean_pmus_used_clean"],
                "min_pmus_used_per_frame": robust["min_pmus_used_clean"],
            },
            "dynamic_metrics": dyn_df,
            "robustness_metrics": robust,
            "diagnostics": {
                "estimated_csv": str(est_dir / "estimated_bus_states.csv"),
                "frame_diagnostics_csv": str(est_dir / "frame_diagnostics.csv"),
                "pmu_fit_csv": str(est_dir / "pmu_fit_diagnostics.csv"),
            },
        }
        estimator_payloads[est_name] = payload
        all_per_bus.append(per_bus_df)
        all_dynamic_rows.append({"estimator": est_name, **dyn_df})
        all_robust_rows.append({"estimator": est_name, **robust})
        all_global_rows.extend(
            [
                {"estimator": est_name, "metric": "RMSE_V_MAG_ALL", "value": payload["global_metrics"].get("RMSE_V_MAG_ALL", np.nan)},
                {"estimator": est_name, "metric": "RMSE_ANG_ALL", "value": payload["global_metrics"].get("RMSE_ANG_ALL", np.nan)},
                {"estimator": est_name, "metric": "RMSE_V_MAG_NONPMU", "value": payload["global_metrics"].get("RMSE_V_MAG_NONPMU", np.nan)},
                {"estimator": est_name, "metric": "RMSE_ANG_NONPMU", "value": payload["global_metrics"].get("RMSE_ANG_NONPMU", np.nan)},
            ]
        )
        if est_name == "M8_M7_BEST_REFERENCE":
            m7_ref_nonpmu = float(payload["global_metrics"].get("RMSE_V_MAG_NONPMU", np.nan))
        if est_name == "M8_GRAPH_REGULARIZED_WLS_DYNAMIC_PRIOR":
            prior_baseline_nonpmu = float(payload["global_metrics"].get("RMSE_V_MAG_NONPMU", np.nan))

    if np.isnan(m7_ref_nonpmu):
        m7_ref_nonpmu = float(np.nanmean([v["global_metrics"].get("RMSE_V_MAG_NONPMU", np.nan) for v in estimator_payloads.values()]))
    if np.isnan(prior_baseline_nonpmu):
        prior_baseline_nonpmu = float(np.nanmean([v["global_metrics"].get("RMSE_V_MAG_NONPMU", np.nan) for v in estimator_payloads.values()]))

    thresholds = {
        "min_clean_pmu_usage": 7.5,
        "max_frozen_nonpmu_fraction": 0.75,
        "max_pmu_fit_rmse": 0.05,
        "max_missing_data_degradation": 2.5,
        "max_solver_failure_rate": 0.25,
        "min_dvdt_corr": 0.1,
        "prior_baseline_rmse_nonpmu": prior_baseline_nonpmu,
    }
    for name, payload in estimator_payloads.items():
        verdict = _verdict(payload=payload, thresholds=thresholds, m7_reference_nonpmu_rmse=m7_ref_nonpmu)
        verdict["overall_status"] = bool(
            verdict["passes_truth_validation"]
            and verdict["uses_all_valid_pmus"]
            and verdict["pmu_fit_is_good"]
            and verdict["nonpmu_buses_not_overfrozen"]
            and verdict["stable_under_missing_data"]
            and verdict["numerically_stable"]
        )
        payload["verdict"] = verdict
        payload["overall_score"] = _overall_score(payload)

    ranking_overall = sorted([(k, v["overall_score"]) for k, v in estimator_payloads.items()], key=lambda x: x[1])
    ranking_rmse_mag_nonpmu = sorted([(k, v["global_metrics"].get("RMSE_V_MAG_NONPMU", np.inf)) for k, v in estimator_payloads.items()], key=lambda x: x[1])
    ranking_rmse_ang_nonpmu = sorted([(k, v["global_metrics"].get("RMSE_ANG_NONPMU", np.inf)) for k, v in estimator_payloads.items()], key=lambda x: x[1])
    ranking_dynamic = sorted([(k, -float(v["dynamic_metrics"].get("dvdt_corr_mean", -np.inf))) for k, v in estimator_payloads.items()], key=lambda x: x[1])
    best = ranking_overall[0][0]

    global_df = pd.DataFrame(all_global_rows)
    per_bus_df = pd.concat(all_per_bus, ignore_index=True) if all_per_bus else pd.DataFrame()
    scenario_df = pd.DataFrame(all_scenario_rows)
    dynamic_df = pd.DataFrame(all_dynamic_rows)
    robust_df = pd.DataFrame(all_robust_rows)
    runtime_df = pd.DataFrame(
        [
            {
                "estimator": k,
                "mean_runtime_per_frame_s": v["runtime"]["mean_runtime_per_frame_s"],
                "RMSE_V_MAG_NONPMU": v["global_metrics"].get("RMSE_V_MAG_NONPMU", np.nan),
            }
            for k, v in estimator_payloads.items()
        ]
    )
    ranking_df = pd.DataFrame([{"estimator": k, "overall_score": v["overall_score"]} for k, v in estimator_payloads.items()]).sort_values("overall_score")

    global_df.to_csv(paths["metrics"] / "global_metrics.csv", index=False)
    per_bus_df.to_csv(paths["metrics"] / "per_bus_metrics.csv", index=False)
    scenario_df.to_csv(paths["metrics"] / "scenario_metrics.csv", index=False)
    pd.DataFrame([{"estimator": k, **v.get("group_metrics", {})} for k, v in estimator_payloads.items()]).to_csv(paths["metrics"] / "group_metrics.csv", index=False)
    dynamic_df.to_csv(paths["metrics"] / "dynamic_metrics.csv", index=False)
    robust_df.to_csv(paths["metrics"] / "robustness_metrics.csv", index=False)
    ranking_df.to_csv(paths["metrics"] / "estimator_ranking.csv", index=False)

    # metadata/config exports
    (paths["metadata"] / "bus_index_map.json").write_text(json.dumps({b: i for i, b in enumerate(network.bus_order)}, indent=2), encoding="utf-8")
    (paths["metadata"] / "pmu_bus_mapping.json").write_text(json.dumps(pmu_buses, indent=2), encoding="utf-8")
    (paths["metadata"] / "network_summary.json").write_text(
        json.dumps({"bus_count": len(network.bus_order), "base_mva": network.base_mva, "generator_bus_count": len(generator_buses)}, indent=2),
        encoding="utf-8",
    )
    (paths["metadata"] / "scenario_manifest.json").write_text(
        json.dumps([{"scenario_id": s["id"], "frame_count": int(len(s["idx"]))} for s in scenarios], indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(dist, index=network.bus_order, columns=network.bus_order).to_csv(paths["metadata"] / "electrical_distance_matrix.csv")
    (paths["config"] / "run_config.json").write_text(
        json.dumps(
            {
                "raw_path": str(Path(raw_path)),
                "pmu_location_path": str(Path(pmu_location_path)),
                "output_dir": str(out),
                "scenario_set": scenario_set,
                "start_time": start_time,
                "end_time": end_time,
                "stride": stride,
                "estimators": chosen,
                "default_estimator": default_estimator,
                "m7_results_path": str(m7_results_path) if m7_results_path else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (paths["config"] / "estimator_configs.json").write_text(
        json.dumps({k: v["config"] for k, v in estimator_payloads.items()}, indent=2),
        encoding="utf-8",
    )
    (paths["config"] / "threshold_config.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    generate_m8_benchmark_plots(
        plots_dir=paths["plots"],
        global_df=global_df,
        per_bus_df=per_bus_df,
        scenario_df=scenario_df,
        runtime_df=runtime_df,
        dynamic_df=dynamic_df,
        robustness_df=robust_df,
    )

    overall_conclusion = {
        "is_m8_working": bool(any(v["verdict"]["overall_status"] for v in estimator_payloads.values())),
        "does_m8_improve_over_m7": bool(estimator_payloads[best]["global_metrics"].get("RMSE_V_MAG_NONPMU", np.inf) < m7_ref_nonpmu),
        "main_strengths": [
            "Hybrid dynamic variants include swing dynamics and graph regularization.",
            "ANDES truth validation and per-scenario metrics are populated.",
        ],
        "main_failures": [k for k, v in estimator_payloads.items() if not v["verdict"]["overall_status"]],
        "recommended_default_estimator": best,
        "next_actions": [
            "Tune Q/R and graph alpha per scenario.",
            "Expand dynamic model with richer generator/load pseudo-measurements.",
        ],
    }

    payload = {
        "run_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(out),
            "scenario_set": scenario_set,
            "default_estimator": default_estimator,
        },
        "data_sources": {
            "raw_path": str(Path(raw_path)),
            "pmu_location_path": str(Path(pmu_location_path)),
            "truth_source": "ANDES",
            "m7_reference_source": str(m7_results_path) if m7_results_path else "auto_default",
        },
        "network": {
            "bus_count": len(network.bus_order),
            "pmu_buses": pmu_buses,
            "generator_buses": generator_buses,
            "electrical_distance_meta": dist_meta,
        },
        "scenarios": [{"scenario_id": s["id"], "frame_count": int(len(s["idx"]))} for s in scenarios],
        "estimators": estimator_payloads,
        "ranking": {
            "by_overall_score": ranking_overall,
            "by_rmse_v_mag_nonpmu": ranking_rmse_mag_nonpmu,
            "by_rmse_ang_nonpmu": ranking_rmse_ang_nonpmu,
            "by_dynamic_tracking_score": ranking_dynamic,
        },
        "best_estimator": {
            "name": best,
            "why": [
                "Best weighted multi-criteria overall score.",
                "Balances non-PMU accuracy, PMU fit, and robustness.",
            ],
        },
        "overall_conclusion": overall_conclusion,
    }
    write_m8_reports(paths["report"], payload)
    return {"output_dir": str(out), "best_estimator": best, "overall_conclusion": overall_conclusion, "ranking": payload["ranking"]}
