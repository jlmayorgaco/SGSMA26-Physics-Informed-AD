"""Generate engineering-review artifacts for the SGSMA submission.

The artifacts are intended for a power-systems reviewer: they show which
detector fired, what the PMUs measured, what the Ybus state proxy estimated at
hidden buses, the units used, and how the IEEE-39 topology supports the
localization claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.classifier.train_lgbm import _build_detector
from src.detector.chi2 import Chi2Detector, compute_eta_simple, extract_data_present
from src.estimator.calibration import channel_cols
from src.estimator.topology_state import TopologyStateEstimator, state_feature_summary
from src.eval.metrics import classification_metrics, detection_metrics
from src.eval.splits import make_splits
from src.io.load_csv import PMU_BUSES, load_all
from src.io.load_events import load_events
from src.pipeline.run_inference import (
    _build_per_bus_prediction_columns,
    _build_prediction_columns,
    _classify_and_localize,
    _extract_gt_windows,
    _gt_bus_for_alarm_frame,
    _load_grid,
    _run_detection,
    _train_model,
)
from src.localizer.cosine_match import locate


LABEL_NAMES = {
    0: "normal",
    1: "fault",
    2: "line_outage",
    3: "generation_change",
    4: "load_change",
    5: "missing_data",
    6: "missing_data_plus_physical",
    7: "bad_data",
    8: "unknown_event",
}


UNITS = {
    "TIMESTAMP": "seconds from start",
    "VA_ANG/VB_ANG/VC_ANG": "electrical degrees",
    "VA_MAG/VB_MAG/VC_MAG": "volts, line-to-neutral RMS",
    "IA_MAG/IB_MAG/IC_MAG": "amperes RMS",
    "Freq": "Hz",
    "ROCOF": "Hz/s",
    "DATA_PRESENT": "1 valid PMU frame, 0 missing frame",
    "Event/Predicted_Event": "integer class 0..8",
    "Predicted_Location": "bus id 1..39 or -1; line events use deterministic endpoint",
    "eta": "dimensionless normalized chi-square-like residual",
    "bad_data_score": "dimensionless phase-unbalance ratio",
    "estimated_vm": "per-unit voltage magnitude",
    "estimated_va": "electrical degrees",
    "state_residual_energy": "dimensionless normalized voltage/angle residual",
}


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _phase_voltage_pu(df: pd.DataFrame, grid, bus: int, frames: np.ndarray) -> np.ndarray:
    cols = [f"BUS{bus}_{phase}_MAG" for phase in ("VA", "VB", "VC")]
    vals = df.iloc[frames][cols].to_numpy(dtype=float)
    psse = grid.comp_to_psse.get(bus)
    idx = grid.psse_to_idx.get(psse) if psse is not None else None
    base_v_ln = 1.0
    if idx is not None:
        base_v_ln = grid.buses[idx].base_kv * 1000.0 / np.sqrt(3.0)
    out = np.full(len(frames), np.nan, dtype=float)
    valid_rows = np.isfinite(vals).any(axis=1)
    if np.any(valid_rows):
        with np.errstate(invalid="ignore"):
            out[valid_rows] = np.nanmean(vals[valid_rows], axis=1) / max(base_v_ln, 1.0)
    return out


def _event_segments(df: pd.DataFrame, events) -> list[dict]:
    ev = df["Event"].to_numpy(dtype=int)
    ts = df["TIMESTAMP"].to_numpy(dtype=float)
    windows = _extract_gt_windows(ev, ts)
    segments: list[dict] = []
    for idx, (start_s, end_s) in enumerate(windows):
        start_frame = int(np.searchsorted(ts, start_s))
        end_frame = int(np.searchsorted(ts, end_s, side="right") - 1)
        label = int(ev[start_frame])
        segments.append(
            {
                "event_id": idx + 1,
                "label": label,
                "label_name": LABEL_NAMES.get(label, "unknown"),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_s": float(start_s),
                "end_s": float(end_s),
                "duration_s": float(max(0.0, end_s - start_s)),
                "true_bus": _gt_bus_for_alarm_frame(df, start_frame, events),
            }
        )
    return segments


def _align_alarm_windows(alarm_times: np.ndarray, segments: list[dict], tol_sec: float = 5.0) -> dict[int, int | None]:
    matched: set[int] = set()
    result: dict[int, int | None] = {}
    for alarm_pos, t_alarm in sorted(enumerate(alarm_times), key=lambda item: item[1]):
        candidates: list[tuple[float, int]] = []
        for idx, seg in enumerate(segments):
            start = float(seg["start_s"])
            end = float(seg["end_s"])
            if (start - tol_sec) <= float(t_alarm) <= (end + tol_sec):
                if start <= float(t_alarm) <= end:
                    dist = abs(float(t_alarm) - start)
                else:
                    dist = min(abs(float(t_alarm) - start), abs(float(t_alarm) - end))
                candidates.append((dist, idx))
        unmatched = [(dist, idx) for dist, idx in candidates if idx not in matched]
        if unmatched:
            _dist, idx = min(unmatched, key=lambda item: item[0])
            matched.add(idx)
            result[alarm_pos] = idx
        elif candidates:
            result[alarm_pos] = min(candidates, key=lambda item: item[0])[1]
        else:
            result[alarm_pos] = None
    return result


def _spectral_layout(grid) -> dict[int, tuple[float, float]]:
    buses = list(grid.ext_bus_order)
    idx = {bus: i for i, bus in enumerate(buses)}
    n = len(buses)
    A = np.zeros((n, n), dtype=float)
    for a, b in grid.branch_list:
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            A[i, j] = A[j, i] = 1.0
    L = np.diag(A.sum(axis=1)) - A
    try:
        vals, vecs = np.linalg.eigh(L)
        order = np.argsort(vals)
        xy = vecs[:, order[1:3]]
        xy = xy / np.maximum(np.max(np.abs(xy), axis=0), 1e-12)
    except Exception:
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xy = np.c_[np.cos(theta), np.sin(theta)]
    return {bus: (float(xy[i, 0]), float(xy[i, 1])) for bus, i in idx.items()}


def _plot_topology(grid, out_path: Path, event_buses: set[int]) -> dict[int, tuple[float, float]]:
    coords = _spectral_layout(grid)
    fig, ax = plt.subplots(figsize=(10, 8))
    for a, b in grid.branch_list:
        if a not in coords or b not in coords:
            continue
        xa, ya = coords[a]
        xb, yb = coords[b]
        is_target_line = {a, b} == {23, 24}
        ax.plot(
            [xa, xb],
            [ya, yb],
            color="#d46a1f" if is_target_line else "#b8b8b8",
            lw=2.8 if is_target_line else 1.0,
            zorder=1,
        )
    for bus, (x, y) in coords.items():
        is_pmu = bus in PMU_BUSES
        is_event = bus in event_buses
        color = "#d62728" if is_pmu else "#1f77b4"
        if is_event:
            color = "#ffbf00"
        ax.scatter(
            [x],
            [y],
            s=160 if is_pmu or is_event else 70,
            marker="s" if is_pmu else "o",
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(x, y, str(bus), ha="center", va="center", fontsize=8, zorder=4)
    ax.set_title("IEEE 39-bus topology: PMUs, event buses, and target line 23-24")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return coords


def _selected_buses(pred_bus: int, true_bus: int, state_by_bus: dict[int, float], label: int) -> list[int]:
    buses: list[int] = []
    for bus in (pred_bus, true_bus, 7, 23, 24, 39, 2):
        if isinstance(bus, (int, np.integer)) and 1 <= int(bus) <= 39 and int(bus) not in buses:
            buses.append(int(bus))
    for bus, _score in sorted(state_by_bus.items(), key=lambda item: -item[1]):
        if bus not in buses:
            buses.append(int(bus))
        if len(buses) >= 6:
            break
    if label == 7:
        return buses[:4]
    return buses[:6]


def _plot_event(
    *,
    df: pd.DataFrame,
    grid,
    estimator: TopologyStateEstimator,
    eta: np.ndarray,
    eta_threshold: float,
    bad_score: np.ndarray,
    scenario: dict,
    state_by_bus: dict[int, float],
    out_path: Path,
) -> None:
    ts = df["TIMESTAMP"].to_numpy(dtype=float)
    frame = int(scenario["alarm_frame"])
    label = int(scenario["predicted_label"])
    matched = scenario.get("matched_event") or {}
    start_frame = int(matched.get("start_frame", frame))

    before_s = 8.0 if label in {1, 2, 7} else 30.0
    after_s = 20.0 if label in {1, 2, 7} else 180.0
    lo = max(0, np.searchsorted(ts, ts[frame] - before_s))
    hi = min(len(df), np.searchsorted(ts, ts[frame] + after_s))
    if hi <= lo:
        hi = min(len(df), lo + 2)
    frames = np.arange(lo, hi)
    if len(frames) > 800:
        frames = np.unique(np.linspace(lo, hi - 1, 800).astype(int))
    rel_t = ts[frames] - float(matched.get("start_s", ts[start_frame]))

    selected = _selected_buses(
        int(scenario.get("predicted_location_bus", -1)),
        int(matched.get("true_bus", -1)),
        state_by_bus,
        label,
    )

    est_vm: dict[int, list[float]] = {bus: [] for bus in selected}
    for f in frames:
        vm, _va = estimator.estimate_window(df, int(f), int(f) + 1)
        for bus in selected:
            idx = estimator.bus_to_idx.get(bus)
            est_vm[bus].append(float(vm[idx]) if idx is not None else float("nan"))

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=False)

    for bus in PMU_BUSES:
        axes[0].plot(rel_t, _phase_voltage_pu(df, grid, bus, frames), lw=1.0, label=f"PMU {bus}")
    axes[0].axvline(ts[frame] - float(matched.get("start_s", ts[start_frame])), color="black", ls="--", lw=1)
    axes[0].set_ylabel("|V| observed (p.u.)")
    axes[0].set_title(
        f"Alarm {scenario['alarm_id']:02d}: label {label} {LABEL_NAMES.get(label)} "
        f"at t={scenario['timestamp_s']:.3f}s"
    )
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(alpha=0.25)

    for bus in selected:
        axes[1].plot(rel_t, est_vm[bus], lw=1.2, label=f"Est. bus {bus}")
    axes[1].axvline(ts[frame] - float(matched.get("start_s", ts[start_frame])), color="black", ls="--", lw=1)
    axes[1].set_ylabel("|V| estimated (p.u.)")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(alpha=0.25)

    axes[2].plot(rel_t, eta[frames] / max(float(eta_threshold), 1e-12), color="#1f77b4", label="eta / threshold")
    axes[2].plot(rel_t, bad_score[frames] / 0.04, color="#d62728", alpha=0.8, label="bad-data score / threshold")
    axes[2].axhline(1.0, color="black", lw=0.8, ls=":")
    axes[2].axvline(ts[frame] - float(matched.get("start_s", ts[start_frame])), color="black", ls="--", lw=1)
    axes[2].set_ylabel("detector score")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    top = sorted(state_by_bus.items(), key=lambda item: -item[1])[:10]
    axes[3].bar([str(bus) for bus, _ in top], [score for _, score in top], color="#4c78a8")
    axes[3].set_xlabel("Bus")
    axes[3].set_ylabel("Ybus residual energy")
    axes[3].set_title("Top reconstructed-state residual buses at alarm")
    axes[3].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def generate_review(
    *,
    data_dir: Path,
    raw_path: Path,
    synthetic_dir: Path | None,
    out_dir: Path,
    seed: int,
    fps: float = 30.0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(data_dir)
    splits = make_splits(df, fps=fps)
    grid, J_cols, branches, zbus, ext_bus_order = _load_grid(raw_path)
    estimator = TopologyStateEstimator(grid)

    df_train = df.iloc[splits["train"]].reset_index(drop=True)
    df_val = df.iloc[splits["val"]].reset_index(drop=True)
    clf, h0_flat, offset, R, threshold = _train_model(
        df_train, df_val, grid, J_cols, None, synthetic_dir
    )

    cols = channel_cols()
    eta = compute_eta_simple(df, h0_flat, offset, R, cols)
    dp = extract_data_present(df)
    ts = df["TIMESTAMP"].to_numpy(dtype=float)

    main_det = Chi2Detector(n_z=len(cols), k_on=3, k_off=15, fps=fps)
    main_det.threshold = threshold
    main_result = main_det.detect(eta, dp, ts)

    det_result, eta = _run_detection(df, h0_flat, offset, R, threshold, fps)
    onset_idx = det_result["alarm_indices"]
    forced = {int(i): 7 for i in det_result.get("bad_data_indices", [])}
    predicted_labels, top3_buses_all, top3_lines_all = _classify_and_localize(
        df,
        onset_idx,
        clf,
        grid,
        J_cols,
        branches,
        h0_flat,
        offset,
        R,
        fps,
        forced_labels_by_frame=forced,
    )
    pred_event_col, pred_loc_col = _build_prediction_columns(
        df,
        det_result,
        onset_idx,
        predicted_labels,
        top3_buses_all,
        top3_lines_all,
        fps=fps,
    )
    pred_event_by_bus, _pred_loc_by_bus = _build_per_bus_prediction_columns(
        df,
        pred_event_col,
        pred_loc_col,
        onset_idx,
        predicted_labels,
        top3_buses_all,
        top3_lines_all,
        fps=fps,
    )

    events = load_events(raw_path.parent)
    segments = _event_segments(df, events)
    windows = [(float(s["start_s"]), float(s["end_s"])) for s in segments]
    total_sec = float(ts[-1] - ts[0])
    full_metrics = detection_metrics(ts[onset_idx], windows, total_sec)
    main_metrics = detection_metrics(ts[main_result["alarm_indices"]], windows, total_sec)
    true_at_alarms = np.array([int(df["Event"].iloc[int(i)]) for i in onset_idx], dtype=int)
    class_metrics = classification_metrics(true_at_alarms, predicted_labels)
    alignment = _align_alarm_windows(ts[onset_idx], segments)

    labels_0_8 = list(range(9))
    labels_0_7 = list(range(8))
    y_true_parts = []
    y_pred_parts = []
    per_bus_sample: dict[str, dict] = {}
    for bus in PMU_BUSES:
        true_bus = df[f"BUS{bus}_Event"].fillna(0).astype(int).to_numpy()
        pred_bus = pred_event_by_bus[int(bus)]
        present_labels = sorted(set(true_bus.tolist()) | set(pred_bus.tolist()))
        y_true_parts.append(true_bus)
        y_pred_parts.append(pred_bus)
        per_bus_sample[str(bus)] = {
            "macro_f1_0_8": float(f1_score(true_bus, pred_bus, labels=labels_0_8, average="macro", zero_division=0)),
            "macro_f1_visible_0_7": float(f1_score(true_bus, pred_bus, labels=labels_0_7, average="macro", zero_division=0)),
            "macro_f1_present_labels": float(f1_score(true_bus, pred_bus, labels=present_labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(true_bus, pred_bus, labels=labels_0_8, average="weighted", zero_division=0)),
            "true_counts": {str(int(k)): int(v) for k, v in pd.Series(true_bus).value_counts().sort_index().items()},
            "pred_counts": {str(int(k)): int(v) for k, v in pd.Series(pred_bus).value_counts().sort_index().items()},
        }
    y_true_all = np.concatenate(y_true_parts)
    y_pred_all = np.concatenate(y_pred_parts)
    per_bus_sample_metrics = {
        "definition": "aggregate all eight PMU-specific Event columns against their matching per-bus prediction arrays",
        "macro_f1_0_8": float(f1_score(y_true_all, y_pred_all, labels=labels_0_8, average="macro", zero_division=0)),
        "macro_f1_visible_0_7": float(f1_score(y_true_all, y_pred_all, labels=labels_0_7, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_all, y_pred_all, labels=labels_0_8, average="weighted", zero_division=0)),
        "per_bus": per_bus_sample,
    }

    event_buses = {int(seg["true_bus"]) for seg in segments if int(seg["true_bus"]) > 0}
    event_buses.update(int(b[0]) for buses in top3_buses_all for b in [buses[:1]] if b)
    coords = _plot_topology(grid, fig_dir / "ieee39_topology_engineering_review.png", event_buses)

    scenarios: list[dict] = []
    localization_explanations: list[dict] = []
    bad_indices = set(int(i) for i in det_result.get("bad_data_indices", []))
    bad_score = np.asarray(det_result.get("bad_data_score", np.zeros(len(df))), dtype=float)
    for pos, frame in enumerate(onset_idx.tolist()):
        matched_idx = alignment.get(pos)
        matched = dict(segments[matched_idx]) if matched_idx is not None else None
        label = int(predicted_labels[pos])
        loc_full = locate(
            df,
            int(frame),
            label,
            J_cols,
            branches,
            fps=fps,
            h0_flat=h0_flat,
            offset=offset,
            R=R,
            grid=grid,
            state_estimator=estimator,
        )
        state_features, state_by_bus = state_feature_summary(df, int(frame), estimator, fps=fps)
        detector_source = "phase_unbalance_bad_data" if int(frame) in bad_indices else "chi2_or_data_present"
        if detector_source != "phase_unbalance_bad_data" and (dp[int(frame)] < 1).any():
            detector_source = "data_present_missing"

        scenario = {
            "alarm_id": pos + 1,
            "alarm_frame": int(frame),
            "timestamp_s": float(ts[int(frame)]),
            "timestamp_min": float(ts[int(frame)] / 60.0),
            "detector_source": detector_source,
            "eta": float(eta[int(frame)]),
            "eta_threshold": float(threshold),
            "eta_over_threshold": float(eta[int(frame)] / max(float(threshold), 1e-12)),
            "bad_data_score": float(bad_score[int(frame)]),
            "bad_data_threshold": 0.04,
            "predicted_label": label,
            "predicted_label_name": LABEL_NAMES.get(label, "unknown"),
            "predicted_location_bus": int(top3_buses_all[pos][0]) if top3_buses_all[pos] else -1,
            "top3_buses": [int(b) for b in top3_buses_all[pos]],
            "top3_lines": [
                {"line": [int(a), int(b)], "score": float(score)}
                for (a, b), score in (top3_lines_all[pos] if pos < len(top3_lines_all) else [])
            ],
            "matched_event": matched,
            "state_features": {
                "top_bus": int(state_features[0]),
                "top_energy": float(state_features[1]),
                "entropy": float(state_features[2]),
                "bus7_energy": float(state_features[3]),
                "bus23_energy": float(state_features[4]),
                "bus24_energy": float(state_features[5]),
                "line_24_23_energy": float(state_features[6]),
            },
            "state_top10": [
                {"bus": int(bus), "energy": float(score)}
                for bus, score in sorted(state_by_bus.items(), key=lambda item: -item[1])[:10]
            ],
            "localization_score_table": loc_full.get("score_table", [])[:10],
            "fault_subtype": loc_full.get("fault_subtype"),
        }
        localization_explanations.append(
            {
                "alarm_id": pos + 1,
                "alarm_frame": int(frame),
                "predicted_label": label,
                "predicted_label_name": LABEL_NAMES.get(label, "unknown"),
                "top3_buses": scenario["top3_buses"],
                "top3_lines": scenario["top3_lines"],
                "score_table": loc_full.get("score_table", [])[:10],
                "fault_subtype": loc_full.get("fault_subtype"),
            }
        )
        plot_path = fig_dir / f"event_{pos + 1:02d}_label{label}_{LABEL_NAMES.get(label, 'unknown')}.png"
        _plot_event(
            df=df,
            grid=grid,
            estimator=estimator,
            eta=eta,
            eta_threshold=float(threshold),
            bad_score=bad_score,
            scenario=scenario,
            state_by_bus=state_by_bus,
            out_path=plot_path,
        )
        scenario["plot"] = plot_path.relative_to(out_dir).as_posix()
        scenarios.append(scenario)

    nodes = []
    for bus, (x, y) in coords.items():
        psse = grid.comp_to_psse.get(bus)
        idx = grid.psse_to_idx.get(psse) if psse is not None else None
        rec = grid.buses[idx] if idx is not None else None
        nodes.append(
            {
                "bus": int(bus),
                "is_pmu": bus in PMU_BUSES,
                "is_event_bus": bus in event_buses,
                "base_kv": float(rec.base_kv) if rec is not None else None,
                "base_vm_pu": float(rec.vm_pu) if rec is not None else None,
                "base_va_deg": float(rec.va_deg) if rec is not None else None,
                "layout_x": float(x),
                "layout_y": float(y),
            }
        )
    branches_json = [
        {
            "from_bus": int(a),
            "to_bus": int(b),
            "is_target_line_23_24": set((a, b)) == {23, 24},
        }
        for a, b in grid.branch_list
    ]

    audit = {
        "summary": {
            "visible_detection_after_bad_data_detector": full_metrics,
            "main_chi2_data_present_detector_only": main_metrics,
            "classification_on_matched_visible_alarms": {
                "macro_f1": class_metrics["macro_f1"],
                "weighted_f1": class_metrics["weighted_f1"],
            },
            "sample_level_per_bus_predictions": {
                "macro_f1_0_8": per_bus_sample_metrics["macro_f1_0_8"],
                "macro_f1_visible_0_7": per_bus_sample_metrics["macro_f1_visible_0_7"],
                "weighted_f1": per_bus_sample_metrics["weighted_f1"],
            },
            "interpretation": (
                "The 1.0 detection result is a visible labeled-data audit, not a hidden-test guarantee. "
                "The detector path does not use Event labels for alarm firing; Event is used here only "
                "for scoring and training labels. Calibration uses first-60-second robust candidates "
                "filtered by DATA_PRESENT and phase consistency, not Event == 0."
            ),
        },
        "thresholds": {
            "eta_threshold": float(threshold),
            "bad_data_phase_unbalance_threshold": 0.04,
            "chi2_k_on_frames": 3,
            "chi2_k_off_frames": 15,
            "bad_data_k_on_frames": 2,
            "bad_data_k_off_frames": 5,
        },
        "alarms": scenarios,
    }
    topology = {
        "description": "IEEE 39-bus graph parsed from the RAW file. Coordinates are deterministic spectral-layout coordinates for review plots, not geographic coordinates.",
        "pmu_buses": PMU_BUSES,
        "nodes": sorted(nodes, key=lambda item: item["bus"]),
        "branches": branches_json,
        "diagram": "figures/ieee39_topology_engineering_review.png",
    }
    manifest = {
        "root": out_dir.as_posix(),
        "files": {
            "README": "README.md",
            "audit": "detection_and_scenario_audit.json",
            "units": "units.json",
            "per_bus_sample_metrics": "per_bus_sample_metrics.json",
            "topology": "ieee39_topology.json",
            "scenario_table": "scenario_table.csv",
            "figures": "figures/",
        },
        "n_alarms": len(scenarios),
        "n_figures": len(list(fig_dir.glob("*.png"))),
    }

    _write_json(out_dir / "detection_and_scenario_audit.json", audit)
    _write_json(out_dir.parent / "localization_explanations.json", localization_explanations)
    _write_json(out_dir / "per_bus_sample_metrics.json", per_bus_sample_metrics)
    _write_json(out_dir / "units.json", UNITS)
    _write_json(out_dir / "ieee39_topology.json", topology)
    _write_json(out_dir / "manifest.json", manifest)
    pd.DataFrame(scenarios).to_csv(out_dir / "scenario_table.csv", index=False)
    readme = f"""# Engineering Review Artifacts

These artifacts audit the visible SGSMA 2026 run from an electrical-engineering
perspective. They are not hidden-test claims; they document why each visible
alarm fired and what the Ybus state proxy estimated from the eight PMUs.

Key result:

- Full detector visible audit: precision {full_metrics['precision']:.3f},
  recall {full_metrics['recall']:.3f}, F1 {full_metrics['f1']:.3f},
  TP/FP/FN {full_metrics['tp']}/{full_metrics['fp']}/{full_metrics['fn']}.
- Main chi2/DATA_PRESENT detector only: precision {main_metrics['precision']:.3f},
  recall {main_metrics['recall']:.3f}, F1 {main_metrics['f1']:.3f},
  TP/FP/FN {main_metrics['tp']}/{main_metrics['fp']}/{main_metrics['fn']}.
- The improvement comes from adding the deterministic phase-unbalance detector
  for label 7 bad-data bursts and from matching stacked labels as separate
  transitions.
- Per-bus sample audit: macro-F1 0--8
  {per_bus_sample_metrics['macro_f1_0_8']:.3f}, visible-label macro-F1 0--7
  {per_bus_sample_metrics['macro_f1_visible_0_7']:.3f}, weighted-F1
  {per_bus_sample_metrics['weighted_f1']:.3f}.

Files:

- `detection_and_scenario_audit.json`: alarm-by-alarm source, thresholds,
  labels, locations, top residual buses, and figure links.
- `units.json`: channel units and derived-score units.
- `per_bus_sample_metrics.json`: each PMU-specific Event column compared with
  the matching per-bus prediction stream.
- `ieee39_topology.json`: buses, PMU flags, branch list, and diagram metadata.
- `scenario_table.csv`: flat table for spreadsheet review.
- `figures/`: IEEE-39 topology diagram and one plot per alarm.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--raw", type=Path, default=Path("data/metadata/IEEE 39 Bus Power System.raw"))
    parser.add_argument("--synthetic", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--out", type=Path, default=Path("report/engineering_review"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest = generate_review(
        data_dir=args.data,
        raw_path=args.raw,
        synthetic_dir=args.synthetic,
        out_dir=args.out,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
