from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.domain.services.hybrid_event_detector import HybridEventDetector
from src.detectors.domain.services.non0_detector_v2 import Non0DetectorV2
from src.detectors.pipelines.common import build_frame_output, write_json
from src.detectors.pipelines.raw_holdout_loader import load_raw_holdout_frames
from src.detectors.training.datasets.detector_dataset_builder import DetectorDatasetBuilder, merge_detection_inputs
from src.detectors.training.datasets.detector_split_loader import SplitRecord, load_split_csv
from src.detectors.training.evaluators.detector_evaluator import BinaryDetectorEvaluator
from src.detectors.training.evaluators.plotting import plot_confusion_matrix


@dataclass(slots=True)
class FusionTuneConfig:
    event5_trigger_min: float
    event7_trigger_min: float
    physical_trigger_min: float
    branch_agreement_min: float
    decision_threshold: float
    start_threshold: float
    stop_threshold: float
    min_on_frames: int
    min_off_frames: int
    quiet_reset_threshold: float
    quiet_reset_frames: int


@dataclass(slots=True)
class SplitMasks:
    dev_mask: np.ndarray
    holdout_mask: np.ndarray
    interval_rows: pd.DataFrame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Non-0 Detector V2 RAW protocol fix + branch-aware hardening")
    p.add_argument("--train-split", type=Path, default=Path("train_scenarios_v2.csv"))
    p.add_argument("--val-split", type=Path, default=Path("val_scenarios_v2.csv"))
    p.add_argument("--test-split", type=Path, default=Path("test_scenarios_v2.csv"))
    p.add_argument("--raw-input-root", type=Path, default=Path("data/RAW0001"))
    p.add_argument("--m10-model-path", type=Path, default=Path("output/detector_m10_2_ready/models/detector_model.pkl"))
    p.add_argument("--baseline-v2-report", type=Path, default=Path("output/non0_detector_v2/metrics/non0_detector_v2_report.json"))
    p.add_argument("--baseline-raw-report", type=Path, default=Path("output/detector_raw_holdout_eval/metrics/raw_holdout_report.json"))
    p.add_argument("--output-root", type=Path, default=Path("output/non0_detector_v2_rawfix"))
    p.add_argument("--raw-guard-band-frames", type=int, default=64)
    p.add_argument("--holdout-interval-mod", type=int, default=3)
    p.add_argument("--holdout-target-normal-ratio", type=float, default=0.35)
    return p.parse_args()


def _extract_intervals(y: np.ndarray, value: int) -> list[tuple[int, int]]:
    v = np.asarray(y, dtype=int)
    rows: list[tuple[int, int]] = []
    i = 0
    while i < len(v):
        if v[i] != value:
            i += 1
            continue
        j = i
        while j < len(v) and v[j] == value:
            j += 1
        rows.append((i, j - 1))
        i = j
    return rows


def _subset_inputs(inp: DetectionInput, mask: np.ndarray, *, scenario_id: str, split: str) -> DetectionInput:
    idx = np.where(mask.astype(bool))[0]
    if len(idx) == 0:
        return DetectionInput(
            scenario_id=scenario_id,
            split=split,
            x_windows=np.zeros((0, inp.x_windows.shape[1], inp.x_windows.shape[2]), dtype=float),
            timestamps=np.zeros((0,), dtype=float),
            feature_names=list(inp.feature_names),
            metadata=pd.DataFrame(columns=inp.metadata.columns),
        )
    return DetectionInput(
        scenario_id=scenario_id,
        split=split,
        x_windows=inp.x_windows[idx],
        timestamps=inp.timestamps[idx],
        feature_names=list(inp.feature_names),
        metadata=inp.metadata.iloc[idx].reset_index(drop=True),
    )


def _build_interval_aware_split(
    y_true: np.ndarray,
    timestamps: np.ndarray,
    *,
    scenario_id: str,
    guard_band: int,
    holdout_interval_mod: int,
    holdout_target_normal_ratio: float,
) -> SplitMasks:
    n = len(y_true)
    holdout = np.zeros((n,), dtype=bool)
    abnormal = _extract_intervals(y_true, 1)
    normal = _extract_intervals(y_true, 0)

    # Full abnormal intervals with expanded guard-bands.
    for i, (s, e) in enumerate(abnormal):
        if holdout_interval_mod <= 1 or (i % holdout_interval_mod) == 1:
            gs = max(0, s - guard_band)
            ge = min(n - 1, e + guard_band)
            holdout[gs : ge + 1] = True

    # Ensure holdout has realistic normal context.
    abnormal_holdout = int(((y_true == 1) & holdout).sum())
    normal_holdout = int(((y_true == 0) & holdout).sum())
    need_normals = int(max(0.0, holdout_target_normal_ratio * max(abnormal_holdout, 1) - normal_holdout))
    if need_normals > 0:
        normal_sorted = sorted(normal, key=lambda x: (x[1] - x[0] + 1), reverse=True)
        for s, e in normal_sorted:
            if need_normals <= 0:
                break
            if np.any(holdout[s : e + 1]):
                continue
            holdout[s : e + 1] = True
            need_normals -= (e - s + 1)

    # Safety: if holdout got no abnormal intervals, force one interval.
    if int(((y_true == 1) & holdout).sum()) == 0 and abnormal:
        s, e = abnormal[len(abnormal) // 2]
        gs = max(0, s - guard_band)
        ge = min(n - 1, e + guard_band)
        holdout[gs : ge + 1] = True

    dev = ~holdout
    rows: list[dict[str, Any]] = []
    for split_name, mask in [("raw_dev", dev), ("raw_holdout", holdout)]:
        for label, ivals in [("abnormal", _extract_intervals(((y_true == 1) & mask).astype(int), 1)), ("normal", _extract_intervals(((y_true == 0) & mask).astype(int), 1))]:
            for idx, (s, e) in enumerate(ivals):
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "split": split_name,
                        "interval_type": label,
                        "interval_index": idx,
                        "start_index": int(s),
                        "end_index": int(e),
                        "start_timestamp": float(timestamps[s]),
                        "end_timestamp": float(timestamps[e]),
                        "duration_s": float(max(0.0, timestamps[e] - timestamps[s])),
                        "frame_count": int(e - s + 1),
                    }
                )
    return SplitMasks(dev_mask=dev, holdout_mask=holdout, interval_rows=pd.DataFrame(rows))


def _build_raw_dev_holdout_mask(y_true: np.ndarray, *, guard_band: int, holdout_mod: int) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.arange(len(y_true), dtype=float)
    split = _build_interval_aware_split(
        np.asarray(y_true, dtype=int),
        timestamps,
        scenario_id="raw",
        guard_band=guard_band,
        holdout_interval_mod=holdout_mod,
        holdout_target_normal_ratio=0.30,
    )
    return split.dev_mask, split.holdout_mask


def _eval_bundle(evaluator: BinaryDetectorEvaluator, frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return evaluator.metric_bundle(
            y_true=np.zeros((0,), dtype=int),
            y_pred=np.zeros((0,), dtype=int),
            y_score=np.zeros((0,), dtype=float),
            timestamps=np.zeros((0,), dtype=float),
            scenario_ids=np.asarray([], dtype=object),
        )
    return evaluator.metric_bundle(
        y_true=frame["y_true"].to_numpy(dtype=int),
        y_pred=frame["y_pred_stable"].to_numpy(dtype=int),
        y_score=frame["p_non0"].to_numpy(dtype=float),
        timestamps=frame["timestamp"].to_numpy(dtype=float),
        scenario_ids=frame["scenario_id"].to_numpy() if "scenario_id" in frame.columns else None,
    )


def _eval_split_quality(frame: pd.DataFrame, intervals: pd.DataFrame, *, split_name: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "split": split_name,
            "windows": 0,
            "abnormal_windows": 0,
            "normal_windows": 0,
            "abnormal_ratio": 0.0,
            "normal_ratio": 0.0,
            "duration_minutes": 0.0,
            "abnormal_intervals": 0,
            "normal_intervals": 0,
        }
    abnormal_windows = int(frame["y_true"].astype(int).sum())
    windows = int(len(frame))
    normal_windows = windows - abnormal_windows
    duration_minutes = float(max(0.0, frame["timestamp"].max() - frame["timestamp"].min()) / 60.0) if windows > 1 else 0.0
    abn_intervals = int(((intervals["split"] == split_name) & (intervals["interval_type"] == "abnormal")).sum()) if not intervals.empty else 0
    norm_intervals = int(((intervals["split"] == split_name) & (intervals["interval_type"] == "normal")).sum()) if not intervals.empty else 0
    return {
        "split": split_name,
        "windows": windows,
        "abnormal_windows": abnormal_windows,
        "normal_windows": normal_windows,
        "abnormal_ratio": float(abnormal_windows / max(windows, 1)),
        "normal_ratio": float(normal_windows / max(windows, 1)),
        "duration_minutes": duration_minutes,
        "abnormal_intervals": abn_intervals,
        "normal_intervals": norm_intervals,
    }


def _enrich(frame: pd.DataFrame, diagnostics: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    out["p_non0"] = out["p_abnormal"]
    out["p_event5"] = pd.Series(diagnostics.get("p_event5", []), dtype=float)
    out["p_event7"] = pd.Series(diagnostics.get("p_event7", []), dtype=float)
    out["event5_state"] = pd.Series(diagnostics.get("event5_states", []), dtype=str)
    out["event7_mode"] = pd.Series(diagnostics.get("event7_modes", []), dtype=str)
    out["fusion_attribution"] = pd.Series(diagnostics.get("fusion_attribution", []), dtype=str)
    out["fusion_trigger_reason"] = pd.Series(diagnostics.get("fusion_trigger_reason", []), dtype=str)
    out["state_machine_state"] = pd.Series(diagnostics.get("state_machine_states", []), dtype=str)
    return out


def _apply_config(detector: Non0DetectorV2, cfg: FusionTuneConfig) -> None:
    detector.fusion.event5_trigger_min = float(cfg.event5_trigger_min)
    detector.fusion.event7_trigger_min = float(cfg.event7_trigger_min)
    detector.fusion.physical_trigger_min = float(cfg.physical_trigger_min)
    detector.fusion.branch_agreement_min = float(cfg.branch_agreement_min)
    detector.fusion.decision_threshold = float(cfg.decision_threshold)
    detector.state_machine.start_threshold = float(cfg.start_threshold)
    detector.state_machine.stop_threshold = float(cfg.stop_threshold)
    detector.state_machine.min_on_frames = int(cfg.min_on_frames)
    detector.state_machine.min_off_frames = int(cfg.min_off_frames)
    detector.state_machine.quiet_reset_threshold = float(cfg.quiet_reset_threshold)
    detector.state_machine.quiet_reset_frames = int(cfg.quiet_reset_frames)


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_f1": float(metrics.get("f1_abnormal", 0.0) or 0.0),
        f"{prefix}_precision": float(metrics.get("precision_abnormal", 0.0) or 0.0),
        f"{prefix}_recall": float(metrics.get("recall_abnormal", 0.0) or 0.0),
        f"{prefix}_fp_per_min": float(metrics.get("false_positives_per_minute", 0.0) or 0.0),
    }


def _sweep_configs() -> list[FusionTuneConfig]:
    rows: list[FusionTuneConfig] = []
    for e7 in [0.76, 0.80, 0.84]:
        for dec in [0.64, 0.68, 0.72]:
            for start, stop in [(0.70, 0.52), (0.74, 0.54), (0.78, 0.58)]:
                rows.append(
                    FusionTuneConfig(
                        event5_trigger_min=0.90,
                        event7_trigger_min=e7,
                        physical_trigger_min=0.84,
                        branch_agreement_min=0.68,
                        decision_threshold=dec,
                        start_threshold=start,
                        stop_threshold=stop,
                        min_on_frames=3,
                        min_off_frames=4,
                        quiet_reset_threshold=0.45,
                        quiet_reset_frames=3,
                    )
                )
    return rows


def _plot_scores(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        plt.figure(figsize=(8, 3))
        plt.title("No data")
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        return
    g = frame.sort_values(["scenario_id", "timestamp"]).reset_index(drop=True)
    plt.figure(figsize=(12, 4))
    plt.plot(g["timestamp"], g["p_event5"], label="p_event5")
    plt.plot(g["timestamp"], g["p_event7"], label="p_event7")
    plt.plot(g["timestamp"], g["p_physical"], label="p_physical")
    plt.plot(g["timestamp"], g["p_non0"], label="p_non0", linewidth=2.0)
    plt.step(g["timestamp"], g["y_true"], where="mid", label="y_true", alpha=0.6)
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _plot_timeline(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        plt.figure(figsize=(8, 3))
        plt.title("No data")
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        return
    g = frame.sort_values(["scenario_id", "timestamp"]).reset_index(drop=True)
    plt.figure(figsize=(12, 4))
    plt.step(g["timestamp"], g["y_true"], where="mid", label="true")
    plt.step(g["timestamp"], g["y_pred_stable"], where="mid", label="pred")
    plt.fill_between(g["timestamp"], 0.0, g["p_non0"], step="mid", alpha=0.20, label="p_non0")
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _interval_alignment(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario_id, g in frame.groupby("scenario_id"):
        gg = g.sort_values("timestamp").reset_index(drop=True)
        y_true = gg["y_true"].to_numpy(dtype=int)
        y_pred = gg["y_pred_stable"].to_numpy(dtype=int)
        ts = gg["timestamp"].to_numpy(dtype=float)
        true_intervals = _extract_intervals(y_true, 1)
        pred_intervals = _extract_intervals(y_pred, 1)
        matched_pred: set[int] = set()
        for i, (s, e) in enumerate(true_intervals):
            best_idx = None
            best_ov = 0
            for j, (ps, pe) in enumerate(pred_intervals):
                ov = max(0, min(e, pe) - max(s, ps) + 1)
                if ov > best_ov:
                    best_ov = ov
                    best_idx = j
            if best_idx is not None and best_ov > 0:
                matched_pred.add(best_idx)
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "interval_role": "true",
                    "interval_index": i,
                    "start_timestamp": float(ts[s]),
                    "end_timestamp": float(ts[e]),
                    "status": "matched" if best_idx is not None and best_ov > 0 else "missed",
                    "best_match_index": best_idx,
                    "overlap_frames": int(best_ov),
                }
            )
        for j, (ps, pe) in enumerate(pred_intervals):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "interval_role": "predicted",
                    "interval_index": j,
                    "start_timestamp": float(ts[ps]),
                    "end_timestamp": float(ts[pe]),
                    "status": "matched" if j in matched_pred else "false_alarm",
                    "best_match_index": None,
                    "overlap_frames": None,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    config_dir = output_root / "config"
    models_dir = output_root / "models"
    metrics_dir = output_root / "metrics"
    plots_dir = output_root / "plots"
    report_dir = output_root / "report"
    metadata_dir = output_root / "metadata"
    for d in [config_dir, models_dir, metrics_dir, plots_dir, report_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    aux = HybridEventDetector.load(args.m10_model_path) if args.m10_model_path.exists() else None
    builder = DetectorDatasetBuilder(aux.config if aux is not None else HybridEventDetector().config)
    if aux is not None and aux.preprocessing_state is not None:
        builder.normalization_state = aux.preprocessing_state  # type: ignore[assignment]
        builder.feature_columns = list(aux.feature_names)

    train_records = load_split_csv(args.train_split, split_name="train")
    val_records = load_split_csv(args.val_split, split_name="val")
    test_records = load_split_csv(args.test_split, split_name="test")
    train_ds = builder.build_for_records(train_records, fit=(builder.normalization_state is None))
    val_ds = builder.build_for_records(val_records, fit=False)
    test_ds = builder.build_for_records(test_records, fit=False)
    train_input = merge_detection_inputs(train_ds.by_scenario.values(), scenario_id="train", split="train")
    val_input = merge_detection_inputs(val_ds.by_scenario.values(), scenario_id="val", split="val")
    test_input = merge_detection_inputs(test_ds.by_scenario.values(), scenario_id="test", split="test")
    normal_mask = pd.to_numeric(train_input.metadata.get("y_binary", pd.Series([], dtype=int)), errors="coerce").fillna(0).astype(int).eq(0).to_numpy()
    normal_train_input = _subset_inputs(train_input, normal_mask, scenario_id="train_normal", split="train")

    detector = Non0DetectorV2(auxiliary_m10=aux)
    detector.fit(normal_train_input)
    evaluator = BinaryDetectorEvaluator()

    # RAW inputs + fixed interval-aware split maps.
    raw_inputs: list[DetectionInput] = []
    split_rows: list[pd.DataFrame] = []
    raw_dev_parts: list[DetectionInput] = []
    raw_holdout_parts: list[DetectionInput] = []
    for ref in load_raw_holdout_frames(args.raw_input_root):
        ds = builder.build_for_records(
            [
                SplitRecord(
                    scenario_id=ref.scenario_id,
                    scenario_dir=ref.scenario_dir,
                    split="raw",
                    template_name="RAW_HOLDOUT",
                    event_coarse=None,
                    difficulty_level="external_holdout",
                    scenario_family="raw_holdout",
                    seed_family="",
                )
            ],
            fit=False,
        )
        inp = ds.by_scenario.get(ref.scenario_id)
        if inp is None or inp.x_windows.shape[0] == 0:
            continue
        y_true = pd.to_numeric(inp.metadata.get("event_coarse", pd.Series(0, index=inp.metadata.index)), errors="coerce").fillna(0).astype(int).gt(0).astype(int).to_numpy()
        masks = _build_interval_aware_split(
            y_true,
            inp.timestamps,
            scenario_id=ref.scenario_id,
            guard_band=max(16, int(args.raw_guard_band_frames)),
            holdout_interval_mod=max(2, int(args.holdout_interval_mod)),
            holdout_target_normal_ratio=float(args.holdout_target_normal_ratio),
        )
        split_rows.append(masks.interval_rows)
        raw_inputs.append(inp)
        raw_dev_parts.append(_subset_inputs(inp, masks.dev_mask, scenario_id=ref.scenario_id, split="raw_dev"))
        raw_holdout_parts.append(_subset_inputs(inp, masks.holdout_mask, scenario_id=ref.scenario_id, split="raw_holdout"))

    raw_full_input = merge_detection_inputs(raw_inputs, scenario_id="raw_full", split="raw") if raw_inputs else DetectionInput("raw_full", "raw", np.zeros((0, 1, 1)), np.zeros((0,)), [], pd.DataFrame())
    raw_dev_input = merge_detection_inputs(raw_dev_parts, scenario_id="raw_dev", split="raw_dev") if raw_dev_parts else DetectionInput("raw_dev", "raw_dev", np.zeros((0, 1, 1)), np.zeros((0,)), [], pd.DataFrame())
    raw_holdout_input = merge_detection_inputs(raw_holdout_parts, scenario_id="raw_holdout", split="raw_holdout") if raw_holdout_parts else DetectionInput("raw_holdout", "raw_holdout", np.zeros((0, 1, 1)), np.zeros((0,)), [], pd.DataFrame())
    split_intervals = pd.concat(split_rows, ignore_index=True) if split_rows else pd.DataFrame()
    split_intervals.loc[split_intervals["split"] == "raw_dev"].to_csv(metadata_dir / "raw_dev_intervals.csv", index=False)
    split_intervals.loc[split_intervals["split"] == "raw_holdout"].to_csv(metadata_dir / "raw_holdout_intervals.csv", index=False)

    # Tune fusion/postprocessing on synthetic val + RAW_DEV only.
    sweep_rows: list[dict[str, Any]] = []
    best_cfg = None
    best_obj = -1e18
    for cfg in _sweep_configs():
        _apply_config(detector, cfg)
        val_out = detector.predict(val_input)
        raw_dev_out = detector.predict(raw_dev_input)
        y_val = pd.to_numeric(val_input.metadata.get("y_binary", pd.Series([], dtype=int)), errors="coerce").fillna(0).astype(int).to_numpy()
        y_raw_dev = pd.to_numeric(raw_dev_input.metadata.get("event_coarse", pd.Series([], dtype=int)), errors="coerce").fillna(0).astype(int).gt(0).astype(int).to_numpy()
        val_frame = _enrich(build_frame_output(val_out, y_val, val_input.timestamps, val_input.metadata), val_out.diagnostics)
        raw_dev_frame = _enrich(build_frame_output(raw_dev_out, y_raw_dev, raw_dev_input.timestamps, raw_dev_input.metadata), raw_dev_out.diagnostics)
        val_m = _eval_bundle(evaluator, val_frame)
        raw_dev_m = _eval_bundle(evaluator, raw_dev_frame)
        obj = (
            1.1 * float(raw_dev_m["recall_abnormal"])
            + 0.6 * float(val_m["recall_abnormal"])
            + 0.4 * float(raw_dev_m["f1_abnormal"])
            - 0.18 * float(raw_dev_m["false_positives_per_minute"])
            - 0.07 * float(val_m["false_positives_per_minute"])
        )
        row = {
            **asdict(cfg),
            **_metric_row("val", val_m),
            **_metric_row("raw_dev", raw_dev_m),
            "objective": float(obj),
        }
        sweep_rows.append(row)
        if obj > best_obj:
            best_obj = obj
            best_cfg = cfg
    sweep = pd.DataFrame(sweep_rows).sort_values("objective", ascending=False).reset_index(drop=True)
    sweep.to_csv(metrics_dir / "threshold_sweep_v5.csv", index=False)
    if best_cfg is None:
        best_cfg = _sweep_configs()[0]
    _apply_config(detector, best_cfg)
    (config_dir / "fusion_threshold_config_v3.json").write_text(json.dumps(asdict(best_cfg), indent=2), encoding="utf-8")

    # Final evaluation with tuned config.
    test_out = detector.predict(test_input)
    raw_full_out = detector.predict(raw_full_input)
    raw_dev_out = detector.predict(raw_dev_input)
    raw_holdout_out = detector.predict(raw_holdout_input)

    y_test = pd.to_numeric(test_input.metadata.get("y_binary", pd.Series([], dtype=int)), errors="coerce").fillna(0).astype(int).to_numpy()
    y_raw_full = pd.to_numeric(raw_full_input.metadata.get("event_coarse", pd.Series([], dtype=int)), errors="coerce").fillna(0).astype(int).gt(0).astype(int).to_numpy()
    y_raw_dev = pd.to_numeric(raw_dev_input.metadata.get("event_coarse", pd.Series([], dtype=int)), errors="coerce").fillna(0).astype(int).gt(0).astype(int).to_numpy()
    y_raw_holdout = pd.to_numeric(raw_holdout_input.metadata.get("event_coarse", pd.Series([], dtype=int)), errors="coerce").fillna(0).astype(int).gt(0).astype(int).to_numpy()

    test_frame = _enrich(build_frame_output(test_out, y_test, test_input.timestamps, test_input.metadata), test_out.diagnostics)
    raw_full_frame = _enrich(build_frame_output(raw_full_out, y_raw_full, raw_full_input.timestamps, raw_full_input.metadata), raw_full_out.diagnostics)
    raw_dev_frame = _enrich(build_frame_output(raw_dev_out, y_raw_dev, raw_dev_input.timestamps, raw_dev_input.metadata), raw_dev_out.diagnostics)
    raw_holdout_frame = _enrich(build_frame_output(raw_holdout_out, y_raw_holdout, raw_holdout_input.timestamps, raw_holdout_input.metadata), raw_holdout_out.diagnostics)

    synthetic_metrics = _eval_bundle(evaluator, test_frame)
    synthetic_family = evaluator.familywise_metrics(test_frame)
    raw_full_metrics = _eval_bundle(evaluator, raw_full_frame)
    raw_dev_metrics = _eval_bundle(evaluator, raw_dev_frame)
    raw_holdout_metrics = _eval_bundle(evaluator, raw_holdout_frame)
    per_scenario = evaluator.per_scenario_metrics(raw_holdout_frame) if not raw_holdout_frame.empty else pd.DataFrame()

    # Reports for protocol quality.
    q_full = _eval_split_quality(raw_full_frame, split_intervals, split_name="raw_full")
    q_dev = _eval_split_quality(raw_dev_frame, split_intervals, split_name="raw_dev")
    q_hold = _eval_split_quality(raw_holdout_frame, split_intervals, split_name="raw_holdout")
    quality_flags = {
        "holdout_has_normal_context": bool(q_hold["normal_windows"] >= 200),
        "holdout_has_abnormal_intervals": bool(q_hold["abnormal_intervals"] >= 1),
        "holdout_not_abnormal_dominated": bool(q_hold["normal_ratio"] >= 0.20),
        "holdout_duration_minutes_minimum": bool(q_hold["duration_minutes"] >= 5.0),
    }
    raw_eval_valid = bool(all(quality_flags.values()))
    raw_eval_quality_report = {
        "raw_full": q_full,
        "raw_dev": q_dev,
        "raw_holdout": q_hold,
        "quality_flags": quality_flags,
        "raw_eval_split_design_valid_for_deployment_judgement": raw_eval_valid,
    }
    write_json(report_dir / "raw_eval_quality_report.json", raw_eval_quality_report)
    (report_dir / "raw_eval_quality_report.md").write_text(
        "\n".join(
            [
                "# RAW Eval Quality Report",
                f"- split_design_valid: `{raw_eval_valid}`",
                f"- holdout_normal_windows: `{q_hold['normal_windows']}`",
                f"- holdout_abnormal_windows: `{q_hold['abnormal_windows']}`",
                f"- holdout_normal_ratio: `{q_hold['normal_ratio']:.4f}`",
                f"- holdout_abnormal_intervals: `{q_hold['abnormal_intervals']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Fusion ablation from sweep top rows.
    fusion_ablation = sweep.head(12).copy()
    fusion_ablation.to_csv(metrics_dir / "fusion_ablation.csv", index=False)

    # Branch trigger forensics.
    forensics = raw_full_frame.loc[raw_full_frame["y_pred_stable"].astype(int) == 1].copy()
    if not forensics.empty:
        forensics["false_alarm"] = (forensics["y_true"].astype(int) == 0).astype(int)
        cols = [
            "scenario_id",
            "timestamp",
            "y_true",
            "y_pred_stable",
            "false_alarm",
            "p_event5",
            "p_event7",
            "p_physical",
            "p_non0",
            "fusion_attribution",
            "fusion_trigger_reason",
            "event7_mode",
            "event5_state",
            "state_machine_state",
        ]
        forensics = forensics[[c for c in cols if c in forensics.columns]]
    forensics.to_csv(metrics_dir / "branch_trigger_forensics.csv", index=False)

    # Event7 ablation by threshold.
    e7_rows: list[dict[str, Any]] = []
    for thr in [0.35, 0.45, 0.55, 0.65, 0.75]:
        event_coarse = pd.to_numeric(raw_full_frame.get("event_coarse", pd.Series(0, index=raw_full_frame.index)), errors="coerce").fillna(0).astype(int)
        y_e7 = (event_coarse == 7).to_numpy(dtype=int)
        pred = (raw_full_frame["p_event7"].to_numpy(dtype=float) >= thr).astype(int)
        tp = int(((pred == 1) & (y_e7 == 1)).sum())
        fp = int(((pred == 1) & (y_e7 == 0)).sum())
        fn = int(((pred == 0) & (y_e7 == 1)).sum())
        precision = float(tp / max(tp + fp, 1))
        recall = float(tp / max(tp + fn, 1))
        e7_rows.append({"event7_threshold": thr, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn})
    event7_ablation = pd.DataFrame(e7_rows)
    event7_ablation.to_csv(metrics_dir / "event7_branch_ablation.csv", index=False)

    # Write required metrics artifacts.
    raw_holdout_frame.to_csv(metrics_dir / "raw_frame_predictions_v3.csv", index=False)
    raw_alignment = _interval_alignment(raw_holdout_frame)
    raw_alignment.to_csv(metrics_dir / "raw_interval_alignment_v3.csv", index=False)
    per_scenario.to_csv(metrics_dir / "per_scenario_metrics_v3.csv", index=False)
    plot_confusion_matrix(raw_holdout_metrics["confusion"], plots_dir / "confusion_matrix_v3.png")
    _plot_timeline(raw_holdout_frame, plots_dir / "raw_timeline_overlay_v3.png")
    _plot_scores(raw_holdout_frame, plots_dir / "branch_score_over_time_v3.png")

    # Before/after.
    baseline_v2 = {}
    if args.baseline_v2_report.exists():
        baseline_v2 = json.loads(args.baseline_v2_report.read_text(encoding="utf-8"))
    baseline_raw = {}
    if args.baseline_raw_report.exists():
        baseline_raw = json.loads(args.baseline_raw_report.read_text(encoding="utf-8"))
    raw_before_global = baseline_raw.get("metrics", {}).get("global", {})
    synth_before = baseline_v2.get("synthetic_metrics", {}).get("test", {})

    usable_on_raw = bool(
        raw_eval_valid
        and float(raw_holdout_metrics.get("false_positives_per_minute", 0.0) or 0.0) <= 5.0
        and float(raw_holdout_metrics.get("recall_abnormal", 0.0) or 0.0) > 0.5
        and float(raw_holdout_metrics.get("precision_abnormal", 0.0) or 0.0) >= 0.4
    )
    detects_non0 = bool(float(raw_holdout_metrics.get("recall_abnormal", 0.0) or 0.0) > 0.0)
    main_failures: list[str] = []
    if not raw_eval_valid:
        main_failures.append("RAW holdout protocol quality checks failed")
    if float(raw_full_metrics.get("false_positives_per_minute", 0.0) or 0.0) > 10.0:
        main_failures.append("RAW_FULL false positives are still too high")
    if float(raw_holdout_metrics.get("false_positives_per_minute", 0.0) or 0.0) > 5.0:
        main_failures.append("RAW_HOLDOUT false positives remain high")
    if float(raw_holdout_metrics.get("recall_abnormal", 0.0) or 0.0) <= 0.5:
        main_failures.append("RAW_HOLDOUT recall is below target")
    if not event7_ablation.empty and float(event7_ablation.loc[event7_ablation["event7_threshold"] == 0.55, "recall"].iloc[0]) < 0.10:
        main_failures.append("Event7 branch is still weak around operational threshold")

    report = {
        "raw_eval_quality": raw_eval_quality_report,
        "fusion_changes": {
            "branch_specific_thresholds": asdict(best_cfg),
            "quiet_state_veto_added": True,
            "persistence_added": True,
            "artifacts": {
                "fusion_ablation": str(metrics_dir / "fusion_ablation.csv"),
                "branch_trigger_forensics": str(metrics_dir / "branch_trigger_forensics.csv"),
            },
        },
        "event7_changes": {
            "added_excess_over_normal_quantile_gating": True,
            "added_persistence_suppression": True,
            "artifacts": {
                "event7_branch_ablation": str(metrics_dir / "event7_branch_ablation.csv"),
            },
        },
        "synthetic_metrics_before_after": {
            "before": synth_before,
            "after": synthetic_metrics,
        },
        "raw_metrics_before_after": {
            "before_raw_full": raw_before_global,
            "after_raw_full": raw_full_metrics,
            "after_raw_dev": raw_dev_metrics,
            "after_raw_holdout": raw_holdout_metrics,
        },
        "synthetic_metrics": {"test": synthetic_metrics, "familywise": synthetic_family},
        "raw_metrics": {"raw_full": raw_full_metrics, "raw_dev": raw_dev_metrics, "raw_holdout": raw_holdout_metrics},
        "verdict": {
            "detects_non0_on_raw": detects_non0,
            "usable_on_raw": usable_on_raw,
            "main_failures": main_failures,
            "next_actions": [
                "raise Event7 replay/stuck persistence realism in simulator and branch calibration",
                "add normal hard-negative RAW-like chunks to synthetic validation for anti-FP tuning",
            ],
        },
    }
    write_json(metrics_dir / "non0_detector_v2_report_v3.json", report)
    write_json(
        metrics_dir / "raw_holdout_v3_report.json",
        {
            "raw_eval_quality": raw_eval_quality_report,
            "before_raw_full": raw_before_global,
            "after_raw_full": raw_full_metrics,
            "after_raw_dev": raw_dev_metrics,
            "after_raw_holdout": raw_holdout_metrics,
            "verdict": report["verdict"],
        },
    )
    (metrics_dir / "non0_detector_v2_report_v3.md").write_text(
        "\n".join(
            [
                "# Non-0 Detector V2 Report V3",
                f"- raw_eval_trustworthy: `{raw_eval_valid}`",
                f"- synthetic_fp_per_min: `{float(synthetic_metrics.get('false_positives_per_minute', 0.0) or 0.0):.4f}`",
                f"- raw_full_fp_per_min: `{float(raw_full_metrics.get('false_positives_per_minute', 0.0) or 0.0):.4f}`",
                f"- raw_holdout_fp_per_min: `{float(raw_holdout_metrics.get('false_positives_per_minute', 0.0) or 0.0):.4f}`",
                f"- raw_holdout_recall: `{float(raw_holdout_metrics.get('recall_abnormal', 0.0) or 0.0):.4f}`",
                f"- usable_on_raw: `{usable_on_raw}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "fusion_hardening_report.md").write_text(
        "\n".join(
            [
                "# Fusion Hardening Report",
                "- Added branch-specific trigger thresholds.",
                "- Added branch-agreement gating and quiet-state veto.",
                "- Added stronger persistence constraints in state machine.",
                f"- Selected config: `{json.dumps(asdict(best_cfg))}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (report_dir / "event7_hardening_report.md").write_text(
        "\n".join(
            [
                "# Event7 Hardening Report",
                "- Event7 now uses excess-over-normal quantile gating (jump/outlier/stuck/drift).",
                "- Added persistence suppression to reduce moderate baseline firing.",
                f"- Best event7 ablation row: `{event7_ablation.sort_values('recall', ascending=False).head(1).to_dict(orient='records')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
