from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.detectors.domain.models.detector_config import DetectorConfigV2
from src.detectors.domain.services.hybrid_event_detector import HybridEventDetector
from src.detectors.pipelines.common import build_frame_output, write_json
from src.detectors.training.datasets import (
    DetectorDatasetBuilder,
    discover_scenario_pool,
    export_rebuilt_splits,
    load_split_csv,
    load_split_frames,
    merge_detection_inputs,
    rebuild_split_v3,
)
from src.detectors.training.evaluators import CalibratorFactory, ThresholdTuner, save_calibrator
from src.detectors.training.evaluators.calibration_evaluator import CalibrationEvaluator
from src.detectors.training.evaluators.detector_evaluator import BinaryDetectorEvaluator
from src.detectors.training.evaluators.plotting import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_curve,
    plot_familywise_metrics,
    plot_per_scenario_f1,
    plot_threshold_tradeoff,
)
from src.detectors.training.reports.detector_report_builder import write_detector_training_report
from src.simulation.m9.targeted_ready import generate_targeted_ready_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M10/M10.1 hardened detector trainer")
    p.add_argument("--train-split", type=Path, default=Path("train_scenarios_v2.csv"))
    p.add_argument("--val-split", type=Path, default=Path("val_scenarios_v2.csv"))
    p.add_argument("--test-split", type=Path, default=Path("test_scenarios_v2.csv"))
    p.add_argument("--workspace-root", type=Path, default=Path("."))
    p.add_argument("--raw-path", type=Path, default=Path("data/metadata/IEEE_39_Bus_Power_System.raw"))
    p.add_argument("--pmu-location-path", type=Path, default=Path("data/metadata/PMUbus_ Location.txt"))
    p.add_argument("--reference-pmu-dir", type=Path, default=Path("data/RAW0001"))
    p.add_argument("--output-root", type=Path, default=Path("output/detector_m10_2_ready"))
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--window-stride", type=int, default=16)
    p.add_argument("--positive-ratio-threshold", type=float, default=0.10)
    p.add_argument("--hardening-mode", action="store_true", default=True)
    p.add_argument("--rebuild-splits-v3", action="store_true", default=True)
    p.add_argument("--skip-rebuild-splits-v3", dest="rebuild_splits_v3", action="store_false")
    p.add_argument("--expand-scenario-pool", action="store_true", default=False)
    p.add_argument("--additional-val-per-family", type=int, default=0)
    p.add_argument("--additional-test-per-family", type=int, default=0)
    p.add_argument("--preprocessing-modes", type=str, default="ffill_bfill,ffill_limit1,bfill_only")
    p.add_argument("--f1-abnormal-min", type=float, default=0.85)
    p.add_argument("--recall-abnormal-min", type=float, default=0.80)
    p.add_argument("--false-positives-per-minute-max", type=float, default=1.5)
    p.add_argument("--detection-delay-max", type=float, default=1.5)
    p.add_argument("--calibration-max-ece", type=float, default=0.10)
    p.add_argument("--cyber-family-min-recall", type=float, default=0.70)
    p.add_argument("--physical-family-min-recall", type=float, default=0.75)
    p.add_argument("--concurrent-family-min-recall", type=float, default=0.70)
    p.add_argument("--generate-targeted-ready-data", action="store_true", default=True)
    p.add_argument("--skip-targeted-ready-data", dest="generate_targeted_ready_data", action="store_false")
    p.add_argument("--targeted-seed", type=int, default=20260420)
    return p.parse_args()


def _policy_apply(detector: HybridEventDetector, selected: dict[str, Any]) -> None:
    decision_threshold = float(selected.get("threshold", selected.get("start_threshold", 0.5)))
    start_threshold = float(selected.get("start_threshold", decision_threshold))
    stop_threshold = float(selected.get("stop_threshold", max(0.01, start_threshold - 0.10)))
    min_on_frames = int(selected.get("min_on_frames", 2))
    min_off_frames = int(selected.get("min_off_frames", 2))
    warmup_frames = int(selected.get("warmup_frames", detector.state_machine.warmup_frames))
    quiet_cyber_max = float(selected.get("quiet_cyber_max", detector.state_machine.quiet_cyber_max))
    quiet_physical_max = float(selected.get("quiet_physical_max", detector.state_machine.quiet_physical_max))
    quiet_abnormal_max = float(selected.get("quiet_abnormal_max", detector.state_machine.quiet_abnormal_max))
    quiet_frames_to_reset = int(selected.get("quiet_frames_to_reset", detector.state_machine.quiet_frames_to_reset))

    detector.fusion.decision_threshold = decision_threshold
    detector.state_machine.start_threshold = start_threshold
    detector.state_machine.stop_threshold = stop_threshold
    detector.state_machine.min_on_frames = min_on_frames
    detector.state_machine.min_off_frames = min_off_frames
    detector.state_machine.warmup_frames = warmup_frames
    detector.state_machine.quiet_cyber_max = quiet_cyber_max
    detector.state_machine.quiet_physical_max = quiet_physical_max
    detector.state_machine.quiet_abnormal_max = quiet_abnormal_max
    detector.state_machine.quiet_frames_to_reset = quiet_frames_to_reset
    detector.config.fusion.decision_threshold = decision_threshold
    detector.config.postprocessing.start_threshold = start_threshold
    detector.config.postprocessing.stop_threshold = stop_threshold
    detector.config.postprocessing.min_on_frames = min_on_frames
    detector.config.postprocessing.min_off_frames = min_off_frames
    detector.config.postprocessing.warmup_frames = warmup_frames
    detector.config.postprocessing.quiet_cyber_max = quiet_cyber_max
    detector.config.postprocessing.quiet_physical_max = quiet_physical_max
    detector.config.postprocessing.quiet_abnormal_max = quiet_abnormal_max
    detector.config.postprocessing.quiet_frames_to_reset = quiet_frames_to_reset


def _family_from_event(event: int) -> str:
    if int(event) == 0:
        return "normal"
    if int(event) in {1, 2, 3, 4}:
        return "physical_heavy"
    if int(event) in {5, 7}:
        return "cyber_heavy"
    if int(event) in {6, 8}:
        return "concurrent_heavy"
    return "unknown"


def _combine_splits_with_targeted(
    *,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    targeted_records: pd.DataFrame,
    metadata_dir: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    frames = {"train": train_df.copy(), "val": val_df.copy(), "test": test_df.copy()}
    if not targeted_records.empty and "split" in targeted_records.columns:
        for split_name, split_frame in targeted_records.groupby("split"):
            frames[str(split_name)] = pd.concat([frames[str(split_name)], split_frame.drop(columns=["split"])], ignore_index=True)
    effective_targeted_records = targeted_records.copy()
    if effective_targeted_records.empty:
        rebuilt_targeted: list[pd.DataFrame] = []
        for split_name, split_frame in frames.items():
            if "targeted_family" not in split_frame.columns:
                continue
            family = split_frame["targeted_family"].fillna("").astype(str).str.strip()
            mask = family.ne("")
            if not mask.any():
                continue
            targeted_frame = split_frame.loc[mask].copy()
            targeted_frame["split"] = split_name
            rebuilt_targeted.append(targeted_frame)
        if rebuilt_targeted:
            effective_targeted_records = pd.concat(rebuilt_targeted, ignore_index=True)
    train_path = metadata_dir / "train_scenarios_v4.csv"
    val_path = metadata_dir / "val_scenarios_v4.csv"
    test_path = metadata_dir / "test_scenarios_v4.csv"
    frames["train"].to_csv(train_path, index=False)
    frames["val"].to_csv(val_path, index=False)
    frames["test"].to_csv(test_path, index=False)

    coverage_rows: list[dict[str, Any]] = []
    manifest_splits: dict[str, list[str]] = {}
    for split_name, frame in frames.items():
        manifest_splits[split_name] = frame["scenario_id"].astype(str).tolist()
        local = frame.copy()
        local["event_coarse"] = pd.to_numeric(local["event_coarse"], errors="coerce").fillna(0).astype(int)
        local["family"] = local["event_coarse"].apply(_family_from_event)
        for family, family_group in local.groupby("family"):
            coverage_rows.append(
                {
                    "split": split_name,
                    "family": family,
                    "scenarios": int(family_group["scenario_id"].nunique()),
                    "windows_expected_from_scenarios": int(family_group["scenario_id"].nunique()) * 19,
                    "targeted_scenarios": int(family_group.get("targeted_family", pd.Series(dtype=str)).notna().sum()) if "targeted_family" in family_group.columns else 0,
                }
            )
        if "targeted_family" in local.columns:
            for targeted_family, fam_group in local.loc[local["targeted_family"].notna()].groupby("targeted_family"):
                coverage_rows.append(
                    {
                        "split": split_name,
                        "family": f"targeted::{targeted_family}",
                        "scenarios": int(fam_group["scenario_id"].nunique()),
                        "windows_expected_from_scenarios": int(fam_group["scenario_id"].nunique()) * 19,
                        "targeted_scenarios": int(fam_group["scenario_id"].nunique()),
                    }
                )

    coverage_frame = pd.DataFrame(coverage_rows).sort_values(["split", "family"]).reset_index(drop=True)
    coverage_path = metrics_dir = metadata_dir.parent / "metrics" / "family_coverage_report_v2.csv"
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_frame.to_csv(coverage_path, index=False)

    targeted_counts = (
        effective_targeted_records.groupby(["targeted_family", "split"])["scenario_id"].nunique().reset_index(name="scenarios")
        if not effective_targeted_records.empty
        else pd.DataFrame(columns=["targeted_family", "split", "scenarios"])
    )
    manifest = {
        "version": 4,
        "counts": {split: int(len(frame)) for split, frame in frames.items()},
        "scenario_ids": manifest_splits,
        "targeted_counts_by_family": (
            effective_targeted_records.groupby("targeted_family")["scenario_id"].nunique().to_dict()
            if not effective_targeted_records.empty and "targeted_family" in effective_targeted_records.columns
            else {}
        ),
        "targeted_split_counts": targeted_counts.to_dict(orient="records"),
        "total_targeted_scenarios": int(effective_targeted_records["scenario_id"].nunique()) if not effective_targeted_records.empty else 0,
    }
    manifest_path = metadata_dir / "split_manifest_v4.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return train_path, val_path, test_path, manifest_path, coverage_path, effective_targeted_records


def _build_normal_fp_forensics(frame: pd.DataFrame, *, quiet_cyber_max: float, quiet_physical_max: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for scenario_id, group in frame.groupby("scenario_id"):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        abnormal_before = ordered["y_true"].shift(1, fill_value=0).astype(int).to_numpy()
        stable = ordered["y_pred_stable"].astype(int).to_numpy()
        truth = ordered["y_true"].astype(int).to_numpy()
        quiet = (
            (ordered["p_cyber"].astype(float) <= quiet_cyber_max)
            & (ordered["p_physical"].astype(float) <= quiet_physical_max)
            & (ordered["p_abnormal"].astype(float) <= 0.66)
        )
        for idx, row in ordered.iterrows():
            if int(truth[idx]) != 0 or int(stable[idx]) != 1:
                continue
            root_cause = "post_event_recovery_tail"
            if not bool(abnormal_before[idx]):
                root_cause = "quiet_state_false_alarm"
            elif bool(quiet.iloc[idx]):
                root_cause = "latched_after_event_despite_quiet_branches"
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "timestamp": float(row["timestamp"]),
                    "template_name": str(row.get("template_name", "")),
                    "event_family": str(row.get("event_family", "")),
                    "targeted_family": str(row.get("targeted_family", "")),
                    "targeted_variant": str(row.get("targeted_variant", "")),
                    "p_abnormal": float(row["p_abnormal"]),
                    "p_cyber": float(row["p_cyber"]),
                    "p_physical": float(row["p_physical"]),
                    "data_present_ratio": float(row.get("data_present_ratio", 0.0)),
                    "root_cause": root_cause,
                    "quiet_branch_agreement": bool(quiet.iloc[idx]),
                    "preceded_by_abnormal_window": bool(abnormal_before[idx]),
                }
            )
    return pd.DataFrame(rows)


def _evaluate(
    detector: HybridEventDetector,
    detection_input,
    evaluator: BinaryDetectorEvaluator,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    output = detector.predict(detection_input)
    y_true = detection_input.metadata["y_binary"].to_numpy(dtype=int)
    scenario_ids = detection_input.metadata["scenario_id"].to_numpy() if "scenario_id" in detection_input.metadata.columns else None
    frame = build_frame_output(output, y_true, detection_input.timestamps, detection_input.metadata)
    metrics = evaluator.metric_bundle(
        y_true=y_true,
        y_pred=output.y_pred_stable,
        y_score=output.p_abnormal,
        timestamps=detection_input.timestamps,
        scenario_ids=scenario_ids,
    )
    branch = {
        "p_abnormal_mean": float(output.p_abnormal.mean()) if len(output.p_abnormal) else 0.0,
        "p_cyber_mean": float(output.p_cyber.mean()) if len(output.p_cyber) else 0.0,
        "p_physical_mean": float(output.p_physical.mean()) if len(output.p_physical) else 0.0,
        "cyber_backend": detector.cyber.model.backend,
        "physical_backend": detector.physical.temporal_model.backend,
    }
    return frame, metrics, branch


def _preprocessing_ablation(
    base_config: DetectorConfigV2,
    modes: list[str],
    train_records,
    val_records,
    *,
    workspace_root: Path,
) -> tuple[str, pd.DataFrame]:
    rows = []
    for mode in modes:
        config = DetectorConfigV2()
        config.window = base_config.window
        config.preprocessing = base_config.preprocessing
        config.preprocessing.fill_method = mode
        config.fusion = base_config.fusion
        config.postprocessing = base_config.postprocessing
        builder = DetectorDatasetBuilder(config)
        train_ds = builder.build_for_records(train_records, fit=True)
        val_ds = builder.build_for_records(val_records, fit=False)
        train_input = merge_detection_inputs(train_ds.by_scenario.values(), scenario_id="ABL_TRAIN", split="train")
        val_input = merge_detection_inputs(val_ds.by_scenario.values(), scenario_id="ABL_VAL", split="val")
        if train_input.x_windows.shape[0] == 0 or "y_binary" not in train_input.metadata.columns:
            rows.append(
                {
                    "fill_method": mode,
                    "f1_abnormal": 0.0,
                    "precision_abnormal": 0.0,
                    "recall_abnormal": 0.0,
                    "balanced_accuracy": 0.0,
                    "false_positives_per_minute": 0.0,
                    "detection_delay_s": None,
                    "roc_auc": None,
                    "pr_auc": None,
                    "brier_score": 1.0,
                    "support_abnormal": 0,
                    "support_normal": 0,
                    "confusion": {"tp": 0, "tn": 0, "fp": 0, "fn": 0},
                    "false_positives": 0,
                    "objective": -1.0,
                }
            )
            continue
        detector = HybridEventDetector(config)
        detector.set_preprocessing_state(builder.normalization_state, builder.feature_columns)
        detector.fit_inputs(train_input)
        val_output = detector.predict(val_input)
        y_true = val_input.metadata["y_binary"].to_numpy(dtype=int)
        metrics = BinaryDetectorEvaluator().metric_bundle(
            y_true=y_true,
            y_pred=val_output.y_pred_stable,
            y_score=val_output.p_abnormal,
            timestamps=val_input.timestamps,
            scenario_ids=val_input.metadata["scenario_id"].to_numpy() if "scenario_id" in val_input.metadata.columns else None,
        )
        objective = float(metrics["f1_abnormal"]) + 0.3 * float(metrics["recall_abnormal"]) - 0.05 * float(metrics["false_positives_per_minute"])
        rows.append({"fill_method": mode, **metrics, "objective": objective})
    frame = pd.DataFrame(rows).sort_values(["objective", "f1_abnormal", "recall_abnormal"], ascending=False).reset_index(drop=True)
    selected = str(frame.iloc[0]["fill_method"]) if not frame.empty else modes[0]
    return selected, frame


def _plot_calibration_before_after(before: pd.DataFrame, after: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    b = before.dropna(subset=["mean_pred", "empirical"]) if not before.empty else before
    a = after.dropna(subset=["mean_pred", "empirical"]) if not after.empty else after
    if not b.empty:
        ax.plot(b["mean_pred"], b["empirical"], marker="o", label="before")
    if not a.empty:
        ax.plot(a["mean_pred"], a["empirical"], marker="s", label="after")
    ax.set_title("Calibration Before vs After")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical abnormal frequency")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    config_dir = output_root / "config"
    models_dir = output_root / "models"
    metrics_dir = output_root / "metrics"
    plots_dir = output_root / "plots"
    report_dir = output_root / "report"
    metadata_dir = output_root / "metadata"
    for d in (config_dir, models_dir, metrics_dir, plots_dir, report_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    base_config = DetectorConfigV2()
    base_config.window.size = int(args.window_size)
    base_config.window.stride = int(args.window_stride)
    base_config.window.positive_ratio_threshold = float(args.positive_ratio_threshold)
    base_config.artifact_root = output_root

    train_df, val_df, test_df = load_split_frames(args.train_split, args.val_split, args.test_split)
    split_report_lines = ["# Split Rebuild Report", ""]
    scenario_generation_report: dict[str, Any] = {
        "generated": False,
        "new_scenarios": [],
        "counts_by_family": {},
    }
    if args.rebuild_splits_v3:
        extra_pool = pd.DataFrame()
        if args.expand_scenario_pool:
            existing_ids = pd.concat([train_df["scenario_id"], val_df["scenario_id"], test_df["scenario_id"]], ignore_index=True).astype(str).tolist()
            extra_pool = discover_scenario_pool(args.workspace_root.resolve(), existing_ids=existing_ids)
        rebuilt = rebuild_split_v3(
            train_df,
            val_df,
            test_df,
            min_val_per_family=1,
            min_test_per_family=1,
            additional_val_per_family=max(0, int(args.additional_val_per_family)),
            additional_test_per_family=max(0, int(args.additional_test_per_family)),
            extra_pool_df=extra_pool,
        )
        train_csv_v3, val_csv_v3, test_csv_v3, manifest_path, coverage_path = export_rebuilt_splits(rebuilt, metadata_dir)
        split_report_lines.extend(
            [
                f"- train_v3: `{train_csv_v3}`",
                f"- val_v3: `{val_csv_v3}`",
                f"- test_v3: `{test_csv_v3}`",
                f"- manifest_v3: `{manifest_path}`",
                f"- family_coverage: `{coverage_path}`",
                f"- expand_scenario_pool: `{bool(args.expand_scenario_pool)}`",
                f"- extra_pool_rows: `{int(len(extra_pool))}`",
                f"- additional_val_per_family: `{int(args.additional_val_per_family)}`",
                f"- additional_test_per_family: `{int(args.additional_test_per_family)}`",
            ]
        )
        base_train_df = pd.read_csv(train_csv_v3)
        base_val_df = pd.read_csv(val_csv_v3)
        base_test_df = pd.read_csv(test_csv_v3)
    else:
        manifest_path = metadata_dir / "split_manifest_v3.json"
        coverage_path = metrics_dir / "family_coverage_report.csv"
        base_train_df = train_df.copy()
        base_val_df = val_df.copy()
        base_test_df = test_df.copy()

    targeted_records = pd.DataFrame()
    if args.generate_targeted_ready_data:
        generated = generate_targeted_ready_dataset(
            workspace_root=args.workspace_root.resolve(),
            raw_path=args.raw_path,
            pmu_location_path=args.pmu_location_path,
            reference_pmu_dir=args.reference_pmu_dir,
            seed=int(args.targeted_seed),
        )
        targeted_records = generated["records"].copy()
        scenario_generation_report = {
            "generated": True,
            "counts_by_family": generated["counts_by_family"],
            "new_scenarios": generated["scenarios"].to_dict(orient="records"),
        }

    train_split_csv, val_split_csv, test_split_csv, manifest_path, coverage_path, effective_targeted_records = _combine_splits_with_targeted(
        train_df=base_train_df,
        val_df=base_val_df,
        test_df=base_test_df,
        targeted_records=targeted_records,
        metadata_dir=metadata_dir,
    )
    targeted_records = effective_targeted_records
    (report_dir / "split_rebuild_report.md").write_text(
        "\n".join(
            split_report_lines
            + [
                "",
                f"- train_v4: `{train_split_csv}`",
                f"- val_v4: `{val_split_csv}`",
                f"- test_v4: `{test_split_csv}`",
                f"- split_manifest_v4: `{manifest_path}`",
                f"- family_coverage_report_v2: `{coverage_path}`",
                f"- targeted_ready_generation_enabled: `{bool(args.generate_targeted_ready_data)}`",
                f"- targeted_ready_scenarios: `{int(len(targeted_records))}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    workspace_root = args.workspace_root.resolve()
    train_records = load_split_csv(train_split_csv, split_name="train", workspace_root=workspace_root)
    val_records = load_split_csv(val_split_csv, split_name="validation", workspace_root=workspace_root)
    test_records = load_split_csv(test_split_csv, split_name="test", workspace_root=workspace_root)

    modes = [m.strip() for m in args.preprocessing_modes.split(",") if m.strip()]
    selected_mode, prep_ablation = _preprocessing_ablation(base_config, modes, train_records, val_records, workspace_root=workspace_root)
    prep_ablation.to_csv(metrics_dir / "preprocessing_ablation.csv", index=False)
    (report_dir / "preprocessing_hardening.md").write_text(
        "\n".join(
            [
                "# Preprocessing Hardening",
                f"- selected_fill_method: `{selected_mode}`",
                "- compared_modes:",
                *[f"  - `{m}`" for m in modes],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = DetectorConfigV2()
    config.window = base_config.window
    config.preprocessing = base_config.preprocessing
    config.preprocessing.fill_method = selected_mode
    config.fusion = base_config.fusion
    config.postprocessing = base_config.postprocessing
    config.artifact_root = output_root

    builder = DetectorDatasetBuilder(config)
    builder.nan_count_before = 0
    builder.nan_count_after = 0
    train_ds = builder.build_for_records(train_records, fit=True)
    train_nan_before = int(builder.nan_count_before)
    train_nan_after = int(builder.nan_count_after)
    builder.nan_count_before = 0
    builder.nan_count_after = 0
    val_ds = builder.build_for_records(val_records, fit=False)
    val_nan_before = int(builder.nan_count_before)
    val_nan_after = int(builder.nan_count_after)
    builder.nan_count_before = 0
    builder.nan_count_after = 0
    test_ds = builder.build_for_records(test_records, fit=False)
    test_nan_before = int(builder.nan_count_before)
    test_nan_after = int(builder.nan_count_after)

    train_input = merge_detection_inputs(train_ds.by_scenario.values(), scenario_id="TRAIN", split="train")
    val_input = merge_detection_inputs(val_ds.by_scenario.values(), scenario_id="VAL", split="validation")
    test_input = merge_detection_inputs(test_ds.by_scenario.values(), scenario_id="TEST", split="test")

    detector = HybridEventDetector(config)
    detector.set_preprocessing_state(builder.normalization_state, builder.feature_columns)
    detector.fit_inputs(train_input)

    # Cyber branch hardening evidence artifacts.
    cyber_importance = pd.DataFrame(
        [{"feature": k, "importance": v} for k, v in detector.cyber.model.feature_importance().items()]
    )
    if cyber_importance.empty or "importance" not in cyber_importance.columns:
        cyber_importance = pd.DataFrame([{"feature": "<none>", "importance": 0.0}])
    else:
        cyber_importance = cyber_importance.sort_values("importance", ascending=False).reset_index(drop=True)
    cyber_importance.to_csv(metrics_dir / "cyber_branch_feature_importance.csv", index=False)

    cyber_batch = detector.cyber.extractor.extract(val_input)
    ml_prob = detector.cyber.model.predict_proba(cyber_batch.x)
    rule_prob = cyber_batch.rule_score
    fused_prob = detector.cyber.score(val_input).probability
    y_val = val_input.metadata["y_binary"].to_numpy(dtype=int)
    evalr = BinaryDetectorEvaluator()
    cyber_ablation = pd.DataFrame(
        [
            {
                "mode": "rule_only",
                **evalr.metric_bundle(
                    y_true=y_val,
                    y_pred=(rule_prob >= 0.5).astype(int),
                    y_score=rule_prob,
                    timestamps=val_input.timestamps,
                    scenario_ids=val_input.metadata["scenario_id"].to_numpy() if "scenario_id" in val_input.metadata.columns else None,
                ),
            },
            {
                "mode": "ml_only",
                **evalr.metric_bundle(
                    y_true=y_val,
                    y_pred=(ml_prob >= 0.5).astype(int),
                    y_score=ml_prob,
                    timestamps=val_input.timestamps,
                    scenario_ids=val_input.metadata["scenario_id"].to_numpy() if "scenario_id" in val_input.metadata.columns else None,
                ),
            },
            {
                "mode": "hybrid",
                **evalr.metric_bundle(
                    y_true=y_val,
                    y_pred=(fused_prob >= 0.5).astype(int),
                    y_score=fused_prob,
                    timestamps=val_input.timestamps,
                    scenario_ids=val_input.metadata["scenario_id"].to_numpy() if "scenario_id" in val_input.metadata.columns else None,
                ),
            },
        ]
    )
    cyber_ablation.to_csv(metrics_dir / "cyber_branch_ablation.csv", index=False)
    (report_dir / "cyber_branch_hardening.md").write_text(
        "# Cyber Branch Hardening\n- strengthened missingness/partial-dropout/stuck-after-missing features\n- increased cyber rule weight and strong-missing overrides\n",
        encoding="utf-8",
    )

    # Calibration selection on validation only.
    val_raw = detector.predict(val_input)
    cal_selection = CalibratorFactory.select_best(y_val, val_raw.p_abnormal)
    detector.calibrator = cal_selection.calibrator
    cal_selection.comparison.to_csv(metrics_dir / "calibration_comparison.csv", index=False)
    save_calibrator(models_dir / "calibrator.pkl", cal_selection.calibrator)
    write_json(config_dir / "calibration_config.json", {"selected_method": cal_selection.selected_name})

    cal_eval = CalibrationEvaluator(n_bins=10)
    before_cal = cal_eval.evaluate(y_val, val_raw.p_abnormal)
    after_cal = cal_eval.evaluate(y_val, detector.calibrator.transform(val_raw.p_abnormal))
    _plot_calibration_before_after(before_cal.bins, after_cal.bins, plots_dir / "calibration_curve_before_after.png")
    (report_dir / "calibration_hardening.md").write_text(
        f"# Calibration Hardening\n- selected_method: `{cal_selection.selected_name}`\n- val_ece_before: `{before_cal.ece:.4f}`\n- val_ece_after: `{after_cal.ece:.4f}`\n",
        encoding="utf-8",
    )

    # Tune thresholds on calibrated validation predictions only.
    val_cal = detector.predict(val_input)
    val_frame_for_tuning = build_frame_output(
        val_cal,
        y_val,
        val_input.timestamps,
        val_input.metadata,
    )
    threshold_tuner = ThresholdTuner()
    selected, sweep = threshold_tuner.tune(
        y_true=y_val,
        p_abnormal=val_cal.p_abnormal,
        p_cyber=val_cal.p_cyber,
        p_physical=val_cal.p_physical,
        frame_predictions=val_cal.y_pred_frame,
        timestamps=val_input.timestamps,
        scenario_ids=val_input.metadata["scenario_id"].to_numpy() if "scenario_id" in val_input.metadata.columns else None,
        frame=val_frame_for_tuning,
    )
    _policy_apply(detector, selected)
    sweep.to_csv(metrics_dir / "threshold_sweep_v4.csv", index=False)
    write_json(config_dir / "threshold_config_v4.json", selected)
    (report_dir / "fusion_threshold_hardening.md").write_text(
        "\n".join(
            [
                "# Fusion / Threshold Hardening",
                f"- selected_threshold: `{selected.get('threshold')}`",
                f"- start_threshold: `{selected.get('start_threshold')}`",
                f"- stop_threshold: `{selected.get('stop_threshold')}`",
                f"- min_on_frames: `{selected.get('min_on_frames')}`",
                f"- min_off_frames: `{selected.get('min_off_frames')}`",
                f"- quiet_cyber_max: `{selected.get('quiet_cyber_max')}`",
                f"- quiet_physical_max: `{selected.get('quiet_physical_max')}`",
                f"- quiet_abnormal_max: `{selected.get('quiet_abnormal_max')}`",
                f"- quiet_frames_to_reset: `{selected.get('quiet_frames_to_reset')}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    evaluator = BinaryDetectorEvaluator()
    train_frame, train_metrics, train_branch = _evaluate(detector, train_input, evaluator)
    val_frame, val_metrics, val_branch = _evaluate(detector, val_input, evaluator)
    test_frame, test_metrics, test_branch = _evaluate(detector, test_input, evaluator)
    familywise = evaluator.familywise_metrics(test_frame)
    per_scenario = pd.concat([evaluator.per_scenario_metrics(val_frame), evaluator.per_scenario_metrics(test_frame)], ignore_index=True)

    # Export core artifacts
    detector.save(models_dir / "detector_model.pkl")
    detector.cyber.model.save(models_dir / "cyber_model.pkl")
    detector.physical.temporal_model.save(models_dir / "physical_model.pt")
    detector.save(output_root / "hybrid_detector.pkl")

    train_frame.to_csv(metrics_dir / "train_frame_predictions.csv", index=False)
    val_frame.to_csv(metrics_dir / "validation_frame_predictions.csv", index=False)
    test_frame.to_csv(metrics_dir / "test_frame_predictions_v2.csv", index=False)
    test_frame.to_csv(metrics_dir / "hardened_test_frame_predictions_v4.csv", index=False)
    per_scenario.to_csv(metrics_dir / "per_scenario_metrics_v2.csv", index=False)
    per_scenario.to_csv(metrics_dir / "per_scenario_metrics_v4.csv", index=False)
    if not targeted_records.empty:
        targeted_records.to_csv(metadata_dir / "new_targeted_scenarios_v4.csv", index=False)
    if coverage_path.exists():
        coverage_frame = pd.read_csv(coverage_path)
        coverage_frame.to_csv(metrics_dir / "family_coverage_report.csv", index=False)
        coverage_frame.to_csv(metrics_dir / "family_coverage_report_v2.csv", index=False)

    roc_frame, pr_frame = evaluator.curve_points(test_frame["y_true"].to_numpy(dtype=int), test_frame["p_abnormal"].to_numpy(dtype=float))
    roc_frame.to_csv(metrics_dir / "roc_curve_points_v2.csv", index=False)
    pr_frame.to_csv(metrics_dir / "pr_curve_points_v2.csv", index=False)
    test_cal = CalibrationEvaluator(n_bins=10).evaluate(
        y_true=test_frame["y_true"].to_numpy(dtype=int),
        y_prob=test_frame["p_abnormal"].to_numpy(dtype=float),
    )

    plot_confusion_matrix(dict(test_metrics.get("confusion", {})), plots_dir / "confusion_matrix_v2.png")
    plot_curve(pr_frame, x="recall", y="precision", output_path=plots_dir / "pr_curve_v2.png", title="PR Curve", xlabel="Recall", ylabel="Precision")
    plot_curve(roc_frame, x="fpr", y="tpr", output_path=plots_dir / "roc_curve_v2.png", title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    plot_calibration_curve(test_cal.bins, plots_dir / "calibration_curve_v2.png")
    plot_threshold_tradeoff(sweep, plots_dir / "threshold_tradeoff_v2.png")
    plot_familywise_metrics(familywise, plots_dir / "familywise_metrics_v2.png")
    plot_per_scenario_f1(per_scenario, plots_dir / "per_scenario_f1_v2.png")

    normal_fp_forensics = _build_normal_fp_forensics(
        test_frame,
        quiet_cyber_max=float(selected.get("quiet_cyber_max", detector.state_machine.quiet_cyber_max)),
        quiet_physical_max=float(selected.get("quiet_physical_max", detector.state_machine.quiet_physical_max)),
    )
    normal_fp_forensics.to_csv(metrics_dir / "normal_fp_forensics_v2.csv", index=False)

    criteria = {
        "test_f1_abnormal_min": float(args.f1_abnormal_min),
        "test_recall_abnormal_min": float(args.recall_abnormal_min),
        "test_false_positives_per_minute_max": float(args.false_positives_per_minute_max),
        "test_detection_delay_s_max": float(args.detection_delay_max),
        "calibration_ece_test_max": float(args.calibration_max_ece),
        "cyber_family_recall_min": float(args.cyber_family_min_recall),
        "physical_family_recall_min": float(args.physical_family_min_recall),
        "concurrent_family_recall_min": float(args.concurrent_family_min_recall),
    }

    failed: list[str] = []
    passed: list[str] = []

    def check(name: str, ok: bool) -> None:
        (passed if ok else failed).append(name)

    check("test_f1_abnormal", float(test_metrics["f1_abnormal"]) >= criteria["test_f1_abnormal_min"])
    check("test_recall_abnormal", float(test_metrics["recall_abnormal"]) >= criteria["test_recall_abnormal_min"])
    check("test_false_positives_per_minute", float(test_metrics["false_positives_per_minute"]) <= criteria["test_false_positives_per_minute_max"])
    delay = float(test_metrics["detection_delay_s"]) if test_metrics.get("detection_delay_s") is not None else 1e9
    check("test_detection_delay_s", delay <= criteria["test_detection_delay_s_max"])
    check("calibration_ece_test", float(test_cal.ece) <= criteria["calibration_ece_test_max"])

    for fam, key, min_req in [
        ("cyber_heavy", "cyber_family_recall_min", criteria["cyber_family_recall_min"]),
        ("physical_heavy", "physical_family_recall_min", criteria["physical_family_recall_min"]),
        ("concurrent_heavy", "concurrent_family_recall_min", criteria["concurrent_family_recall_min"]),
    ]:
        fam_windows = int(familywise.get(fam, {}).get("windows", 0) or 0)
        fam_recall = float(familywise.get(fam, {}).get("recall_abnormal", 0.0) or 0.0)
        if fam_windows == 0:
            failed.append(f"{fam}_coverage_zero")
        else:
            check(f"{fam}_recall", fam_recall >= min_req)

    ready = len(failed) == 0
    verdict = "ready" if ready else ("needs_more_data" if any(k.endswith("coverage_zero") for k in failed) else "needs_more_hardening")

    previous_failed_criteria = [
        "test_false_positives_per_minute",
        "cyber_heavy_recall",
    ]
    detector_logic_changes = [
        "adaptive smoothing now reacts faster to strong cyber/physical evidence so cyber-heavy events trigger earlier",
        "state-machine quiet-state veto now clears latched post-event tails when both branches agree the system is quiet",
        "state-machine frame-agreement veto now drops post-event tails as soon as frame-level evidence returns to normal",
        "fusion now lets strong cyber evidence win sooner when the physical branch is weak",
        "exported model metadata now records the actual cyber/physical/fusion backends instead of placeholders",
        "validation threshold tuning now optimizes for normal-family FP suppression and cyber-heavy recall, not only global F1",
    ]
    scenario_generation_changes = [
        "added 60 targeted normal hard negatives with benign drift, ringdown-like, angle, and frequency wobble behavior",
        "added 30 targeted cyber-heavy missing scenarios spanning short/long bursts, partial dropout, and bursty periodic dropout",
        "added 30 targeted cyber-heavy bad-data scenarios spanning spikes, bias/drift, replay, stuck-after-replay, and timing corruption",
        "added 40 targeted concurrent scenarios with subtle physical events overlapped by missing or bad-data corruption",
    ]

    training_report_v4 = {
        "run_metadata": {
            "train_split": str(train_split_csv),
            "val_split": str(val_split_csv),
            "test_split": str(test_split_csv),
            "raw_path": str(args.raw_path),
            "pmu_location_path": str(args.pmu_location_path),
            "reference_pmu_dir": str(args.reference_pmu_dir),
            "output_root": str(output_root),
            "window_size": int(config.window.size),
            "window_stride": int(config.window.stride),
            "train_windows": int(train_input.x_windows.shape[0]),
            "val_windows": int(val_input.x_windows.shape[0]),
            "test_windows": int(test_input.x_windows.shape[0]),
        },
        "nan_and_data_present_handling": {
            "strategy": f"{selected_mode} + mask channels + z-score normalization",
            "data_present_feature_used": "DATA_PRESENT" in train_input.feature_names,
            "nan_count_before_preprocessing_train": train_nan_before,
            "nan_count_before_preprocessing_val": val_nan_before,
            "nan_count_before_preprocessing_test": test_nan_before,
            "nan_count_after_preprocessing_train": train_nan_after,
            "nan_count_after_preprocessing_val": val_nan_after,
            "nan_count_after_preprocessing_test": test_nan_after,
        },
        "threshold_tuning_validation": {
            "selected": selected,
            "top_candidates": sweep.head(10).to_dict(orient="records"),
            "rows": int(len(sweep)),
        },
        "detector_logic_changes": detector_logic_changes,
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": {**test_metrics, "brier_score": float(test_cal.brier)},
            "confusion_matrix_test": test_metrics.get("confusion", {}),
            "calibration_ece_test": float(test_cal.ece),
            "cyber_vs_physical": familywise,
            "raw_reference_normal_baseline": {"available": False},
        },
        "branch_summary": {
            "train": train_branch,
            "validation": val_branch,
            "test": test_branch,
        },
        "downstream_readiness": {
            "binary_detector_ready_for_classifier_localizer": ready,
            "verdict": verdict,
            "criteria": criteria,
        },
    }

    report_json, report_md = write_detector_training_report(training_report_v4, output_root=output_root)
    report_json_path = Path(report_json)
    report_md_path = Path(report_md)
    (metrics_dir / "detector_training_report_v4.json").write_text(report_json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (metrics_dir / "detector_training_report_v4.md").write_text(report_md_path.read_text(encoding="utf-8"), encoding="utf-8")
    (metrics_dir / "detector_training_report_v2.json").write_text(report_json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (metrics_dir / "detector_training_report_v2.md").write_text(report_md_path.read_text(encoding="utf-8"), encoding="utf-8")

    hardening_report = {
        "run_metadata": training_report_v4["run_metadata"],
        "split_rebuild": {
            "enabled": bool(args.rebuild_splits_v3),
            "manifest_path": str(manifest_path),
            "family_coverage_report": str(metrics_dir / "family_coverage_report_v2.csv"),
        },
        "preprocessing_ablation": {
            "selected_mode": selected_mode,
            "table_path": str(metrics_dir / "preprocessing_ablation.csv"),
        },
        "cyber_branch_hardening": {
            "feature_importance_path": str(metrics_dir / "cyber_branch_feature_importance.csv"),
            "ablation_path": str(metrics_dir / "cyber_branch_ablation.csv"),
            "notes": ["added missing-burst/asymmetry/partial-dropout/stuck-after-missing features", "increased cyber rule influence"],
        },
        "fusion_threshold_hardening": {
            "threshold_config": selected,
            "sweep_path": str(metrics_dir / "threshold_sweep_v4.csv"),
        },
        "calibration_hardening": {
            "selected_method": cal_selection.selected_name,
            "comparison_path": str(metrics_dir / "calibration_comparison.csv"),
            "ece_before": float(before_cal.ece),
            "ece_after": float(after_cal.ece),
        },
        "metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": {**test_metrics, "brier_score": float(test_cal.brier), "calibration_ece": float(test_cal.ece)},
            "familywise": familywise,
            "per_scenario_summary": {
                "rows": int(len(per_scenario)),
                "path": str(metrics_dir / "per_scenario_metrics_v4.csv"),
            },
        },
        "readiness": {
            "binary_detector_ready_for_classifier_localizer": ready,
            "criteria": criteria,
            "failed_criteria": failed,
            "passed_criteria": passed,
            "verdict": verdict,
        },
        "overall_conclusion": {
            "main_improvements": [
                "rebuilt split coverage with cyber/concurrent families in validation/test",
                "added targeted ready dataset focused on normal hard negatives, cyber-heavy, and concurrent overlap",
                "added mask-aware preprocessing ablation and selected mode",
                "strengthened cyber branch features and rule influence",
                "added calibration method comparison and selected calibrator",
                "added quiet-state veto and fast cyber activation in postprocessing",
            ],
            "remaining_weaknesses": failed,
            "next_actions": [
                "increase scenario diversity if family coverage remains sparse",
                "revisit branch-specific model capacity for underperforming families",
            ],
        },
    }
    write_json(report_dir / "m10_1_hardening_report.json", hardening_report)
    (report_dir / "m10_1_hardening_report.md").write_text(
        "\n".join(
            [
                "# M10.1 Hardening Report",
                f"- cyber/concurrent coverage fixed: `{not any(x.endswith('coverage_zero') for x in failed)}`",
                f"- recall improved target met: `{float(test_metrics['recall_abnormal']) >= criteria['test_recall_abnormal_min']}`",
                f"- delay improved target met: `{delay <= criteria['test_detection_delay_s_max']}`",
                f"- calibration improved target met: `{float(test_cal.ece) <= criteria['calibration_ece_test_max']}`",
                f"- detector ready for downstream gating: `{ready}`",
                f"- verdict: `{verdict}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    targeted_counts_by_family = scenario_generation_report.get("counts_by_family") or manifest_payload.get("targeted_counts_by_family", {})
    scenario_generation_json = {
        "generated_targeted_data": bool(scenario_generation_report.get("generated", False)),
        "what_changed_in_scenario_generation": scenario_generation_changes,
        "counts_by_family": targeted_counts_by_family,
        "new_scenarios": scenario_generation_report.get("new_scenarios", []),
        "split_manifest_v4": str(manifest_path),
        "updated_split_csvs": {
            "train": str(train_split_csv),
            "val": str(val_split_csv),
            "test": str(test_split_csv),
        },
        "family_coverage_report_v2": str(metrics_dir / "family_coverage_report_v2.csv"),
    }
    write_json(metrics_dir / "scenario_generation_patch_report.json", scenario_generation_json)
    (metrics_dir / "scenario_generation_patch_report.md").write_text(
        "\n".join(
            [
                "# Scenario Generation Patch Report",
                *[f"- {line}" for line in scenario_generation_changes],
                f"- split_manifest_v4: `{manifest_path}`",
                f"- new_targeted_scenarios: `{int(manifest_payload.get('total_targeted_scenarios', len(targeted_records)))}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    family_metrics = {
        "normal": familywise.get("normal", {}),
        "physical_heavy": familywise.get("physical_heavy", {}),
        "cyber_heavy": familywise.get("cyber_heavy", {}),
        "concurrent_heavy": familywise.get("concurrent_heavy", {}),
    }
    ready_report_json = {
        "previous_failed_criteria": previous_failed_criteria,
        "what_changed_in_detector_logic": detector_logic_changes,
        "what_changed_in_scenario_generation": scenario_generation_changes,
        "how_many_new_scenarios_were_generated_by_targeted_family": targeted_counts_by_family,
        "new_train_val_test_metrics": {
            "train": train_metrics,
            "validation": val_metrics,
            "test": {**test_metrics, "calibration_ece_test": float(test_cal.ece), "brier_score": float(test_cal.brier)},
        },
        "family_wise_metrics": family_metrics,
        "readiness_verdict": {
            "binary_detector_ready_for_classifier_localizer": ready,
            "verdict": verdict,
        },
        "artifacts": {
            "detector_training_report_v4": str(metrics_dir / "detector_training_report_v4.json"),
            "per_scenario_metrics_v4": str(metrics_dir / "per_scenario_metrics_v4.csv"),
            "threshold_sweep_v4": str(metrics_dir / "threshold_sweep_v4.csv"),
            "hardened_test_frame_predictions_v4": str(metrics_dir / "hardened_test_frame_predictions_v4.csv"),
            "family_coverage_report_v2": str(metrics_dir / "family_coverage_report_v2.csv"),
            "normal_fp_forensics_v2": str(metrics_dir / "normal_fp_forensics_v2.csv"),
            "scenario_generation_patch_report": str(metrics_dir / "scenario_generation_patch_report.json"),
            "split_manifest_v4": str(manifest_path),
        },
    }
    write_json(metrics_dir / "m10_2_ready_report.json", ready_report_json)
    (metrics_dir / "m10_2_ready_report.md").write_text(
        "\n".join(
            [
                "# M10.2 Ready Report",
                f"- previous_failed_criteria: `{', '.join(previous_failed_criteria)}`",
                *[f"- detector_change: {item}" for item in detector_logic_changes],
                *[f"- generation_change: {item}" for item in scenario_generation_changes],
                f"- new_targeted_scenarios: `{int(len(targeted_records))}`",
                f"- test_f1_abnormal: `{float(test_metrics['f1_abnormal']):.4f}`",
                f"- test_recall_abnormal: `{float(test_metrics['recall_abnormal']):.4f}`",
                f"- test_false_positives_per_minute: `{float(test_metrics['false_positives_per_minute']):.4f}`",
                f"- test_detection_delay_s: `{test_metrics.get('detection_delay_s')}`",
                f"- calibration_ece_test: `{float(test_cal.ece):.4f}`",
                f"- cyber_heavy_recall: `{float(familywise.get('cyber_heavy', {}).get('recall_abnormal', 0.0) or 0.0):.4f}`",
                f"- concurrent_heavy_recall: `{float(familywise.get('concurrent_heavy', {}).get('recall_abnormal', 0.0) or 0.0):.4f}`",
                f"- binary_detector_ready_for_classifier_localizer: `{ready}`",
                f"- verdict: `{verdict}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    write_json(
        metrics_dir / "hardened_test_eval_report_v4.json",
        {
            "run_name": "hardened_test",
            "split_csv": str(test_split_csv),
            "windows": int(test_input.x_windows.shape[0]),
            "metrics": {**test_metrics, "calibration_ece": float(test_cal.ece), "brier_score": float(test_cal.brier)},
            "familywise": familywise,
        },
    )

    root_copy_map: list[tuple[Path, Path]] = [
        (metrics_dir / "detector_training_report_v4.json", output_root / "detector_training_report_v4.json"),
        (metrics_dir / "detector_training_report_v4.md", output_root / "detector_training_report_v4.md"),
        (metrics_dir / "per_scenario_metrics_v4.csv", output_root / "per_scenario_metrics_v4.csv"),
        (metrics_dir / "threshold_sweep_v4.csv", output_root / "threshold_sweep_v4.csv"),
        (metrics_dir / "hardened_test_frame_predictions_v4.csv", output_root / "hardened_test_frame_predictions_v4.csv"),
        (metrics_dir / "family_coverage_report_v2.csv", output_root / "family_coverage_report_v2.csv"),
        (metrics_dir / "normal_fp_forensics_v2.csv", output_root / "normal_fp_forensics_v2.csv"),
        (metrics_dir / "scenario_generation_patch_report.json", output_root / "scenario_generation_patch_report.json"),
        (metrics_dir / "scenario_generation_patch_report.md", output_root / "scenario_generation_patch_report.md"),
        (metrics_dir / "m10_2_ready_report.json", output_root / "m10_2_ready_report.json"),
        (metrics_dir / "m10_2_ready_report.md", output_root / "m10_2_ready_report.md"),
        (metadata_dir / "split_manifest_v4.json", output_root / "split_manifest_v4.json"),
        (train_split_csv, output_root / "train_scenarios_v4.csv"),
        (val_split_csv, output_root / "val_scenarios_v4.csv"),
        (test_split_csv, output_root / "test_scenarios_v4.csv"),
        (metadata_dir / "new_targeted_scenarios_v4.csv", output_root / "new_targeted_scenarios_v4.csv"),
    ]
    for src, dst in root_copy_map:
        if not src.exists():
            continue
        if src.suffix.lower() == ".csv":
            try:
                pd.read_csv(src).to_csv(dst, index=False)
            except pd.errors.EmptyDataError:
                # Preserve explicit empty-CSV artifacts without failing the full training run.
                pd.DataFrame().to_csv(dst, index=False)
        else:
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    write_json(
        metrics_dir / "training_summary.json",
        {
            "ready": ready,
            "verdict": verdict,
            "test_f1_abnormal": float(test_metrics["f1_abnormal"]),
            "test_recall_abnormal": float(test_metrics["recall_abnormal"]),
            "test_false_positives_per_minute": float(test_metrics["false_positives_per_minute"]),
            "test_detection_delay_s": test_metrics.get("detection_delay_s"),
            "calibration_ece_test": float(test_cal.ece),
            "selected_calibration": cal_selection.selected_name,
        },
    )
    write_json(output_root / "training_summary.json", json.loads((metrics_dir / "training_summary.json").read_text(encoding="utf-8")))

    # backward compatibility from phase-2/3
    detector.save(output_root / "detector_model.pkl")
    test_frame.to_csv(output_root / "test_frame_predictions.csv", index=False)
    test_frame.to_csv(output_root / "hardened_test_frame_predictions_v4.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
