"""Train and infer with the staged physics-aware PMU pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.ml.pmu_features import PMU_BUSES, extract_window_features, feature_frame, frame_time_bounds, read_pmu_frames_from_dir
from src.ml.pmu_grid_pipeline import (
    ALL_BUSES,
    DatasetBundle,
    Metrics,
    PmuDatasetBuilder,
    ScenarioSplitter,
    SegmentBuilder,
)
from src.ml.pmu_plotting import PredictionPlotter, TrainingPlotter
from src.ml.staged.localization import PhysicsAwareLocalizer
from src.ml.staged.model import StagedModelConfig, StagedPmuGridModel
from src.ml.staged.postprocessing import EventPostProcessor, PostProcessConfig


@dataclass(frozen=True)
class StagedTrainingConfig:
    synthetic_dir: Path
    output_dir: Path
    model: StagedModelConfig = StagedModelConfig()
    window_sec: float = 1.0
    samples_per_event: int = 3
    normal_samples_per_scenario: int = 8
    max_scenarios: int | None = None
    train_fraction: float = 0.70
    runs: int = 1
    seed: int = 20260412


@dataclass(frozen=True)
class StagedInferenceConfig:
    model_path: Path
    input_dir: Path
    output_path: Path
    window_sec: float | None = None
    stride_sec: float = 0.25
    threshold: float = 0.45
    max_windows: int | None = None
    postprocess: PostProcessConfig = PostProcessConfig()


class StagedPmuTrainingPipeline:
    """Train/test detector, classifier, location, and state heads separately."""

    def __init__(self, config: StagedTrainingConfig) -> None:
        self.config = config
        self.metrics = Metrics()

    def run(self) -> dict[str, Any]:
        start = time.perf_counter()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        bundle = PmuDatasetBuilder(
            self.config.synthetic_dir,
            self.config.window_sec,
            self.config.samples_per_event,
            self.config.normal_samples_per_scenario,
            self.config.max_scenarios,
        ).build()
        reports = [self._run_one(bundle, run_index) for run_index in range(1, self.config.runs + 1)]
        best = max(reports, key=lambda report: report["selection_score"])
        best_model_path = self.config.output_dir / "pmu_grid_model.pkl"
        joblib.dump(joblib.load(best["model_path"]), best_model_path)
        history_path = self.config.output_dir / "pmu_grid_training_history.csv"
        history = pd.DataFrame(reports)
        history.to_csv(history_path, index=False)
        plotter = TrainingPlotter(self.config.output_dir)
        figure_paths = []
        figure_paths.extend(plotter.plot_history(history))
        figure_paths.extend(plotter.plot_label_counts(bundle.targets))
        summary = {
            "pipeline_kind": "staged_physics",
            "best_run": int(best["run"]),
            "best_model_path": str(best_model_path.resolve()),
            "history_path": str(history_path.resolve()),
            "elapsed_sec": round(time.perf_counter() - start, 3),
            "n_runs": int(self.config.runs),
            "n_samples": int(len(bundle.targets)),
            "n_features_raw": int(bundle.features.shape[1]),
            "n_scenarios": int(bundle.targets["scenario_id"].nunique()),
            "train_fraction": float(self.config.train_fraction),
            "test_fraction": round(1.0 - float(self.config.train_fraction), 6),
            "input_contract": self._input_contract(),
            "figures": figure_paths,
            "best_metrics": best,
        }
        summary_path = self.config.output_dir / "pmu_grid_training_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return summary

    def _run_one(self, bundle: DatasetBundle, run_index: int) -> dict[str, Any]:
        seed = self.config.seed + (run_index - 1) * 1009
        split = ScenarioSplitter(self.config.train_fraction, seed).split(bundle.scenario_ids)
        model = StagedPmuGridModel(self.config.model)
        model.fit(bundle.features, bundle.targets, bundle.bus_states, split.train_idx, seed)
        predictions = model.predict(bundle.features.iloc[split.test_idx])
        report = self._report(bundle, split, model, predictions, run_index, seed)

        run_dir = self.config.output_dir / f"run_{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        model_path = run_dir / "pmu_grid_model.pkl"
        report_path = run_dir / "metrics.json"
        artifact = model.artifact(self._input_contract(), self.config.window_sec)
        artifact["train_sims"] = split.train_sims
        artifact["test_sims"] = split.test_sims
        joblib.dump(artifact, model_path)
        report["model_path"] = str(model_path.resolve())
        report["report_path"] = str(report_path.resolve())
        report["serialized_size_bytes"] = int(model_path.stat().st_size)
        report["figures"] = TrainingPlotter(self.config.output_dir).plot_train_test_sims(
            split.train_sims,
            split.test_sims,
            run_index,
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return self._flat_report(report)

    def _report(
        self,
        bundle: DatasetBundle,
        split: Any,
        model: StagedPmuGridModel,
        predictions: dict[str, Any],
        run_index: int,
        seed: int,
    ) -> dict[str, Any]:
        test_targets = bundle.targets.iloc[split.test_idx]
        true_event = test_targets["event_label"].to_numpy(dtype=int)
        pred_event = np.asarray(predictions["event_labels"], dtype=int)
        event_metrics = self._event_metrics(true_event, pred_event)
        detector_metrics = self._detector_metrics(true_event != 0, pred_event != 0)
        location_metrics = self._location_metrics(test_targets, predictions)
        bus_state_metrics = self.metrics.bus_state(
            bundle.bus_states.iloc[split.test_idx].to_numpy(dtype=int),
            predictions["bus_states"],
        )
        selection_score = (
            0.35 * event_metrics["macro_f1"]
            + 0.25 * detector_metrics["macro_f1"]
            + 0.25 * bus_state_metrics["per_bus_macro_f1"]
            + 0.15 * bus_state_metrics["active_bus_jaccard_mean"]
        )
        return {
            "run": int(run_index),
            "seed": int(seed),
            "model_name": model.resolved_name,
            "train_sims": split.train_sims,
            "test_sims": split.test_sims,
            "n_train_samples": int(len(split.train_idx)),
            "n_test_samples": int(len(split.test_idx)),
            "event_label_counts_all": self._label_counts(bundle.targets["event_label"]),
            "event_label_counts_train": self._label_counts(bundle.targets.iloc[split.train_idx]["event_label"]),
            "event_label_counts_test": self._label_counts(bundle.targets.iloc[split.test_idx]["event_label"]),
            "detector_metrics": detector_metrics,
            "event_metrics": event_metrics,
            "location_metrics": location_metrics,
            "bus_state_metrics": bus_state_metrics,
            "selection_score": float(selection_score),
            "warnings": self._warnings(bundle),
        }

    def _event_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        labels = list(range(0, 9))
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
            "positive_macro_f1": float(f1_score(y_true, y_pred, labels=list(range(1, 9)), average="macro", zero_division=0)),
            "report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        }

    def _detector_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    def _location_metrics(self, targets: pd.DataFrame, predictions: dict[str, Any]) -> dict[str, Any]:
        true_positive = targets["event_label"].to_numpy(dtype=int) != 0
        pred_positive = np.asarray(predictions["event_labels"], dtype=int) != 0
        mask = true_positive & pred_positive
        if not np.any(mask):
            return {"note": "No windows with both true and predicted positive event labels."}
        true_loc = targets.loc[mask, "location"].astype(str).to_numpy()
        pred_loc = np.asarray(predictions["locations"], dtype=object)[mask].astype(str)
        labels = sorted(set(true_loc) | set(pred_loc))
        return {
            "accuracy": float(accuracy_score(true_loc, pred_loc)),
            "macro_f1": float(f1_score(true_loc, pred_loc, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(true_loc, pred_loc, labels=labels, average="weighted", zero_division=0)),
        }

    def _flat_report(self, report: dict[str, Any]) -> dict[str, Any]:
        return {
            "run": report["run"],
            "seed": report["seed"],
            "model_name": report["model_name"],
            "n_train_samples": report["n_train_samples"],
            "n_test_samples": report["n_test_samples"],
            "detector_macro_f1": report["detector_metrics"]["macro_f1"],
            "event_macro_f1": report["event_metrics"]["macro_f1"],
            "event_positive_macro_f1": report["event_metrics"]["positive_macro_f1"],
            "event_weighted_f1": report["event_metrics"]["weighted_f1"],
            "bus_state_macro_f1": report["bus_state_metrics"]["per_bus_macro_f1"],
            "bus_state_weighted_f1": report["bus_state_metrics"]["per_bus_weighted_f1"],
            "full_state_exact_match": report["bus_state_metrics"]["full_state_exact_match"],
            "active_bus_jaccard_mean": report["bus_state_metrics"]["active_bus_jaccard_mean"],
            "selection_score": report["selection_score"],
            "model_path": report.get("model_path"),
            "report_path": report.get("report_path"),
            "serialized_size_bytes": report.get("serialized_size_bytes"),
        }

    def _label_counts(self, values: pd.Series) -> dict[str, int]:
        return {str(int(label)): int(count) for label, count in values.value_counts().sort_index().items()}

    def _warnings(self, bundle: DatasetBundle) -> list[str]:
        warnings: list[str] = []
        if int(bundle.targets["scenario_id"].nunique()) < 500:
            warnings.append("Smoke-scale dataset: use 5,000 scenarios before trusting raw-file predictions.")
        missing = sorted(set(range(1, 9)) - set(int(label) for label in bundle.targets["event_label"].unique()))
        if missing:
            warnings.append("Missing event labels in training data: " + ", ".join(str(label) for label in missing))
        return warnings

    def _input_contract(self) -> dict[str, Any]:
        return {
            "feature_buses": PMU_BUSES,
            "uses_non_pmu_csv_as_features": False,
            "hidden_non_pmu_csv_role": "39_bus_state_truth_and_review_only",
            "target_buses": ALL_BUSES,
            "pipeline": "detector -> event classifier -> location -> physics localizer -> postprocess",
        }


class StagedPmuInferencePipeline:
    """Apply the staged model and attach physics localization candidates."""

    def __init__(self, config: StagedInferenceConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        artifact = joblib.load(self.config.model_path)
        model = StagedPmuGridModel.from_artifact(artifact)
        frames = read_pmu_frames_from_dir(self.config.input_dir, artifact["pmu_buses"])
        window_sec = float(self.config.window_sec if self.config.window_sec is not None else artifact["window_sec"])
        centers = self._centers(frames, window_sec)
        features = feature_frame([extract_window_features(frames, float(center), window_sec) for center in centers])
        predictions = model.predict(features, threshold=self.config.threshold, return_features=True)
        segments = SegmentBuilder().build(
            centers,
            predictions["event_labels"],
            predictions["locations"],
            predictions["event_confidence"],
            predictions["bus_states"],
            self.config.stride_sec,
        )
        segments = EventPostProcessor(self.config.postprocess).process(segments)
        self._attach_localization(segments, centers, predictions)
        output = {
            "input_dir": str(self.config.input_dir.resolve()),
            "model": str(self.config.model_path.resolve()),
            "pipeline_kind": "staged_physics",
            "input_contract": artifact["input_contract"],
            "window_sec": window_sec,
            "stride_sec": float(self.config.stride_sec),
            "threshold": float(self.config.threshold),
            "pmu_buses": artifact["pmu_buses"],
            "target_buses": artifact["all_buses"],
            "n_windows": int(len(centers)),
            "events": segments,
        }
        output["figures"] = PredictionPlotter(self.config.output_path).plot(output)
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(output, indent=2))
        return output

    def _attach_localization(self, segments: list[dict[str, Any]], centers: np.ndarray, predictions: dict[str, Any]) -> None:
        localizer = PhysicsAwareLocalizer(top_k=5)
        features = predictions["augmented_features"]
        bus_states = predictions["bus_states"]
        for segment in segments:
            candidates = localizer.rank_segment(segment, centers, features, bus_states)
            segment["localization_candidates"] = candidates
            if candidates:
                segment["physics_location"] = f"bus_{candidates[0]['bus']}"

    def _centers(self, frames: dict[int, pd.DataFrame], window_sec: float) -> np.ndarray:
        start, end = frame_time_bounds(frames)
        first = start + window_sec / 2.0
        last = end - window_sec / 2.0
        if last < first:
            centers = np.array([(start + end) / 2.0], dtype=float)
        else:
            centers = np.arange(first, last + 1e-9, self.config.stride_sec, dtype=float)
        if self.config.max_windows is not None:
            centers = centers[: int(self.config.max_windows)]
        return centers
