"""Object-oriented V2 ML pipeline for 8-PMU to full-grid event inference."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.ml.pmu_features import (
    PMU_BUSES,
    collect_scenario_jsons,
    event_primary_location,
    event_sample_centers,
    extract_window_features,
    feature_frame,
    frame_time_bounds,
    normal_sample_centers,
    read_pmu_frames_from_dir,
    read_scenario_pmu_frames,
)
from src.ml.pmu_plotting import PredictionPlotter, TrainingPlotter

warnings.filterwarnings("ignore", message="X does not have valid feature names")

try:  # pragma: no cover - optional local dependency.
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - fallback path.
    LGBMClassifier = None
    HAS_LIGHTGBM = False


ALL_BUSES = list(range(1, 40))
BUS_STATE_COLUMNS = [f"bus_{bus:02d}_state" for bus in ALL_BUSES]
LABEL_PRIORITY = {0: 0, 4: 20, 3: 30, 2: 40, 1: 50, 8: 55, 5: 60, 6: 70, 7: 80}


@dataclass(frozen=True)
class TrainingConfig:
    synthetic_dir: Path
    output_dir: Path
    model_name: str = "lightgbm"
    window_sec: float = 1.0
    samples_per_event: int = 3
    normal_samples_per_scenario: int = 8
    max_scenarios: int | None = None
    train_fraction: float = 0.70
    runs: int = 1
    seed: int = 20260412


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path
    input_dir: Path
    output_path: Path
    window_sec: float | None = None
    stride_sec: float = 0.25
    threshold: float = 0.35
    max_windows: int | None = None


@dataclass(frozen=True)
class DatasetBundle:
    features: pd.DataFrame
    targets: pd.DataFrame
    bus_states: pd.DataFrame

    @property
    def scenario_ids(self) -> np.ndarray:
        return self.targets["scenario_id"].astype(str).to_numpy()


@dataclass(frozen=True)
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_sims: list[str]
    test_sims: list[str]


class ModelFactory:
    """Factory for compact sklearn-compatible models."""

    def __init__(self, model_name: str, seed: int) -> None:
        self.model_name = model_name
        self.seed = int(seed)

    @property
    def resolved_name(self) -> str:
        if self.model_name == "lightgbm" and not HAS_LIGHTGBM:
            return "histgb"
        return self.model_name

    def make_imputer(self) -> SimpleImputer:
        try:
            return SimpleImputer(strategy="median", keep_empty_features=True)
        except TypeError:
            return SimpleImputer(strategy="median")

    def make_classifier(self, n_classes: int, seed_offset: int = 0) -> Any:
        if n_classes < 2:
            return DummyClassifier(strategy="most_frequent")
        seed = self.seed + seed_offset
        if self.resolved_name == "lightgbm" and HAS_LIGHTGBM:
            objective = "binary" if n_classes == 2 else "multiclass"
            return LGBMClassifier(
                objective=objective,
                n_estimators=150,
                learning_rate=0.055,
                num_leaves=15,
                max_depth=5,
                min_child_samples=2,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
        if self.resolved_name == "histgb":
            return HistGradientBoostingClassifier(
                max_iter=160,
                learning_rate=0.06,
                max_leaf_nodes=15,
                l2_regularization=0.1,
                random_state=seed,
            )
        return ExtraTreesClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )

    def make_pipeline(self, n_classes: int, seed_offset: int = 0) -> Pipeline:
        return Pipeline([("imputer", self.make_imputer()), ("classifier", self.make_classifier(n_classes, seed_offset))])

    def make_bus_state_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("imputer", self.make_imputer()),
                (
                    "classifier",
                    ExtraTreesClassifier(
                        n_estimators=140,
                        max_depth=14,
                        min_samples_leaf=1,
                        max_features="sqrt",
                        random_state=self.seed + 101,
                        n_jobs=-1,
                    ),
                ),
            ]
        )


class ScenarioSplitter:
    """70/30 split by SIM id, never by individual PMU windows."""

    def __init__(self, train_fraction: float, seed: int) -> None:
        if not 0.0 < float(train_fraction) < 1.0:
            raise ValueError("train_fraction must be between 0 and 1.")
        self.train_fraction = float(train_fraction)
        self.seed = int(seed)

    def split(self, scenario_ids: Iterable[str]) -> Split:
        scenario_array = np.array(list(scenario_ids), dtype=object)
        unique = np.array(sorted(set(str(item) for item in scenario_array)), dtype=object)
        if unique.size < 2:
            index = np.arange(len(scenario_array))
            sims = unique.tolist()
            return Split(index, index, sims, sims)
        rng = np.random.default_rng(self.seed)
        shuffled = unique.copy()
        rng.shuffle(shuffled)
        n_train = int(round(len(shuffled) * self.train_fraction))
        n_train = min(max(n_train, 1), len(shuffled) - 1)
        train_sims = sorted(str(item) for item in shuffled[:n_train])
        test_sims = sorted(str(item) for item in shuffled[n_train:])
        train_idx = np.flatnonzero(np.isin(scenario_array, train_sims))
        test_idx = np.flatnonzero(np.isin(scenario_array, test_sims))
        return Split(train_idx, test_idx, train_sims, test_sims)


class FullStateLabeler:
    """Build 39-bus state targets from scenario metadata."""

    def __init__(self, buses: Iterable[int] = ALL_BUSES) -> None:
        self.buses = [int(bus) for bus in buses]

    def label_window(self, events: list[dict[str, Any]], center_sec: float) -> tuple[int, str, np.ndarray, str]:
        active = [event for event in events if self._is_active(event, center_sec)]
        states = {bus: 0 for bus in self.buses}
        strongest_event = self._strongest_event(active)
        for event in active:
            label = int(event.get("label", 0))
            for bus in self._target_buses(event):
                if bus not in states:
                    continue
                current = states[bus]
                if LABEL_PRIORITY[label] >= LABEL_PRIORITY[current]:
                    states[bus] = label
        event_label = self._event_label_from_states(states, strongest_event)
        location = event_primary_location(strongest_event) if strongest_event else "normal"
        active_kinds = ",".join(str(event.get("kind", "unknown")) for event in active) if active else "normal"
        return event_label, location, np.array([states[bus] for bus in self.buses], dtype=int), active_kinds

    def _is_active(self, event: dict[str, Any], center_sec: float) -> bool:
        return float(event["start_sec"]) <= float(center_sec) <= float(event["end_sec"])

    def _strongest_event(self, events: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not events:
            return None
        return max(events, key=lambda event: LABEL_PRIORITY[int(event.get("label", 0))])

    def _event_label_from_states(self, states: dict[int, int], strongest_event: dict[str, Any] | None) -> int:
        labels = set(states.values()) - {0}
        if labels:
            return max(labels, key=lambda label: LABEL_PRIORITY[int(label)])
        return int(strongest_event.get("label", 0)) if strongest_event else 0

    def _target_buses(self, event: dict[str, Any]) -> list[int]:
        label = int(event.get("label", 0))
        if label in {5, 6, 7} and event.get("pmu_bus") is not None:
            return [int(event["pmu_bus"])]
        affected = event.get("affected_buses") or []
        if affected:
            return [int(bus) for bus in affected]
        line = event.get("line") or []
        if line:
            return [int(bus) for bus in line]
        nodes = event.get("nodes") or []
        if nodes:
            return [int(bus) for bus in nodes]
        if event.get("pmu_bus") is not None:
            return [int(event["pmu_bus"])]
        return []


class PmuDatasetBuilder:
    """Build feature rows from 8 PMU CSVs and targets from synthetic truth."""

    def __init__(
        self,
        synthetic_dir: Path,
        window_sec: float,
        samples_per_event: int,
        normal_samples_per_scenario: int,
        max_scenarios: int | None = None,
    ) -> None:
        self.synthetic_dir = Path(synthetic_dir)
        self.window_sec = float(window_sec)
        self.samples_per_event = int(samples_per_event)
        self.normal_samples_per_scenario = int(normal_samples_per_scenario)
        self.max_scenarios = max_scenarios
        self.labeler = FullStateLabeler()

    def build(self) -> DatasetBundle:
        paths = collect_scenario_jsons(self.synthetic_dir, self.max_scenarios)
        if not paths:
            raise FileNotFoundError(f"No scenario JSON files found under {self.synthetic_dir}")
        feature_rows: list[dict[str, float]] = []
        target_rows: list[dict[str, Any]] = []
        state_rows: list[np.ndarray] = []
        for scenario_json in paths:
            scenario, frames = read_scenario_pmu_frames(scenario_json, PMU_BUSES)
            self._append_scenario(scenario, frames, feature_rows, target_rows, state_rows)
        return DatasetBundle(
            features=feature_frame(feature_rows),
            targets=pd.DataFrame(target_rows),
            bus_states=pd.DataFrame(np.vstack(state_rows), columns=BUS_STATE_COLUMNS),
        )

    def _append_scenario(
        self,
        scenario: dict[str, Any],
        frames: dict[int, pd.DataFrame],
        feature_rows: list[dict[str, float]],
        target_rows: list[dict[str, Any]],
        state_rows: list[np.ndarray],
    ) -> None:
        events = list(scenario.get("events", []))
        for center_sec in self._sample_centers(float(scenario.get("duration_sec", 0.0)), events):
            event_label, location, bus_state, active_kinds = self.labeler.label_window(events, center_sec)
            feature_rows.append(extract_window_features(frames, center_sec, self.window_sec))
            target_rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "center_sec": float(center_sec),
                    "sample_kind": "event" if event_label else "normal",
                    "event_label": int(event_label),
                    "location": location,
                    "active_event_kinds": active_kinds,
                }
            )
            state_rows.append(bus_state)

    def _sample_centers(self, duration_sec: float, events: list[dict[str, Any]]) -> list[float]:
        centers: list[float] = []
        for event in events:
            centers.extend(event_sample_centers(event, self.samples_per_event))
        centers.extend(
            normal_sample_centers(
                duration_sec,
                events,
                self.normal_samples_per_scenario,
                margin_sec=max(0.25, self.window_sec / 2.0),
            )
        )
        return sorted({round(float(center), 4) for center in centers})


class LabelCodec:
    """JSON-friendly label encoder wrapper."""

    def __init__(self) -> None:
        self.encoder = LabelEncoder()
        self.classes: list[Any] = []

    def fit_transform(self, values: pd.Series) -> np.ndarray:
        encoded = self.encoder.fit_transform(values.astype(str))
        self.classes = [int(label) if str(label).isdigit() else str(label) for label in self.encoder.classes_]
        return encoded

    def transform(self, values: pd.Series) -> np.ndarray:
        return self.encoder.transform(values.astype(str))

    def decode(self, encoded: np.ndarray) -> list[Any]:
        decoded = self.encoder.inverse_transform(np.asarray(encoded, dtype=int))
        return [int(item) if str(item).isdigit() else str(item) for item in decoded]


class PmuGridModel:
    """Three-head model: event type, event location, and 39-bus state."""

    def __init__(self, factory: ModelFactory, feature_names: list[str] | None = None) -> None:
        self.factory = factory
        self.feature_names = feature_names or []
        self.event_codec = LabelCodec()
        self.location_codec = LabelCodec()
        self.event_model: Pipeline | None = None
        self.location_model: Pipeline | None = None
        self.bus_state_model: Pipeline | None = None

    def fit(self, bundle: DatasetBundle, train_idx: np.ndarray) -> None:
        self.feature_names = list(bundle.features.columns)
        event_y = self.event_codec.fit_transform(bundle.targets["event_label"])
        positive_mask = bundle.targets["event_label"].to_numpy(dtype=int) != 0
        loc_values = bundle.targets.loc[positive_mask, "location"] if np.any(positive_mask) else pd.Series(["normal"])
        self.location_codec.fit_transform(loc_values)

        train_features = bundle.features.iloc[train_idx]
        self.event_model = self.factory.make_pipeline(len(self.event_codec.classes), seed_offset=0)
        self.event_model.fit(train_features, event_y[train_idx])

        train_positive = positive_mask[train_idx]
        self.location_model = self.factory.make_pipeline(len(self.location_codec.classes), seed_offset=17)
        if np.any(train_positive):
            location_y = np.full(len(bundle.targets), -1, dtype=int)
            location_y[positive_mask] = self.location_codec.transform(bundle.targets.loc[positive_mask, "location"])
            self.location_model.fit(train_features[train_positive], location_y[train_idx][train_positive])
        else:
            self.location_model.fit(train_features, np.zeros(len(train_features), dtype=int))

        self.bus_state_model = self.factory.make_bus_state_pipeline()
        self.bus_state_model.fit(train_features, bundle.bus_states.iloc[train_idx].to_numpy(dtype=int))

    def predict(self, features: pd.DataFrame, threshold: float | None = None) -> dict[str, Any]:
        self._require_fitted()
        aligned = features.reindex(columns=self.feature_names, fill_value=np.nan)
        event_encoded = self.event_model.predict(aligned)
        event_labels = np.array(self.event_codec.decode(event_encoded), dtype=int)
        confidence = self._max_probability(self.event_model, aligned)
        if threshold is not None:
            event_labels = np.where(confidence >= float(threshold), event_labels, 0)
        positive = event_labels != 0
        locations = np.array(["normal"] * len(aligned), dtype=object)
        if np.any(positive):
            location_encoded = self.location_model.predict(aligned.loc[positive])
            locations[positive] = np.array(self.location_codec.decode(location_encoded), dtype=object)
        bus_states = self.bus_state_model.predict(aligned).astype(int)
        if threshold is not None:
            bus_states[event_labels == 0, :] = 0
        return {
            "event_labels": event_labels,
            "event_confidence": confidence,
            "locations": locations,
            "bus_states": bus_states,
        }

    def artifact(self, input_contract: dict[str, Any], window_sec: float) -> dict[str, Any]:
        self._require_fitted()
        return {
            "event_model": self.event_model,
            "location_model": self.location_model,
            "bus_state_model": self.bus_state_model,
            "event_encoder": self.event_codec.encoder,
            "location_encoder": self.location_codec.encoder,
            "event_classes": self.event_codec.classes,
            "location_classes": self.location_codec.classes,
            "feature_names": self.feature_names,
            "pmu_buses": PMU_BUSES,
            "all_buses": ALL_BUSES,
            "bus_state_columns": BUS_STATE_COLUMNS,
            "window_sec": float(window_sec),
            "model_name": self.factory.resolved_name,
            "input_contract": input_contract,
        }

    def _require_fitted(self) -> None:
        if self.event_model is None or self.location_model is None or self.bus_state_model is None:
            raise RuntimeError("PmuGridModel is not fitted.")

    def _max_probability(self, model: Pipeline, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            return np.ones(len(features), dtype=float)
        proba = model.predict_proba(features)
        if isinstance(proba, list):
            proba = proba[0]
        return np.max(np.asarray(proba, dtype=float), axis=1)


class Metrics:
    """Metric helpers for event, location, and 39-bus state predictions."""

    def classification(self, y_true: np.ndarray, y_pred: np.ndarray, classes: list[Any]) -> dict[str, Any]:
        labels = list(range(len(classes)))
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "report": classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=[str(item) for item in classes],
                output_dict=True,
                zero_division=0,
            ),
        }

    def bus_state(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        flat_true = y_true.reshape(-1)
        flat_pred = y_pred.reshape(-1)
        active_true = y_true != 0
        active_pred = y_pred != 0
        intersection = np.logical_and(active_true, active_pred).sum(axis=1)
        union = np.logical_or(active_true, active_pred).sum(axis=1)
        jaccard = np.ones_like(union, dtype=float)
        np.divide(intersection, union, out=jaccard, where=union > 0)
        return {
            "full_state_exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
            "per_bus_accuracy": float(accuracy_score(flat_true, flat_pred)),
            "per_bus_macro_f1": float(f1_score(flat_true, flat_pred, average="macro", zero_division=0)),
            "per_bus_weighted_f1": float(f1_score(flat_true, flat_pred, average="weighted", zero_division=0)),
            "active_bus_jaccard_mean": float(np.mean(jaccard)),
        }

    def top_feature_importance(self, model: Pipeline | None, feature_names: list[str], limit: int = 25) -> list[dict[str, Any]]:
        if model is None:
            return []
        classifier = model.named_steps["classifier"]
        values = getattr(classifier, "feature_importances_", None)
        if values is None:
            return []
        pairs = sorted(zip(feature_names, values), key=lambda item: float(item[1]), reverse=True)
        return [
            {"feature": name, "importance": float(value)}
            for name, value in pairs[:limit]
            if float(value) > 0
        ]


class PmuGridTrainingPipeline:
    """End-to-end train/test workflow with repeated 70/30 SIM splits."""

    def __init__(self, config: TrainingConfig) -> None:
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
            "best_run": int(best["run"]),
            "best_model_path": str(best_model_path.resolve()),
            "history_path": str(history_path.resolve()),
            "elapsed_sec": round(time.perf_counter() - start, 3),
            "n_runs": int(self.config.runs),
            "n_samples": int(len(bundle.targets)),
            "n_features": int(bundle.features.shape[1]),
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
        model = PmuGridModel(ModelFactory(self.config.model_name, seed))
        model.fit(bundle, split.train_idx)
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
        split: Split,
        model: PmuGridModel,
        predictions: dict[str, Any],
        run_index: int,
        seed: int,
    ) -> dict[str, Any]:
        test_targets = bundle.targets.iloc[split.test_idx]
        true_event = model.event_codec.transform(test_targets["event_label"])
        pred_event = model.event_codec.encoder.transform(predictions["event_labels"].astype(str))
        event_metrics = self.metrics.classification(true_event, pred_event, model.event_codec.classes)

        positive = test_targets["event_label"].to_numpy(dtype=int) != 0
        predicted_positive = np.asarray(predictions["event_labels"], dtype=int) != 0
        location_eval = positive & predicted_positive
        if np.any(location_eval):
            true_loc = model.location_codec.transform(test_targets.loc[location_eval, "location"])
            pred_loc = model.location_codec.encoder.transform(predictions["locations"][location_eval].astype(str))
            location_metrics = self.metrics.classification(true_loc, pred_loc, model.location_codec.classes)
        else:
            location_metrics = {"note": "No windows with both true and predicted positive event labels."}

        true_bus = bundle.bus_states.iloc[split.test_idx].to_numpy(dtype=int)
        bus_state_metrics = self.metrics.bus_state(true_bus, predictions["bus_states"])
        selection_score = (
            0.45 * event_metrics["macro_f1"]
            + 0.35 * bus_state_metrics["per_bus_macro_f1"]
            + 0.20 * bus_state_metrics["active_bus_jaccard_mean"]
        )
        return {
            "run": int(run_index),
            "seed": int(seed),
            "model_name": model.factory.resolved_name,
            "train_sims": split.train_sims,
            "test_sims": split.test_sims,
            "n_train_samples": int(len(split.train_idx)),
            "n_test_samples": int(len(split.test_idx)),
            "event_label_counts_all": self._label_counts(bundle.targets["event_label"]),
            "event_label_counts_train": self._label_counts(bundle.targets.iloc[split.train_idx]["event_label"]),
            "event_label_counts_test": self._label_counts(bundle.targets.iloc[split.test_idx]["event_label"]),
            "event_metrics": event_metrics,
            "location_metrics": location_metrics,
            "bus_state_metrics": bus_state_metrics,
            "event_top_features": self.metrics.top_feature_importance(model.event_model, model.feature_names),
            "location_top_features": self.metrics.top_feature_importance(model.location_model, model.feature_names),
            "bus_state_top_features": self.metrics.top_feature_importance(model.bus_state_model, model.feature_names),
            "selection_score": float(selection_score),
            "warnings": self._warnings(bundle),
        }

    def _flat_report(self, report: dict[str, Any]) -> dict[str, Any]:
        return {
            "run": report["run"],
            "seed": report["seed"],
            "model_name": report["model_name"],
            "n_train_samples": report["n_train_samples"],
            "n_test_samples": report["n_test_samples"],
            "event_macro_f1": report["event_metrics"]["macro_f1"],
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
        warnings_list: list[str] = []
        scenario_count = int(bundle.targets["scenario_id"].nunique())
        if scenario_count < 500:
            warnings_list.append("Smoke-scale dataset: use 5,000 scenarios before trusting raw-file predictions.")
        missing = sorted(set(range(1, 9)) - set(int(label) for label in bundle.targets["event_label"].unique()))
        if missing:
            warnings_list.append("Missing event labels in training data: " + ", ".join(str(label) for label in missing))
        return warnings_list

    def _input_contract(self) -> dict[str, Any]:
        return {
            "feature_buses": PMU_BUSES,
            "uses_non_pmu_csv_as_features": False,
            "hidden_non_pmu_csv_role": "39_bus_state_truth_and_review_only",
            "target_buses": ALL_BUSES,
        }


class SegmentBuilder:
    """Convert per-window predictions into event intervals."""

    def build(
        self,
        centers: np.ndarray,
        labels: np.ndarray,
        locations: np.ndarray,
        confidences: np.ndarray,
        bus_states: np.ndarray,
        stride_sec: float,
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        def close() -> None:
            nonlocal current
            if current is None:
                return
            conf = np.asarray(current.pop("_confidences"), dtype=float)
            state_matrix = np.vstack(current.pop("_bus_states"))
            current["confidence_mean"] = float(np.mean(conf)) if conf.size else 0.0
            current["confidence_max"] = float(np.max(conf)) if conf.size else 0.0
            current["bus_states"] = self._majority_bus_state(state_matrix)
            current["active_buses"] = [
                {"bus": int(bus), "state": int(state)}
                for bus, state in current["bus_states"].items()
                if int(state) != 0
            ]
            segments.append(current)
            current = None

        for center, label, location, confidence, state in zip(centers, labels, locations, confidences, bus_states):
            label = int(label)
            if label == 0:
                close()
                continue
            needs_new = (
                current is None
                or current["label"] != label
                or current["location"] != str(location)
                or float(center) - float(current["_last_center"]) > float(stride_sec) * 1.75
            )
            if needs_new:
                close()
                current = {
                    "label": label,
                    "location": str(location),
                    "start_sec": float(max(0.0, center - stride_sec / 2.0)),
                    "end_sec": float(center + stride_sec / 2.0),
                    "window_count": 1,
                    "_last_center": float(center),
                    "_confidences": [float(confidence)],
                    "_bus_states": [np.asarray(state, dtype=int)],
                }
            else:
                current["end_sec"] = float(center + stride_sec / 2.0)
                current["window_count"] = int(current["window_count"]) + 1
                current["_last_center"] = float(center)
                current["_confidences"].append(float(confidence))
                current["_bus_states"].append(np.asarray(state, dtype=int))
        close()
        for segment in segments:
            segment.pop("_last_center", None)
        return segments

    def _majority_bus_state(self, states: np.ndarray) -> dict[str, int]:
        result: dict[str, int] = {}
        for index, bus in enumerate(ALL_BUSES):
            values, counts = np.unique(states[:, index], return_counts=True)
            result[str(bus)] = int(values[np.argmax(counts)])
        return result


class PmuGridInferencePipeline:
    """Apply a trained V2 model to raw or synthetic 8-PMU CSV folders."""

    def __init__(self, config: InferenceConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        artifact = joblib.load(self.config.model_path)
        frames = read_pmu_frames_from_dir(self.config.input_dir, artifact["pmu_buses"])
        window_sec = float(self.config.window_sec if self.config.window_sec is not None else artifact["window_sec"])
        centers = self._centers(frames, window_sec)
        features = feature_frame([extract_window_features(frames, float(center), window_sec) for center in centers])
        model = self._model_from_artifact(artifact)
        predictions = model.predict(features, threshold=self.config.threshold)
        segments = SegmentBuilder().build(
            centers,
            predictions["event_labels"],
            predictions["locations"],
            predictions["event_confidence"],
            predictions["bus_states"],
            self.config.stride_sec,
        )
        output = {
            "input_dir": str(self.config.input_dir.resolve()),
            "model": str(self.config.model_path.resolve()),
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

    def _model_from_artifact(self, artifact: dict[str, Any]) -> PmuGridModel:
        model = PmuGridModel(ModelFactory(artifact["model_name"], seed=0), artifact["feature_names"])
        model.event_model = artifact["event_model"]
        model.location_model = artifact["location_model"]
        model.bus_state_model = artifact["bus_state_model"]
        model.event_codec.encoder = artifact["event_encoder"]
        model.event_codec.classes = artifact["event_classes"]
        model.location_codec.encoder = artifact["location_encoder"]
        model.location_codec.classes = artifact["location_classes"]
        return model
