"""Staged detector/classifier/localizer model for 8-PMU event inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.ml.models import build_bus_state_estimator, build_event_estimator, resolve_model_name
from src.ml.pmu_grid_pipeline import ALL_BUSES, BUS_STATE_COLUMNS, LabelCodec
from src.ml.staged.physics import PhysicsFeatureEngineer


@dataclass(frozen=True)
class StagedModelConfig:
    detector_model: str = "lightgbm"
    event_model: str = "lightgbm"
    location_model: str = "lightgbm"
    state_model: str = "extratrees"
    use_physics_features: bool = True
    detector_threshold: float = 0.45


class StagedPmuGridModel:
    """Detector -> event classifier -> location head -> 39-bus state estimator."""

    def __init__(self, config: StagedModelConfig | None = None, raw_feature_names: list[str] | None = None) -> None:
        self.config = config or StagedModelConfig()
        self.raw_feature_names = raw_feature_names or []
        self.augmented_feature_names: list[str] = []
        self.event_codec = LabelCodec()
        self.location_codec = LabelCodec()
        self.physics = PhysicsFeatureEngineer(include_original=True)
        self.detector_model: Pipeline | None = None
        self.event_model: Pipeline | None = None
        self.location_model: Pipeline | None = None
        self.bus_state_model: Pipeline | None = None

    @property
    def resolved_name(self) -> str:
        parts = [
            resolve_model_name(self.config.detector_model),
            resolve_model_name(self.config.event_model),
            resolve_model_name(self.config.state_model),
        ]
        return "staged_" + "_".join(parts)

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame, bus_states: pd.DataFrame, train_idx: np.ndarray, seed: int) -> None:
        self.raw_feature_names = list(features.columns)
        train_raw = features.iloc[train_idx].reindex(columns=self.raw_feature_names, fill_value=np.nan)
        if self.config.use_physics_features:
            self.physics.fit(train_raw)
        augmented = self._transform_features(features)
        self.augmented_feature_names = list(augmented.columns)

        train_features = augmented.iloc[train_idx]
        event_labels = targets["event_label"].to_numpy(dtype=int)
        detector_y = (event_labels != 0).astype(int)
        positive = event_labels != 0

        self.detector_model = build_event_estimator(self.config.detector_model, 2, seed)
        self.detector_model.fit(train_features, detector_y[train_idx])

        self.event_codec.fit_transform(pd.Series(event_labels[positive], dtype=int))
        self.event_model = build_event_estimator(self.config.event_model, len(self.event_codec.classes), seed + 17)
        train_positive = positive[train_idx]
        if np.any(train_positive):
            event_encoded = np.full(len(targets), -1, dtype=int)
            event_encoded[positive] = self.event_codec.transform(pd.Series(event_labels[positive], dtype=int))
            self.event_model.fit(train_features[train_positive], event_encoded[train_idx][train_positive])
        else:
            self.event_model.fit(train_features, np.zeros(len(train_features), dtype=int))

        loc_values = targets.loc[positive, "location"] if np.any(positive) else pd.Series(["normal"])
        self.location_codec.fit_transform(loc_values)
        self.location_model = build_event_estimator(self.config.location_model, len(self.location_codec.classes), seed + 31)
        if np.any(train_positive):
            loc_encoded = np.full(len(targets), -1, dtype=int)
            loc_encoded[positive] = self.location_codec.transform(targets.loc[positive, "location"])
            self.location_model.fit(train_features[train_positive], loc_encoded[train_idx][train_positive])
        else:
            self.location_model.fit(train_features, np.zeros(len(train_features), dtype=int))

        self.bus_state_model = build_bus_state_estimator(self.config.state_model, seed + 101)
        self.bus_state_model.fit(train_features, bus_states.iloc[train_idx].to_numpy(dtype=int))

    def predict(self, features: pd.DataFrame, threshold: float | None = None, return_features: bool = False) -> dict[str, Any]:
        self._require_fitted()
        augmented = self._transform_features(features).reindex(columns=self.augmented_feature_names, fill_value=np.nan)
        detector_confidence = self._positive_probability(self.detector_model, augmented)
        cutoff = self.config.detector_threshold if threshold is None else float(threshold)
        event_labels = np.zeros(len(augmented), dtype=int)
        locations = np.array(["normal"] * len(augmented), dtype=object)
        positive = detector_confidence >= cutoff
        event_confidence = detector_confidence.copy()
        if np.any(positive):
            event_encoded = self.event_model.predict(augmented.loc[positive])
            event_labels[positive] = np.asarray(self.event_codec.decode(event_encoded), dtype=int)
            event_confidence[positive] = detector_confidence[positive] * self._max_probability(self.event_model, augmented.loc[positive])
            location_encoded = self.location_model.predict(augmented.loc[positive])
            locations[positive] = np.asarray(self.location_codec.decode(location_encoded), dtype=object)
        bus_states = self.bus_state_model.predict(augmented).astype(int)
        bus_states[event_labels == 0, :] = 0
        result = {
            "event_labels": event_labels,
            "event_confidence": event_confidence,
            "locations": locations,
            "bus_states": bus_states,
        }
        if return_features:
            result["augmented_features"] = augmented
        return result

    def artifact(self, input_contract: dict[str, Any], window_sec: float) -> dict[str, Any]:
        self._require_fitted()
        return {
            "model_kind": "staged_physics",
            "config": asdict(self.config),
            "detector_model": self.detector_model,
            "event_model": self.event_model,
            "location_model": self.location_model,
            "bus_state_model": self.bus_state_model,
            "event_encoder": self.event_codec.encoder,
            "location_encoder": self.location_codec.encoder,
            "event_classes": self.event_codec.classes,
            "location_classes": self.location_codec.classes,
            "raw_feature_names": self.raw_feature_names,
            "augmented_feature_names": self.augmented_feature_names,
            "physics": self.physics,
            "pmu_buses": [2, 5, 6, 10, 19, 22, 29, 39],
            "all_buses": ALL_BUSES,
            "bus_state_columns": BUS_STATE_COLUMNS,
            "window_sec": float(window_sec),
            "model_name": self.resolved_name,
            "input_contract": input_contract,
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "StagedPmuGridModel":
        model = cls(StagedModelConfig(**artifact["config"]), artifact["raw_feature_names"])
        model.augmented_feature_names = artifact["augmented_feature_names"]
        model.physics = artifact["physics"]
        model.detector_model = artifact["detector_model"]
        model.event_model = artifact["event_model"]
        model.location_model = artifact["location_model"]
        model.bus_state_model = artifact["bus_state_model"]
        model.event_codec.encoder = artifact["event_encoder"]
        model.location_codec.encoder = artifact["location_encoder"]
        model.event_codec.classes = artifact["event_classes"]
        model.location_codec.classes = artifact["location_classes"]
        return model

    def _transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        raw = features.reindex(columns=self.raw_feature_names, fill_value=np.nan)
        if not self.config.use_physics_features:
            return raw
        return self.physics.transform(raw)

    def _positive_probability(self, model: Pipeline, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            return np.asarray(model.predict(features), dtype=float)
        proba = model.predict_proba(features)
        if isinstance(proba, list):
            proba = proba[0]
        arr = np.nan_to_num(np.asarray(proba, dtype=float), nan=0.0)
        return arr[:, -1] if arr.ndim == 2 and arr.shape[1] > 1 else arr.reshape(-1)

    def _max_probability(self, model: Pipeline, features: pd.DataFrame) -> np.ndarray:
        if not hasattr(model, "predict_proba"):
            return np.ones(len(features), dtype=float)
        proba = model.predict_proba(features)
        if isinstance(proba, list):
            proba = proba[0]
        return np.max(np.nan_to_num(np.asarray(proba, dtype=float), nan=0.0), axis=1)

    def _require_fitted(self) -> None:
        if any(item is None for item in [self.detector_model, self.event_model, self.location_model, self.bus_state_model]):
            raise RuntimeError("StagedPmuGridModel is not fitted.")
