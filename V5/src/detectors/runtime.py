from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from src.detectors.configs import DetectorConfig
from src.detectors.cyber.branch import CyberHybridBranch
from src.detectors.fusion.strategy import WeightedFusionStrategy
from src.detectors.models import DetectionRunArtifacts
from src.detectors.postprocessing.chunker import chunk_events
from src.detectors.postprocessing.hysteresis import apply_hysteresis
from src.detectors.postprocessing.state_machine import enforce_event_state_machine
from src.detectors.metrics import compute_binary_metrics
from src.detectors.physical.branch import PhysicalTemporalBranch


class HybridDetector:
    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.cyber = CyberHybridBranch(config.cyber_rules, config.cyber_model)
        self.physical = PhysicalTemporalBranch(config.physical_model)
        self.fusion = WeightedFusionStrategy(config.fusion)
        self.feature_names: list[str] = []

    def fit(self, batch) -> None:
        self.feature_names = list(batch.feature_names)
        self.cyber.fit(batch)
        self.physical.fit(batch)

    def predict_scores(self, batch) -> tuple[np.ndarray, dict[str, float | str]]:
        cyber_scores = self.cyber.predict(batch)
        physical_scores = self.physical.predict(batch)
        fused = self.fusion.fuse(cyber_scores.probabilities, physical_scores.probabilities, batch.metadata)
        details: dict[str, float | str] = {
            "cyber_mean_probability": float(cyber_scores.probabilities.mean()) if len(cyber_scores.probabilities) else 0.0,
            "physical_mean_probability": float(physical_scores.probabilities.mean()) if len(physical_scores.probabilities) else 0.0,
            **cyber_scores.details,
            **physical_scores.details,
        }
        return np.clip(fused, 0.0, 1.0), details

    def evaluate(self, batch) -> DetectionRunArtifacts:
        scores, details = self.predict_scores(batch)
        hysteresis_pred = apply_hysteresis(scores, self.config.postprocessing)
        smoothed = enforce_event_state_machine(hysteresis_pred, self.config.postprocessing)
        chunks = chunk_events(smoothed, batch.timestamps, scores, self.config.postprocessing)
        metrics = compute_binary_metrics(batch.y, scores, threshold=self.config.fusion.decision_threshold)
        frame = pd.DataFrame(
            {
                "timestamp": batch.timestamps,
                "y_true": batch.y,
                "score": scores,
                "y_pred_hysteresis": hysteresis_pred,
                "y_pred_state_machine": smoothed,
            }
        )
        if not batch.metadata.empty:
            frame = pd.concat([frame, batch.metadata.reset_index(drop=True)], axis=1)
        return DetectionRunArtifacts(metrics=metrics, frame_predictions=frame, event_chunks=chunks, branch_summary=details)

    def save(self, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as fh:
            pickle.dump(self, fh)
        meta_path = model_path.with_suffix(".json")
        meta_path.write_text(json.dumps({"config": asdict(self.config), "feature_names": self.feature_names}, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def load(model_path: Path) -> "HybridDetector":
        with model_path.open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, HybridDetector):
            raise TypeError(f"Unexpected model type at {model_path}: {type(obj)!r}")
        return obj

