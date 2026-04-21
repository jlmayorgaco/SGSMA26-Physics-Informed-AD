from __future__ import annotations

import numpy as np

from src.detectors.domain.models.branch_score import BranchScore
from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.physical.features.physical_feature_extractor import PhysicalFeatureExtractor
from src.detectors.physical.ml.tcn_physical_detector import TCNPhysicalDetector
from src.detectors.physical.ml.xgb_physical_detector import XGBPhysicalDetector


class PhysicalBranchService:
    def __init__(self, *, temporal_weight: float = 0.65, rule_weight: float = 0.35, use_aux_baseline: bool = False) -> None:
        self.temporal_weight = float(temporal_weight)
        self.rule_weight = float(rule_weight)
        self.extractor = PhysicalFeatureExtractor()
        self.temporal_model = TCNPhysicalDetector()
        self.aux_model = XGBPhysicalDetector()
        self.use_aux_baseline = bool(use_aux_baseline)
        self._is_fitted = False

    def fit(self, inputs: DetectionInput) -> None:
        if "y_binary" not in inputs.metadata.columns:
            raise ValueError("DetectionInput.metadata must include y_binary for fitting physical branch")
        y = inputs.metadata["y_binary"].to_numpy(dtype=int)
        batch = self.extractor.extract(inputs)
        self.temporal_model.fit(batch.temporal_x, y)
        if self.use_aux_baseline:
            self.aux_model.fit(batch.aux_x, y)
        self._is_fitted = True

    def score(self, inputs: DetectionInput) -> BranchScore:
        batch = self.extractor.extract(inputs)
        temporal_prob = self.temporal_model.predict_proba(batch.temporal_x) if self._is_fitted else np.full((len(batch.temporal_x),), 0.5)
        if self.use_aux_baseline and self._is_fitted:
            aux_prob = self.aux_model.predict_proba(batch.aux_x)
            temporal_prob = 0.7 * temporal_prob + 0.3 * aux_prob
        score = np.clip(self.temporal_weight * temporal_prob + self.rule_weight * batch.heuristic_score, 0.0, 1.0)
        details = {
            "backend": self.temporal_model.backend,
            "temporal_prob_mean": float(temporal_prob.mean()) if len(temporal_prob) else 0.0,
            "heuristic_score_mean": float(batch.heuristic_score.mean()) if len(batch.heuristic_score) else 0.0,
            "evidence": {k: v.tolist() for k, v in batch.heuristic_evidence.items()},
        }
        return BranchScore(probability=score, raw_score=temporal_prob, backend=f"physical::{self.temporal_model.backend}", details=details)

