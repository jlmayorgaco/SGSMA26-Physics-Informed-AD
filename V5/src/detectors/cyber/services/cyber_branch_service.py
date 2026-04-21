from __future__ import annotations

import numpy as np

from src.detectors.cyber.features.cyber_feature_extractor import CyberFeatureExtractor
from src.detectors.cyber.ml.cyber_gbdt_detector import CyberGBDTDetector
from src.detectors.domain.models.branch_score import BranchScore
from src.detectors.domain.models.detection_input import DetectionInput


class CyberBranchService:
    def __init__(self, *, ml_weight: float = 0.5, rule_weight: float = 0.5) -> None:
        self.ml_weight = float(ml_weight)
        self.rule_weight = float(rule_weight)
        self.extractor = CyberFeatureExtractor()
        self.model = CyberGBDTDetector()
        self._is_fitted = False

    def fit(self, inputs: DetectionInput) -> None:
        batch = self.extractor.extract(inputs)
        if "y_binary" not in inputs.metadata.columns:
            raise ValueError("DetectionInput.metadata must include y_binary for fitting cyber branch")
        y = inputs.metadata["y_binary"].to_numpy(dtype=int)
        self.model.fit(batch.x, y, feature_names=batch.feature_names)
        self._is_fitted = True

    def score(self, inputs: DetectionInput) -> BranchScore:
        batch = self.extractor.extract(inputs)
        ml_prob = self.model.predict_proba(batch.x) if self._is_fitted else np.full((len(batch.x),), 0.5, dtype=float)
        strong_missing = (batch.rule_evidence.get("missing_ratio", np.zeros_like(batch.rule_score)) > 0.30) | (
            batch.rule_evidence.get("partial_dropout_ratio", np.zeros_like(batch.rule_score)) > 0.20
        )
        rule_override = np.where(strong_missing, np.maximum(batch.rule_score, 0.80), batch.rule_score)
        score = np.clip(self.ml_weight * ml_prob + self.rule_weight * rule_override, 0.0, 1.0)
        details = {
            "backend": self.model.backend,
            "rule_score_mean": float(rule_override.mean()) if len(rule_override) else 0.0,
            "ml_prob_mean": float(ml_prob.mean()) if len(ml_prob) else 0.0,
            "feature_importance": self.model.feature_importance(),
            "strong_missing_count": int(np.sum(strong_missing)),
            "evidence": {k: v.tolist() for k, v in batch.rule_evidence.items()},
        }
        return BranchScore(probability=score, raw_score=ml_prob, backend=f"cyber::{self.model.backend}", details=details)
