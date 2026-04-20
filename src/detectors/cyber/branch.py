from __future__ import annotations

import numpy as np

from src.detectors.configs import CyberModelConfig, CyberRulesConfig
from src.detectors.cyber.rules import compute_cyber_rule_features, cyber_rule_probabilities
from src.detectors.cyber.tabular import TabularCyberDetector
from src.detectors.models import BranchScores, WindowedBatch


class CyberHybridBranch:
    def __init__(self, rules_config: CyberRulesConfig, model_config: CyberModelConfig) -> None:
        self.rules_config = rules_config
        self.model = TabularCyberDetector(model_config)
        self.data_present_index: int | None = None

    def fit(self, batch: WindowedBatch) -> None:
        self.data_present_index = batch.feature_names.index("DATA_PRESENT") if "DATA_PRESENT" in batch.feature_names else None
        self.model.fit(batch.x, batch.y)

    def predict(self, batch: WindowedBatch) -> BranchScores:
        if self.data_present_index is None and "DATA_PRESENT" in batch.feature_names:
            self.data_present_index = batch.feature_names.index("DATA_PRESENT")
        rule_feats = compute_cyber_rule_features(batch.x, batch.timestamps, data_present_index=self.data_present_index)
        rule_prob = cyber_rule_probabilities(rule_feats, self.rules_config)
        ml_prob = self.model.predict(batch.x)
        probs = np.clip(0.5 * rule_prob + 0.5 * ml_prob, 0.0, 1.0)
        details = {
            "rule_prob_mean": float(rule_prob.mean()) if len(rule_prob) else 0.0,
            "ml_prob_mean": float(ml_prob.mean()) if len(ml_prob) else 0.0,
            "tabular_backend": self.model.backend,
        }
        return BranchScores(probabilities=probs, details=details)

