from __future__ import annotations

import numpy as np

from src.detectors.domain.models.branch_score import BranchScore
from src.detectors.domain.models.detection_output import DetectionOutput
from src.detectors.fusion.strategies.rule_gated_fusion import RuleGatedFusion


class DetectorFusionService:
    def __init__(self, *, decision_threshold: float = 0.5) -> None:
        self.strategy = RuleGatedFusion()
        self.decision_threshold = float(decision_threshold)

    def fuse(self, cyber_score: BranchScore, physical_score: BranchScore) -> DetectionOutput:
        p_abnormal, gates = self.strategy.fuse(cyber_score, physical_score)
        n = len(p_abnormal)
        p_c = cyber_score.probability[:n]
        p_p = physical_score.probability[:n]
        y = (p_abnormal >= self.decision_threshold).astype(int)
        evidence = []
        for i in range(n):
            evidence.append(
                {
                    "p_cyber": float(p_c[i]),
                    "p_physical": float(p_p[i]),
                    "cyber_weight": float(gates["cyber_weight"][i]),
                    "physical_weight": float(gates["physical_weight"][i]),
                    "strong_cyber_gate": bool(gates["strong_cyber_gate"][i] > 0.5),
                    "strong_physical_gate": bool(gates["strong_physical_gate"][i] > 0.5),
                }
            )
        return DetectionOutput(
            p_abnormal=p_abnormal,
            p_cyber=p_c,
            p_physical=p_p,
            y_pred_frame=y,
            y_pred_stable=y.copy(),
            evidence=evidence,
            diagnostics={
                "fusion_strategy": "rule_gated_fusion",
                "decision_threshold": self.decision_threshold,
            },
        )

