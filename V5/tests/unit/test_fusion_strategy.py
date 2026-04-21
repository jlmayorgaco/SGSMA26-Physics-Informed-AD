from __future__ import annotations

import numpy as np

from src.detectors.domain.models.branch_score import BranchScore
from src.detectors.fusion.strategies.rule_gated_fusion import RuleGatedFusion


def test_fusion_strategy_gating_is_deterministic() -> None:
    fusion = RuleGatedFusion(cyber_weight=0.5, physical_weight=0.5, cyber_gate_threshold=0.8, gate_boost=0.3)
    cyber = BranchScore(probability=np.array([0.2, 0.9, 0.85]))
    physical = BranchScore(probability=np.array([0.4, 0.4, 0.9]))
    p, details = fusion.fuse(cyber, physical)
    assert p.shape == (3,)
    assert float(p[1]) > 0.5
    assert float(details["cyber_weight"][1]) > 0.5
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


def test_fusion_strategy_boosts_strong_cyber_when_physical_is_weak() -> None:
    fusion = RuleGatedFusion(cyber_gate_threshold=0.8, cyber_physical_disagreement_floor=0.35)
    cyber = BranchScore(probability=np.array([0.84]))
    physical = BranchScore(probability=np.array([0.12]))
    p, details = fusion.fuse(cyber, physical)
    assert float(p[0]) >= 0.74
    assert bool(details["strong_cyber_weak_physical_gate"][0])
