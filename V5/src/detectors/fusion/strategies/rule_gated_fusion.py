from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detectors.domain.models.branch_score import BranchScore


@dataclass(slots=True)
class RuleGatedFusion:
    cyber_weight: float = 0.5
    physical_weight: float = 0.5
    cyber_gate_threshold: float = 0.72
    physical_gate_threshold: float = 0.80
    gate_boost: float = 0.30
    cyber_override_floor: float = 0.85
    cyber_physical_disagreement_floor: float = 0.42
    strong_cyber_weak_physical_floor: float = 0.74
    strong_cyber_weak_physical_max: float = 0.92

    def fuse(self, cyber: BranchScore, physical: BranchScore) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        n = min(len(cyber.probability), len(physical.probability))
        p_c = cyber.probability[:n]
        p_p = physical.probability[:n]
        cy_w = np.full((n,), self.cyber_weight, dtype=float)
        ph_w = np.full((n,), self.physical_weight, dtype=float)
        strong_cyber = p_c >= self.cyber_gate_threshold
        strong_physical = p_p >= self.physical_gate_threshold
        strong_cyber_weak_physical = strong_cyber & (p_p <= self.cyber_physical_disagreement_floor)
        cy_w[strong_cyber] += self.gate_boost
        ph_w[strong_physical] += self.gate_boost
        denom = np.maximum(cy_w + ph_w, 1e-9)
        fused = np.clip((cy_w * p_c + ph_w * p_p) / denom, 0.0, 1.0)
        fused = np.where(strong_cyber & (p_c >= self.cyber_override_floor), np.maximum(fused, p_c), fused)
        cyber_override = np.clip(
            0.88 * p_c + 0.12 * np.maximum(p_p, self.strong_cyber_weak_physical_floor),
            self.strong_cyber_weak_physical_floor,
            self.strong_cyber_weak_physical_max,
        )
        fused = np.where(strong_cyber_weak_physical, np.maximum(fused, cyber_override), fused)
        return fused, {
            "cyber_weight": cy_w,
            "physical_weight": ph_w,
            "strong_cyber_gate": strong_cyber.astype(float),
            "strong_physical_gate": strong_physical.astype(float),
            "strong_cyber_weak_physical_gate": strong_cyber_weak_physical.astype(float),
        }
