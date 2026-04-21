from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Non0FusionOutput:
    p_non0: np.ndarray
    y_frame: np.ndarray
    attribution: list[str]
    confidence: np.ndarray
    trigger_reason: list[str]


@dataclass(slots=True)
class Non0Fusion:
    event5_trigger_min: float = 0.92
    event7_trigger_min: float = 0.80
    physical_trigger_min: float = 0.84
    branch_agreement_min: float = 0.68
    event5_override: float = 0.96
    event7_override: float = 0.90
    physical_override: float = 0.92
    decision_threshold: float = 0.70
    quiet_state_veto_max: float = 0.52

    def fuse(self, *, p_event5: np.ndarray, p_event7: np.ndarray, p_physical: np.ndarray) -> Non0FusionOutput:
        n = min(len(p_event5), len(p_event7), len(p_physical))
        e5 = np.asarray(p_event5[:n], dtype=float)
        e7 = np.asarray(p_event7[:n], dtype=float)
        phy = np.asarray(p_physical[:n], dtype=float)
        strong5 = e5 >= self.event5_trigger_min
        strong7 = e7 >= self.event7_trigger_min
        strongp = phy >= self.physical_trigger_min
        strong_count = strong5.astype(int) + strong7.astype(int) + strongp.astype(int)
        pair_agree = (
            (np.minimum(e5, e7) >= self.branch_agreement_min)
            | (np.minimum(e5, phy) >= self.branch_agreement_min)
            | (np.minimum(e7, phy) >= self.branch_agreement_min)
        )
        weak_mix = np.clip(0.25 * e5 + 0.30 * e7 + 0.20 * phy, 0.0, 1.0)
        strong_mix = np.clip(0.45 * np.maximum(e5, e7) + 0.30 * e7 + 0.25 * phy, 0.0, 1.0)
        fused = np.where((strong_count >= 1) | pair_agree, strong_mix, weak_mix)
        fused = np.where(strong_count >= 2, np.maximum(fused, np.maximum.reduce([e5, e7, phy])), fused)
        fused = np.where(e5 >= self.event5_override, np.maximum(fused, e5), fused)
        fused = np.where(e7 >= self.event7_override, np.maximum(fused, e7), fused)
        fused = np.where(phy >= self.physical_override, np.maximum(fused, phy), fused)
        quiet = (e5 < self.quiet_state_veto_max) & (e7 < self.quiet_state_veto_max) & (phy < self.quiet_state_veto_max)
        fused = np.where(quiet, np.minimum(fused, self.quiet_state_veto_max), fused)
        y = (fused >= self.decision_threshold).astype(int)
        conf = np.clip(np.abs(fused - self.decision_threshold) / max(self.decision_threshold, 1e-6), 0.0, 1.0)
        attr: list[str] = []
        reasons: list[str] = []
        for i in range(n):
            if e5[i] >= e7[i] and e5[i] >= phy[i]:
                attr.append("event5")
                reasons.append("event5_max")
            elif e7[i] >= phy[i]:
                attr.append("event7")
                reasons.append("event7_max")
            else:
                attr.append("physical")
                reasons.append("physical_max")
            if quiet[i]:
                reasons[-1] = reasons[-1] + "::quiet_veto"
            elif strong_count[i] >= 2:
                reasons[-1] = reasons[-1] + "::multi_branch"
            elif pair_agree[i]:
                reasons[-1] = reasons[-1] + "::pair_agreement"
        return Non0FusionOutput(p_non0=fused, y_frame=y, attribution=attr, confidence=conf, trigger_reason=reasons)
