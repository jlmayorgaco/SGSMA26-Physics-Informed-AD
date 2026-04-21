from __future__ import annotations

import numpy as np

from src.detectors.domain.models.branch_score import BranchScore


class MeanFusionStrategyStub:
    """Phase-1 placeholder fusion strategy.

    This intentionally avoids advanced logic and only provides predictable behavior
    for pipeline wiring and tests.
    """

    def fuse(self, cyber: BranchScore, physical: BranchScore, metadata: object) -> BranchScore:
        n = min(len(cyber.probability), len(physical.probability))
        p_c = cyber.probability[:n]
        p_p = physical.probability[:n]
        fused = np.clip(0.5 * p_c + 0.5 * p_p, 0.0, 1.0)
        return BranchScore(
            probability=fused,
            backend="mean_stub",
            details={
                "p_cyber_mean": float(np.mean(p_c)) if n else 0.0,
                "p_physical_mean": float(np.mean(p_p)) if n else 0.0,
            },
        )

