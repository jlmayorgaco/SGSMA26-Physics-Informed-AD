from __future__ import annotations

from typing import Protocol

from src.detectors.domain.models.branch_score import BranchScore


class FusionStrategy(Protocol):
    """Combine branch scores into a detector-level score payload."""

    def fuse(self, cyber: BranchScore, physical: BranchScore, metadata: object) -> BranchScore: ...

