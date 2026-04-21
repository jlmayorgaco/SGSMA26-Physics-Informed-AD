from __future__ import annotations

from typing import Protocol, Sequence

from src.detectors.domain.models.branch_score import BranchScore
from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.domain.models.window_sample import WindowSample


class BranchDetector(Protocol):
    """Reusable contract for cyber and physical detector branches."""

    def fit(self, samples: Sequence[WindowSample]) -> None: ...

    def score(self, inputs: DetectionInput) -> BranchScore: ...

