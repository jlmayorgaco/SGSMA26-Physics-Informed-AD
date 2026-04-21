from __future__ import annotations

from typing import Protocol

from src.detectors.domain.models.detection_input import DetectionInput


class FeatureExtractor(Protocol):
    """Build branch-ready features from aligned detection input."""

    def extract(self, inputs: DetectionInput) -> dict[str, object]: ...

