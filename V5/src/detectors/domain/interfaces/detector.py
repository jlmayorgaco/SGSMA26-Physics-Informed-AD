from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.domain.models.detection_output import DetectionOutput
from src.detectors.domain.models.detector_config import DetectorConfigV2
from src.detectors.domain.models.window_sample import WindowSample


class Detector(Protocol):
    """Top-level hybrid detector contract.

    Implementations coordinate branch detectors, fusion, and postprocessing.
    """

    def fit(self, samples: Sequence[WindowSample], config: DetectorConfigV2 | None = None) -> None: ...

    def predict(self, inputs: DetectionInput) -> DetectionOutput: ...

    def save(self, model_path: Path) -> None: ...

    @staticmethod
    def load(model_path: Path) -> "Detector": ...

