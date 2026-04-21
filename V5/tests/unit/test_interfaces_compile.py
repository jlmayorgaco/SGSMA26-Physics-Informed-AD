from __future__ import annotations

from pathlib import Path

import numpy as np

from src.detectors.domain.interfaces.branch_detector import BranchDetector
from src.detectors.domain.interfaces.detector import Detector
from src.detectors.domain.interfaces.feature_extractor import FeatureExtractor
from src.detectors.domain.interfaces.fusion_strategy import FusionStrategy
from src.detectors.domain.interfaces.postprocessor import Postprocessor
from src.detectors.domain.interfaces.preprocessor import Preprocessor
from src.detectors.domain.models.branch_score import BranchScore
from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.domain.models.detection_output import DetectionOutput
from src.detectors.domain.models.window_sample import WindowSample


class _DetectorStub:
    def fit(self, samples, config=None) -> None:
        return None

    def predict(self, inputs: DetectionInput) -> DetectionOutput:
        n = len(inputs.timestamps)
        return DetectionOutput(
            p_abnormal=np.zeros(n),
            p_cyber=np.zeros(n),
            p_physical=np.zeros(n),
            y_pred_frame=np.zeros(n, dtype=int),
            y_pred_stable=np.zeros(n, dtype=int),
        )

    def save(self, model_path: Path) -> None:
        model_path.write_text("stub", encoding="utf-8")

    @staticmethod
    def load(model_path: Path) -> "_DetectorStub":
        _ = model_path.read_text(encoding="utf-8")
        return _DetectorStub()


def test_interface_stubs_compile() -> None:
    det: Detector = _DetectorStub()
    branch: BranchDetector = type(
        "BranchStub",
        (),
        {
            "fit": lambda self, samples: None,
            "score": lambda self, inputs: BranchScore(probability=np.zeros(len(inputs.timestamps))),
        },
    )()
    pre: Preprocessor = type("PreStub", (), {"fit": lambda self, frames: None, "transform": lambda self, frame, scenario_id, split: []})()
    feat: FeatureExtractor = type("FeatStub", (), {"extract": lambda self, inputs: {"x": inputs.x_windows}})()
    fusion: FusionStrategy = type("FusionStub", (), {"fuse": lambda self, c, p, m: BranchScore(probability=0.5 * c.probability + 0.5 * p.probability)})()
    post: Postprocessor = type(
        "PostStub",
        (),
        {
            "smooth": lambda self, p: np.asarray(p),
            "state_machine": lambda self, b: np.asarray(b, dtype=int),
            "chunk": lambda self, b, t, p: [],
        },
    )()
    assert det is not None
    assert branch is not None
    assert pre is not None
    assert feat is not None
    assert fusion is not None
    assert post is not None
    assert isinstance(WindowSample.__name__, str)

