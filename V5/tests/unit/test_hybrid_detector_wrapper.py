from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.domain.services.hybrid_event_detector import HybridEventDetector


def test_hybrid_detector_wrapper_end_to_end_callgraph() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 1.0, size=(20, 12, 6))
    x[:, :, -1] = np.clip(np.linspace(1.0, 0.2, 12)[None, :], 0.0, 1.0)
    y = np.array([0] * 10 + [1] * 10, dtype=int)
    metadata = pd.DataFrame(
        {
            "y_binary": y,
            "scenario_id": ["SIM"] * 20,
            "event_coarse": [0] * 10 + [5] * 10,
            "is_cyber_event": [False] * 10 + [True] * 10,
            "is_physical_event": [False] * 20,
            "is_concurrent_event": [False] * 20,
        }
    )
    inp = DetectionInput(
        scenario_id="SIM",
        split="train",
        x_windows=x,
        timestamps=np.arange(20, dtype=float) * 0.1,
        feature_names=["F1", "F2", "F3", "F4", "F5", "DATA_PRESENT"],
        metadata=metadata,
    )
    det = HybridEventDetector()
    det.fit_inputs(inp)
    out = det.predict(inp)
    assert len(out.p_abnormal) == 20
    assert len(out.y_pred_stable) == 20
    assert isinstance(out.chunks, list)

