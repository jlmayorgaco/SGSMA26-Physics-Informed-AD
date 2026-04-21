from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.domain.services.hybrid_event_detector import HybridEventDetector


def test_phase2_branch_fusion_flow() -> None:
    rng = np.random.default_rng(10)
    x = rng.normal(0.0, 1.0, size=(24, 16, 8))
    x[:, :, -1] = np.clip(np.linspace(1.0, 0.1, 16)[None, :], 0.0, 1.0)
    y = np.array([0] * 12 + [1] * 12, dtype=int)
    meta = pd.DataFrame(
        {
            "y_binary": y,
            "scenario_id": ["SIM"] * 24,
            "event_coarse": [0] * 12 + [6] * 12,
            "is_cyber_event": [False] * 12 + [True] * 12,
            "is_physical_event": [False] * 12 + [True] * 12,
            "is_concurrent_event": [False] * 12 + [True] * 12,
            "ESTIMATOR_DIFFICULTY_SCORE": np.linspace(0.0, 1.0, 24),
        }
    )
    inp = DetectionInput(
        scenario_id="SIM",
        split="train",
        x_windows=x,
        timestamps=np.arange(24, dtype=float) * 0.1,
        feature_names=[f"F{i}" for i in range(7)] + ["DATA_PRESENT"],
        metadata=meta,
    )
    det = HybridEventDetector()
    det.fit_inputs(inp)
    out = det.predict(inp)
    assert len(out.p_abnormal) == 24
    assert len(out.evidence) == 24
    assert out.y_pred_stable.dtype.kind in {"i", "u"}

