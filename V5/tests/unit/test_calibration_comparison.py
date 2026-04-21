from __future__ import annotations

import numpy as np

from src.detectors.training.evaluators.calibration_models import CalibratorFactory


def test_calibration_comparison() -> None:
    y = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    p = np.array([0.15, 0.35, 0.45, 0.55, 0.75, 0.90], dtype=float)
    selection = CalibratorFactory.select_best(y, p)
    assert selection.selected_name in {"identity", "temperature", "platt", "isotonic_bin"}
    assert set(selection.comparison["method"].tolist()) == {"identity", "temperature", "platt", "isotonic_bin"}
