from __future__ import annotations

from pathlib import Path

import numpy as np

from src.detectors.physical.ml.tcn_physical_detector import TCNPhysicalDetector


def test_tcn_physical_detector_forward_train_save_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, size=(40, 16, 5))
    y = np.array([0] * 20 + [1] * 20, dtype=int)
    x[20:, :, :] += np.linspace(0.0, 1.0, 16)[None, :, None] * 0.7
    model = TCNPhysicalDetector(epochs=80)
    model.fit(x, y)
    p = model.predict_proba(x)
    assert p.shape == (40,)
    assert float(p[25:].mean()) > float(p[:10].mean())
    path = tmp_path / "tcn.pkl"
    model.save(path)
    loaded = TCNPhysicalDetector.load(path)
    p2 = loaded.predict_proba(x)
    assert p2.shape == (40,)

