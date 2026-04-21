from __future__ import annotations

from pathlib import Path

import numpy as np

from src.detectors.cyber.ml.cyber_gbdt_detector import CyberGBDTDetector


def test_cyber_gbdt_detector_train_predict_save_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=(60, 10))
    y = (x[:, 0] + 0.3 * x[:, 1] > 0.2).astype(int)
    model = CyberGBDTDetector(prefer_lightgbm=False, epochs=80)
    model.fit(x, y, feature_names=[f"f{i}" for i in range(x.shape[1])])
    p = model.predict_proba(x)
    assert p.shape == (60,)
    assert float(p.mean()) > 0.0
    path = tmp_path / "cyber.pkl"
    model.save(path)
    loaded = CyberGBDTDetector.load(path)
    p2 = loaded.predict_proba(x)
    assert p2.shape == (60,)

