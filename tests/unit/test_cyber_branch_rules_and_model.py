from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.configs import CyberModelConfig, CyberRulesConfig
from src.detectors.cyber.branch import CyberHybridBranch
from src.detectors.models import WindowedBatch


def test_cyber_branch_outputs_probability() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, size=(24, 32, 4))
    # Last channel acts as DATA_PRESENT with missing bursts for positive windows.
    x[:12, :, -1] = 1.0
    x[12:, :, -1] = 0.2
    y = np.array([0] * 12 + [1] * 12, dtype=int)
    batch = WindowedBatch(
        x=x,
        y=y,
        timestamps=np.arange(24, dtype=float) * 0.1,
        metadata=pd.DataFrame({"scenario_id": ["S"] * 24}),
        feature_names=["F1", "F2", "F3", "DATA_PRESENT"],
    )
    branch = CyberHybridBranch(CyberRulesConfig(), CyberModelConfig(prefer_lightgbm=False, epochs=60))
    branch.fit(batch)
    out = branch.predict(batch)
    assert out.probabilities.shape == (24,)
    assert float(out.probabilities[18]) > float(out.probabilities[4])

