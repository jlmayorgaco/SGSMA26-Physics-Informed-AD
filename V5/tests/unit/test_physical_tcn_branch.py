from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.configs import PhysicalModelConfig
from src.detectors.models import WindowedBatch
from src.detectors.physical.branch import PhysicalTemporalBranch


def test_physical_branch_detects_dynamic_shift() -> None:
    rng = np.random.default_rng(2)
    n, t, f = 30, 40, 3
    x = rng.normal(0.0, 0.1, size=(n, t, f))
    y = np.array([0] * 15 + [1] * 15, dtype=int)
    for i in range(15, n):
        x[i] += np.linspace(0.0, 1.0, t)[:, None] * 0.5
    batch = WindowedBatch(
        x=x,
        y=y,
        timestamps=np.arange(n, dtype=float),
        metadata=pd.DataFrame({"scenario_id": ["S"] * n}),
        feature_names=["A", "B", "C"],
    )
    branch = PhysicalTemporalBranch(PhysicalModelConfig(epochs=80))
    branch.fit(batch)
    probs = branch.predict(batch).probabilities
    assert probs.shape == (n,)
    assert float(probs[20:].mean()) > float(probs[:10].mean())

