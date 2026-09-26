from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.estimation.metrics import mae, rel_rmse, rmse, safe_corr


@pytest.mark.regression
def test_snapshot_estimation_summary(snapshot_dir: Path) -> None:
    sim = np.array([1.0, 2.0, 3.0])
    est = np.array([1.0, 2.5, 2.5])

    summary = {
        "rmse": rmse(sim, est),
        "mae": mae(sim, est),
        "relative_rmse": rel_rmse(sim, est),
        "corr": safe_corr(sim, est),
    }

    expected = json.loads((snapshot_dir / "snapshot_estimation_summary.json").read_text(encoding="utf-8"))
    assert summary.keys() == expected.keys()
    for key, value in summary.items():
        assert np.isclose(value, expected[key], atol=1e-10)
