from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from src.visualization.state_estimation_plots import plot_error_summaries, plot_selected_bus_voltage


def test_plots_generated() -> None:
    out = Path("output") / "_test_runs" / f"m6_plots_{uuid4().hex[:8]}"
    t = np.asarray([0.0, 0.1, 0.2, 0.3])
    bus_order = ["BUS1", "BUS2"]
    est = np.asarray([[1 + 0j, 0.98 + 0.01j]] * 4, dtype=complex)
    true = np.asarray([[1 + 0j, 1.0 + 0j]] * 4, dtype=complex)
    per_bus = pd.DataFrame([{"bus": "BUS1", "rmse_mag_pu": 0.0}, {"bus": "BUS2", "rmse_mag_pu": 0.02}])
    diag = pd.DataFrame([{"timestamp": x, "residual_norm": 0.01} for x in t])

    p1 = plot_selected_bus_voltage(t, bus_order, est, true, ["BUS1"], out)
    p2 = plot_error_summaries(t, bus_order, est, true, per_bus, diag, out)
    assert all(Path(p).exists() for p in p1.values())
    assert all(Path(p).exists() for p in p2.values())

