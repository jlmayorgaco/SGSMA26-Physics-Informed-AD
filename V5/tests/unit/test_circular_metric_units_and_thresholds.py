from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.m9.final_polish import run_angular_realism_v2


def test_circular_metric_units_and_thresholds(tmp_path, monkeypatch) -> None:
    n = 120
    x = np.linspace(-20.0, 20.0, n)
    ref = pd.DataFrame({"BUS2_VA_ANG": x, "BUS2_VB_ANG": x - 120.0, "BUS2_VC_ANG": x + 120.0})
    sim = pd.DataFrame({"BUS2_VA_ANG": x + 4.0, "BUS2_VB_ANG": x - 116.0, "BUS2_VC_ANG": x + 124.0})
    monkeypatch.setattr("src.simulation.m9.final_polish._load_reference_frames", lambda _: {"BUS2": ref})
    monkeypatch.setattr("src.simulation.m9.final_polish._load_sim_frames", lambda _: {"BUS2": sim})
    summary = run_angular_realism_v2([], tmp_path, tmp_path)
    assert summary["metrics_units"]["wrapped_error"] == "deg"
    assert summary["metrics_units"]["phase_consistency_delta"] == "deg"
    assert summary["thresholds"]["p95_wrapped_error_deg_max"] <= 180.0
    assert "metric_interpretation" in summary
