from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.m9.final_polish import run_freq_rocof_coherence_v2


def test_freq_rocof_coherence_computation(tmp_path, monkeypatch) -> None:
    n = 150
    t = np.arange(n, dtype=float)
    ref_freq = 60.0 + 0.03 * np.sin(t / 7.0)
    ref_rocof = np.gradient(ref_freq)
    sim_freq = 60.0 + 0.031 * np.sin((t + 1.0) / 7.0)
    sim_rocof = np.gradient(sim_freq)
    monkeypatch.setattr(
        "src.simulation.m9.final_polish._load_reference_frames",
        lambda _: {"BUS2": pd.DataFrame({"BUS2_Freq": ref_freq, "BUS2_ROCOF": ref_rocof})},
    )
    monkeypatch.setattr(
        "src.simulation.m9.final_polish._load_sim_frames",
        lambda _: {"BUS2": pd.DataFrame({"BUS2_Freq": sim_freq, "BUS2_ROCOF": sim_rocof})},
    )
    summary = run_freq_rocof_coherence_v2([], tmp_path, tmp_path)
    assert (tmp_path / "metrics" / "freq_rocof_coherence_metrics.csv").exists()
    assert "counts" in summary and "pass" in summary["counts"]
    assert "per_bus" in summary

