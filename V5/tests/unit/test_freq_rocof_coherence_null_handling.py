from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.m9.final_polish import run_freq_rocof_coherence_v2


def test_freq_rocof_coherence_null_handling(tmp_path, monkeypatch) -> None:
    n = 120
    t = np.arange(n, dtype=float)
    ref = pd.DataFrame({"BUS2_Freq": 60.0 + 0.02 * np.sin(t / 8.0), "BUS2_ROCOF": 0.02 * np.cos(t / 8.0)})
    sim = pd.DataFrame({"BUS2_Freq": 60.0 + 0.01 * np.sin(t / 8.0), "BUS2_ROCOF": np.zeros(n)})
    monkeypatch.setattr("src.simulation.m9.final_polish._load_reference_frames", lambda _: {"BUS2": ref})
    monkeypatch.setattr("src.simulation.m9.final_polish._load_sim_frames", lambda _: {"BUS2": sim})
    summary = run_freq_rocof_coherence_v2([], tmp_path, tmp_path)
    assert summary["overall_status"] != "pass"
    reasons = summary["reasons_not_computable"]
    assert any(r["reason"] == "sim_coherence_null" for r in reasons)

