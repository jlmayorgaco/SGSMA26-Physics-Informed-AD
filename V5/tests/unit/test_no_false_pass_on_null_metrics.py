from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.m9.final_polish import run_freq_rocof_coherence_v2


def test_no_false_pass_on_null_metrics(tmp_path, monkeypatch) -> None:
    n = 80
    t = np.arange(n, dtype=float)
    monkeypatch.setattr(
        "src.simulation.m9.final_polish._load_reference_frames",
        lambda _: {"BUS2": pd.DataFrame({"BUS2_Freq": 60.0 + 0.02 * np.sin(t), "BUS2_ROCOF": np.gradient(60.0 + 0.02 * np.sin(t))})},
    )
    monkeypatch.setattr(
        "src.simulation.m9.final_polish._load_sim_frames",
        lambda _: {"BUS2": pd.DataFrame({"BUS2_Freq": np.full(n, 60.0), "BUS2_ROCOF": np.full(n, 0.0)})},
    )
    summary = run_freq_rocof_coherence_v2([], tmp_path, tmp_path)
    assert summary["pass_freq_rocof_coherence_v2"] is False
    assert summary["overall_status"] in {"fail", "not_computable"}

