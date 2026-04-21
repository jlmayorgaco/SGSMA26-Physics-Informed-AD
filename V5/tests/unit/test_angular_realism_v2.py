from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.m9.final_polish import run_angular_realism_v2


def _frame(bus: str, n: int = 200, shift: float = 0.0) -> pd.DataFrame:
    t = np.linspace(0.0, 4.0 * np.pi, n)
    base = 12.0 * np.sin(t) + shift
    return pd.DataFrame(
        {
            f"{bus}_VA_ANG": base,
            f"{bus}_VB_ANG": base - 120.0,
            f"{bus}_VC_ANG": base + 120.0,
            f"{bus}_IA_ANG": 0.8 * base,
            f"{bus}_IB_ANG": 0.8 * base - 120.0,
            f"{bus}_IC_ANG": 0.8 * base + 120.0,
        }
    )


def test_angular_realism_v2_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.simulation.m9.final_polish._load_reference_frames",
        lambda _: {"BUS2": _frame("BUS2", shift=0.0)},
    )
    monkeypatch.setattr(
        "src.simulation.m9.final_polish._load_sim_frames",
        lambda _: {"BUS2": _frame("BUS2", shift=3.0)},
    )
    summary = run_angular_realism_v2([], tmp_path, tmp_path)
    assert (tmp_path / "metrics" / "angular_realism_v2.csv").exists()
    assert (tmp_path / "metadata" / "angular_realism_v2_summary.json").exists()
    assert "overall_angular_realism_pass" in summary
    assert "angle_distribution_pass" in summary

