from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES
from src.simulation.m9.final_polish import recalibrate_cyber_patterns_v2


def _write_bus_csv(path, bus: str, n: int = 160) -> None:
    cols = {f"{bus}_{s}": np.linspace(0.0, 1.0, n) for s in PMU_MEASUREMENT_SUFFIXES}
    cols["Event"] = np.zeros(n, dtype=int)
    cols["DATA_PRESENT"] = np.ones(n, dtype=int)
    pd.DataFrame(cols).to_csv(path, index=False)


def test_partial_dropout_representation(tmp_path, monkeypatch) -> None:
    base = tmp_path / "data" / "scenarios"
    s1 = base / "SIM1001"
    s2 = base / "SIM1002"
    (s1 / "pmu").mkdir(parents=True, exist_ok=True)
    (s2 / "pmu").mkdir(parents=True, exist_ok=True)
    (s1 / "scenario_manifest.json").write_text(json.dumps({"difficulty_level": "hard", "event_coarse": 5}), encoding="utf-8")
    (s2 / "scenario_manifest.json").write_text(json.dumps({"difficulty_level": "adversarial", "event_coarse": 8}), encoding="utf-8")
    _write_bus_csv(s1 / "pmu" / "Bus29_Competition_Data_sim.csv", "BUS29")
    _write_bus_csv(s2 / "pmu" / "Bus29_Competition_Data_sim.csv", "BUS29")
    monkeypatch.setattr("src.simulation.m9.final_polish._load_reference_frames", lambda _: {"BUS29": pd.DataFrame({"DATA_PRESENT": [1] * 120 + [0] * 40})})
    summary = recalibrate_cyber_patterns_v2([s1, s2], tmp_path, tmp_path)
    partial = summary["partial_dropout_representation"]
    assert partial.get("research_style", 0.0) > 0.0
    assert partial.get("adversarial_stress", 0.0) > 0.0
    assert summary["pass_flags"]["partial_dropout_coverage_pass"] is True
