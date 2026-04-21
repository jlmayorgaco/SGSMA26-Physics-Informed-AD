from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES
from src.simulation.m9.final_polish import recalibrate_cyber_patterns_v2


def _write_bus_csv(path, bus: str, n: int = 200) -> None:
    cols = {f"{bus}_{s}": np.linspace(0.0, 1.0, n) for s in PMU_MEASUREMENT_SUFFIXES}
    cols["Event"] = np.zeros(n, dtype=int)
    cols["DATA_PRESENT"] = np.ones(n, dtype=int)
    pd.DataFrame(cols).to_csv(path, index=False)


def test_official_style_missing_calibration(tmp_path, monkeypatch) -> None:
    scenario = tmp_path / "data" / "scenarios" / "SIM0001"
    (scenario / "pmu").mkdir(parents=True, exist_ok=True)
    (scenario / "scenario_manifest.json").write_text(json.dumps({"difficulty_level": "easy", "event_coarse": 5}), encoding="utf-8")
    _write_bus_csv(scenario / "pmu" / "Bus29_Competition_Data_sim.csv", "BUS29")
    ref = pd.DataFrame({"DATA_PRESENT": [1] * 170 + [0] * 30})
    monkeypatch.setattr("src.simulation.m9.final_polish._load_reference_frames", lambda _: {"BUS29": ref})
    summary = recalibrate_cyber_patterns_v2([scenario], tmp_path, tmp_path)
    assert (tmp_path / "metadata" / "cyber_calibration_profile_v2.json").exists()
    assert summary["target_official_style_missing_fraction"] is not None
    assert summary["achieved_bus29_missing_fraction"] is not None

