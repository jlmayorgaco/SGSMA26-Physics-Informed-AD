from __future__ import annotations

from pathlib import Path

from poc.schema import measurement_columns
from poc.synthesis.scenario_grid import build_scenario_grid


def test_generated_fault_csv_loads_with_src_loader():
    from src.io.load_csv import load_all

    scenario_dir = Path("poc/data_synth/faults/fault_smoke_0000")
    assert scenario_dir.exists()
    df = load_all(scenario_dir)
    assert len(df) > 0
    assert set(measurement_columns()).issubset(df.columns)
    assert 1 in set(df["Event"].astype(int).tolist())


def test_scenario_grid_target_size_and_labels():
    scenarios = build_scenario_grid()
    assert len(scenarios) == 1000
    labels = [scenario.label for scenario in scenarios]
    assert labels.count(1) == 200
    assert labels.count(2) == 100
    assert labels.count(3) == 200
    assert labels.count(4) == 200
    assert labels.count(5) == 100
    assert labels.count(6) == 100
    assert labels.count(7) == 100

