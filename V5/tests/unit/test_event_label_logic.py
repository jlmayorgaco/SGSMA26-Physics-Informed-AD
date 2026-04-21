from __future__ import annotations

import pandas as pd

from tests.helpers.m9_test_utils import generate_single


def test_event_label_logic(tmp_path) -> None:
    scenarios = [
        ("TEMPLATE_EVENT0_NORMAL", {0}),
        ("TEMPLATE_EVENT1_FAULT", {1}),
        ("TEMPLATE_EVENT2_LINE_OUTAGE", {2}),
        ("TEMPLATE_EVENT3_GENERATION_CHANGE", {3}),
        ("TEMPLATE_EVENT4_LOAD_CHANGE", {4}),
        ("TEMPLATE_EVENT5_MISSING_ONLY", {5}),
        ("TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL", {3, 6}),
        ("TEMPLATE_EVENT7_BAD_DATA", {7}),
        ("TEMPLATE_EVENT8_UNKNOWN_COMPOSITE", {8}),
    ]
    for i, (template, expected) in enumerate(scenarios, start=1):
        _, scenario_dir = generate_single(tmp_path / f"s{i}", template, scenario_id=f"SIM{i:04d}", seed=20 + i)
        if template in {"TEMPLATE_EVENT5_MISSING_ONLY", "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL"}:
            bus_path = "Bus29_Competition_Data_sim.csv"
        elif template == "TEMPLATE_EVENT7_BAD_DATA":
            bus_path = "Bus10_Competition_Data_sim.csv"
        else:
            bus_path = "Bus39_Competition_Data_sim.csv"
        df = pd.read_csv(scenario_dir / "pmu" / bus_path)
        labels = set(pd.to_numeric(df["Event"], errors="coerce").dropna().astype(int).unique().tolist())
        assert expected.issubset(labels)
