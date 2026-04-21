from __future__ import annotations

import pandas as pd

from tests.helpers.m9_test_utils import generate_single


def test_official_style_multi_event(tmp_path) -> None:
    _, scenario_dir = generate_single(tmp_path, "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT", scenario_id="SIM0001")
    bus29 = pd.read_csv(scenario_dir / "pmu" / "Bus29_Competition_Data_sim.csv")
    bus39 = pd.read_csv(scenario_dir / "pmu" / "Bus39_Competition_Data_sim.csv")
    labels29 = set(bus29["Event"].astype(int).unique().tolist())
    labels39 = set(bus39["Event"].astype(int).unique().tolist())
    assert {5, 6}.issubset(labels29)
    assert {1, 2, 3, 4}.issubset(labels39)
    assert (bus29["DATA_PRESENT"].astype(int) == 0).any()
