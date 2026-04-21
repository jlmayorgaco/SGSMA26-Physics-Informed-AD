from __future__ import annotations

import pandas as pd

from tests.helpers.m9_test_utils import generate_single


def test_bad_data_injection(tmp_path) -> None:
    _, clean_dir = generate_single(tmp_path / "clean", "TEMPLATE_EVENT0_NORMAL", scenario_id="SIMC", seed=31)
    _, bad_dir = generate_single(tmp_path / "bad", "TEMPLATE_EVENT7_BAD_DATA", scenario_id="SIMB", seed=31)
    clean = pd.read_csv(clean_dir / "pmu" / "Bus10_Competition_Data_sim.csv")
    bad = pd.read_csv(bad_dir / "pmu" / "Bus10_Competition_Data_sim.csv")
    diff = (bad["BUS10_VA_ANG"] - clean["BUS10_VA_ANG"]).abs().max()
    assert diff > 1.0
    assert 7 in set(bad["Event"].astype(int).unique().tolist())
    assert bad["DATA_PRESENT"].min() == 1
