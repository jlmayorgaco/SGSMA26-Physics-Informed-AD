from __future__ import annotations

import pandas as pd

from tests.helpers.m9_test_utils import generate_single


def test_full_state_target_alignment(tmp_path) -> None:
    _, scenario_dir = generate_single(tmp_path, "TEMPLATE_EVENT3_GENERATION_CHANGE")
    truth = pd.read_csv(scenario_dir / "all_buses" / "all_bus_truth.csv")
    target = pd.read_csv(scenario_dir / "all_buses" / "full_state_target.csv")
    assert len(truth) == len(target)
    assert truth[["TIMESTAMP", "BUS"]].equals(target[["TIMESTAMP", "BUS"]])
    assert set(target["EVENT"].unique().tolist()) <= set(truth["EVENT"].unique().tolist())
