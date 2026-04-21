from __future__ import annotations

import pandas as pd

from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES
from tests.helpers.m9_test_utils import generate_single


def test_missing_data_semantics(tmp_path) -> None:
    _, scenario_dir = generate_single(tmp_path, "TEMPLATE_EVENT5_MISSING_ONLY")
    path = scenario_dir / "pmu" / "Bus29_Competition_Data_sim.csv"
    df = pd.read_csv(path)
    cols = [f"BUS29_{s}" for s in PMU_MEASUREMENT_SUFFIXES]
    missing = pd.to_numeric(df["DATA_PRESENT"], errors="coerce").fillna(1).astype(int) == 0
    assert missing.any()
    assert df.loc[missing, cols].isna().all(axis=None)
    assert set(df.loc[missing, "Event"].astype(int).unique().tolist()) == {5}
