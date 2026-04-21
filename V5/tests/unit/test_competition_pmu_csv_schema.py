from __future__ import annotations

import pandas as pd

from src.simulation.m9.constants import PMU_BUSES_OFFICIAL, PMU_MEASUREMENT_SUFFIXES
from tests.helpers.m9_test_utils import generate_single


def test_competition_pmu_csv_schema(tmp_path) -> None:
    _, scenario_dir = generate_single(tmp_path, "TEMPLATE_EVENT1_FAULT")
    pmu_dir = scenario_dir / "pmu"
    for bus in PMU_BUSES_OFFICIAL:
        token = int("".join(ch for ch in bus if ch.isdigit()))
        path = pmu_dir / f"Bus{token}_Competition_Data_sim.csv"
        assert path.exists()
        df = pd.read_csv(path)
        expected = ["TIMESTAMP"] + [f"{bus}_{s}" for s in PMU_MEASUREMENT_SUFFIXES] + ["DATA_PRESENT", "Event"]
        assert list(df.columns) == expected
        assert len(df) > 10
