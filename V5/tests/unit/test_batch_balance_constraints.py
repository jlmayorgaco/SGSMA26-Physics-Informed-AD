from __future__ import annotations

from src.simulation.m9.hardening import generate_balanced_batch
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_batch_balance_constraints(tmp_path) -> None:
    out = generate_balanced_batch(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        batch_mode="balanced_core",
        n_scenarios=12,
        difficulty_mode="uniform",
        save_plots=False,
    )
    assert out["balance"]["scenario_count"] == 12
    assert (tmp_path / "data" / "scenarios" / "dataset_balance_report.json").exists()

