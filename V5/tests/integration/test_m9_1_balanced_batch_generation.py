from __future__ import annotations

from src.simulation.m9.hardening import generate_balanced_batch
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_m9_1_balanced_batch_generation(tmp_path) -> None:
    out = generate_balanced_batch(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        batch_mode="cyber_heavy",
        n_scenarios=16,
        difficulty_mode="uniform",
        save_plots=False,
    )
    assert out["balance"]["scenario_count"] == 16
    assert (tmp_path / "data" / "scenarios" / "scenario_registry.json").exists()

