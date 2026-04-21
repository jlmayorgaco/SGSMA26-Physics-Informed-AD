from __future__ import annotations

from src.simulation.m9.hardening import generate_balanced_batch, run_estimator_scoring
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_m9_1_estimator_scoring_pipeline(tmp_path) -> None:
    out = generate_balanced_batch(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        batch_mode="estimator_stress",
        n_scenarios=10,
        difficulty_mode="uniform",
        save_plots=False,
    )
    score = run_estimator_scoring(out["scenario_dirs"], tmp_path)
    assert score["scenario_count"] == 10
    assert (tmp_path / "metrics" / "scenario_scoring_table.csv").exists()

