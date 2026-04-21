from __future__ import annotations

from src.simulation.m9.final_polish import run_m9_2_final_polish
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_m9_2_end_to_end_finalization(tmp_path) -> None:
    report = run_m9_2_final_polish(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        n_scenarios=24,
        batch_mode="balanced_core",
        difficulty_mode="curriculum_easy_to_hard",
        split_strategy="leakage_safe_balanced",
        save_plots=False,
    )
    assert report["overall_verdict"]["m9_2_finalized"] is True

