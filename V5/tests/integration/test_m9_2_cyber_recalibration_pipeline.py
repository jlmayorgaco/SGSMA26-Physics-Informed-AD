from __future__ import annotations

from src.simulation.m9.final_polish import recalibrate_cyber_patterns_v2
from src.simulation.m9.hardening import generate_balanced_batch
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_m9_2_cyber_recalibration_pipeline(tmp_path) -> None:
    batch = generate_balanced_batch(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        n_scenarios=8,
        batch_mode="balanced_core",
        difficulty_mode="curriculum_easy_to_hard",
        save_plots=False,
    )
    summary = recalibrate_cyber_patterns_v2(batch["scenario_dirs"], REFERENCE_PMU_DIR, tmp_path)
    assert (tmp_path / "metadata" / "cyber_calibration_profile_v2.json").exists()
    assert "pass_flags" in summary

