from __future__ import annotations

from src.simulation.m9.final_polish import run_angular_realism_v2, run_freq_rocof_coherence_v2
from src.simulation.m9.hardening import generate_balanced_batch
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_m9_2_angle_and_coherence_pipeline(tmp_path) -> None:
    batch = generate_balanced_batch(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        n_scenarios=10,
        batch_mode="balanced_core",
        difficulty_mode="curriculum_easy_to_hard",
        save_plots=False,
    )
    angular = run_angular_realism_v2(batch["scenario_dirs"], REFERENCE_PMU_DIR, tmp_path)
    freq = run_freq_rocof_coherence_v2(batch["scenario_dirs"], REFERENCE_PMU_DIR, tmp_path)
    assert "overall_angular_realism_pass" in angular
    assert "overall_status" in freq
    assert (tmp_path / "metrics" / "angular_realism_v2.csv").exists()
    assert (tmp_path / "metrics" / "freq_rocof_coherence_metrics.csv").exists()

