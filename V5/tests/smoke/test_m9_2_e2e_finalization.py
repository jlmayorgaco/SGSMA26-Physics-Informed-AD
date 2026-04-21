from __future__ import annotations

import json

from src.simulation.m9.final_polish import run_m9_2_final_polish
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_m9_2_e2e_finalization(tmp_path) -> None:
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
    payload = json.loads((tmp_path / "report" / "m9_2_final_polish_report.json").read_text(encoding="utf-8"))
    assert (tmp_path / "metadata" / "angular_realism_v2_summary.json").exists()
    assert (tmp_path / "metadata" / "freq_rocof_coherence_summary.json").exists()
    assert payload["freq_rocof_coherence_v2"]["overall_status"] != "pass" or payload["freq_rocof_coherence_v2"]["counts"]["not_computable"] == 0
    assert payload["cyber_calibration_v2"]["pass_flags"]["global_missing_fraction_pass"] is True
    assert payload["split_generation_v2"]["no_scenario_leakage"] is True
    assert payload["split_generation_v2"]["acceptable_for_model_selection"] is True
    assert report["overall_verdict"]["m9_2_finalized"] is True

