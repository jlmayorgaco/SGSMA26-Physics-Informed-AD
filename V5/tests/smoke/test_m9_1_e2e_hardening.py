from __future__ import annotations

import json

from src.simulation.m9.hardening import run_m9_1_hardening
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_m9_1_e2e_hardening(tmp_path) -> None:
    report = run_m9_1_hardening(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        batch_mode="balanced_core",
        n_scenarios=24,
        difficulty_mode="curriculum_easy_to_hard",
        run_estimator_scoring_flag=True,
        build_splits_flag=True,
        save_plots=False,
    )
    hardening = json.loads((tmp_path / "report" / "m9_1_hardening_report.json").read_text(encoding="utf-8"))
    assert (tmp_path / "metrics" / "raw_vs_sim_channel_metrics.csv").exists()
    assert (tmp_path / "metrics" / "angular_realism_metrics.csv").exists()
    assert (tmp_path / "metrics" / "scenario_scoring_table.csv").exists()
    assert (tmp_path / "split_manifest.json").exists()
    assert hardening["overall_verdict"]["m9_1_hardened"] == report["overall_verdict"]["m9_1_hardened"]

