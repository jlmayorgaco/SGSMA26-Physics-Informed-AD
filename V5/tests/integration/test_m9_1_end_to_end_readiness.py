from __future__ import annotations

import json

from src.simulation.m9.hardening import run_m9_1_hardening
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_m9_1_end_to_end_readiness(tmp_path) -> None:
    run_m9_1_hardening(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        n_scenarios=20,
        batch_mode="balanced_core",
        difficulty_mode="curriculum_easy_to_hard",
        save_plots=False,
    )
    hardening = json.loads((tmp_path / "report" / "m9_1_hardening_report.json").read_text(encoding="utf-8"))
    balance = json.loads((tmp_path / "data" / "scenarios" / "dataset_balance_report.json").read_text(encoding="utf-8"))
    split = json.loads((tmp_path / "split_balance_report.json").read_text(encoding="utf-8"))
    assert hardening["validator_consistency"]["aggregates"]["overall_readiness"] == hardening["overall_verdict"]["m9_1_hardened"]
    assert hardening["realism_validation_v2"]["summary"]
    assert hardening["angular_realism"]["rows"] > 0
    assert balance["scenario_count"] >= 20
    assert split["no_scenario_leakage"]
    assert split["no_near_duplicate_family_leakage"]

