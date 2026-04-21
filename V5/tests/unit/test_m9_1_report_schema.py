from __future__ import annotations

from src.simulation.m9.hardening import run_m9_1_hardening
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_m9_1_report_schema(tmp_path) -> None:
    report = run_m9_1_hardening(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        n_scenarios=8,
        save_plots=False,
    )
    required = [
        "run_metadata",
        "validator_consistency",
        "realism_validation_v2",
        "angular_realism",
        "cyber_pattern_calibration",
        "balanced_batch_factory",
        "estimator_in_the_loop_scoring",
        "split_generation",
        "downstream_training_readiness",
        "overall_verdict",
    ]
    for k in required:
        assert k in report

