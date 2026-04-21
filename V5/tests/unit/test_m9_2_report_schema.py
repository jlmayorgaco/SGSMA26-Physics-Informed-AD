from __future__ import annotations

from src.simulation.m9.final_polish import run_m9_2_final_polish
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_m9_2_report_schema(tmp_path) -> None:
    report = run_m9_2_final_polish(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        n_scenarios=12,
        save_plots=False,
    )
    required = [
        "run_metadata",
        "angular_realism_v2",
        "freq_rocof_coherence_v2",
        "cyber_calibration_v2",
        "split_generation_v2",
        "downstream_training_readiness",
        "overall_verdict",
    ]
    for key in required:
        assert key in report

