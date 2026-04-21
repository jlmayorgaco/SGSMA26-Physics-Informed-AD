from __future__ import annotations

from src.simulation.m9.hardening import run_m9_1_hardening
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_m9_1_hardening_smoke(tmp_path) -> None:
    report = run_m9_1_hardening(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        n_scenarios=10,
        save_plots=False,
    )
    assert "overall_verdict" in report
    assert (tmp_path / "report" / "m9_1_hardening_report.json").exists()

