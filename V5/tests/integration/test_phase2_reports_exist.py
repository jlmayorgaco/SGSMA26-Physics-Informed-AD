from __future__ import annotations

from pathlib import Path

from src.detectors.training.reports.phase2_reporting import write_phase2_report


def test_phase2_reports_exist(tmp_path: Path) -> None:
    payload = {
        "repo_alignment": {"ok": True},
        "implemented_components": {"cyber_branch": {}, "physical_branch": {}, "fusion": {}, "postprocessing": {}, "pipelines": {}},
        "artifacts": {},
        "tests": {"unit": {"status": "pass"}, "integration": {"status": "pass"}, "e2e": {"status": "pass"}},
        "initial_metrics": {"f1_abnormal": 0.7},
        "known_gaps_for_phase_3": [],
        "overall_verdict": {"phase_2_complete": True, "ready_for_phase_3": True, "main_risks": [], "next_phase": "phase_3_training_validation_hardening"},
    }
    j, m = write_phase2_report(payload, tmp_path / "report")
    assert j.exists()
    assert m.exists()

