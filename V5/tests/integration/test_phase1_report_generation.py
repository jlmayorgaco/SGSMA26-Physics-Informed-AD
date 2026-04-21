from __future__ import annotations

from pathlib import Path

from src.detectors.training.reports.phase1_reporting import write_phase1_foundations_report


def test_phase1_report_generation(tmp_path: Path) -> None:
    payload = {
        "repo_alignment": {"ok": True},
        "created_or_modified_files": ["a.py"],
        "domain_contracts": {"count": 6},
        "shared_preprocessing": {"modules": 7},
        "dataset_foundations": {"ready": True},
        "label_mapping": {"binary": "event0_vs_event1_8"},
        "tests": {"unit": {"count": 1}, "integration": {"count": 1}},
        "known_gaps_for_phase_2": ["implement branches"],
        "overall_verdict": {
            "phase_1_complete": True,
            "ready_for_phase_2": True,
            "main_risks": [],
            "next_phase": "phase_2_branches_and_fusion",
        },
    }
    json_path, md_path = write_phase1_foundations_report(payload, Path(tmp_path) / "report")
    assert json_path.exists()
    assert md_path.exists()

