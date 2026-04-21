from __future__ import annotations

import json
from pathlib import Path


def test_m10_1_report_schema(tmp_path: Path) -> None:
    payload = {
        "run_metadata": {},
        "split_rebuild": {},
        "preprocessing_ablation": {},
        "cyber_branch_hardening": {},
        "fusion_threshold_hardening": {},
        "calibration_hardening": {},
        "metrics": {"train": {}, "validation": {}, "test": {}, "familywise": {}, "per_scenario_summary": {}},
        "readiness": {"binary_detector_ready_for_classifier_localizer": False, "criteria": {}, "failed_criteria": [], "passed_criteria": [], "verdict": "needs_more_hardening"},
        "overall_conclusion": {"main_improvements": [], "remaining_weaknesses": [], "next_actions": []},
    }
    path = tmp_path / "m10_1_hardening_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    required = {"run_metadata", "split_rebuild", "preprocessing_ablation", "cyber_branch_hardening", "fusion_threshold_hardening", "calibration_hardening", "metrics", "readiness", "overall_conclusion"}
    assert required.issubset(set(loaded.keys()))
