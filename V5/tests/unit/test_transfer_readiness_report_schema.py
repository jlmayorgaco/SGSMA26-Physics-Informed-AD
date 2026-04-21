from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_transfer_readiness_report_schema() -> None:
    report_path = Path("output/M2_RAW_INFORMED_CYBER/report/transfer_readiness_report.json")
    if not report_path.exists():
        pytest.skip("transfer_readiness_report.json not generated yet")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert {"raw_gap_addressed", "event5_match_quality", "event7_match_quality", "detector_raw_holdout_before", "detector_raw_holdout_after", "verdict"}.issubset(
        payload.keys()
    )
    verdict = payload["verdict"]
    assert {"cyber_layer_realistic_enough_for_training", "raw_transfer_improved", "usable_for_detector_training", "main_remaining_gaps", "next_actions"}.issubset(verdict.keys())

