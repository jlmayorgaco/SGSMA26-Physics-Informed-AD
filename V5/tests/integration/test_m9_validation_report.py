from __future__ import annotations

import json

from tests.helpers.m9_test_utils import generate_validation_suite


def test_m9_validation_report(tmp_path) -> None:
    report = generate_validation_suite(tmp_path)
    assert "scenario_checks" in report
    assert "downstream_readiness" in report
    assert "overall_verdict" in report
    assert "OFFICIAL_STYLE_MULTI_EVENT" in report["scenario_checks"]
    json_path = tmp_path / "report" / "m9_simulator_validation.json"
    md_path = tmp_path / "report" / "m9_simulator_validation.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "classifier_training_ready" in payload["downstream_readiness"]
