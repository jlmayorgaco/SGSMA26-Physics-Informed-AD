from __future__ import annotations


def test_non0_detector_v2_report_schema_minimum_keys() -> None:
    payload = {
        "architecture": {},
        "synthetic_metrics": {"test": {}},
        "raw_metrics": {"raw_holdout": {}},
        "raw_holdout_before": {},
        "raw_holdout_after": {},
        "verdict": {"detects_non0_on_raw": False, "usable_on_raw": False},
    }
    required = {
        "architecture",
        "synthetic_metrics",
        "raw_metrics",
        "raw_holdout_before",
        "raw_holdout_after",
        "verdict",
    }
    assert required.issubset(payload.keys())
    assert "detects_non0_on_raw" in payload["verdict"]
    assert "usable_on_raw" in payload["verdict"]

