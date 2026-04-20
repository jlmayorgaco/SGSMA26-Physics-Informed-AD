"""Use case for m3 RAW event-0 calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.calibration.event0_calibrator import run_event0_calibration_core


def run_calibrate_event0_use_case(
    raw_chunks_dir: str | Path,
    raw_profile_file: str | Path,
    output_dir: str | Path,
    event_label: int = 0,
    drift_mode: str = "fft_residual_synthesis",
) -> dict[str, Any]:
    """Run end-to-end event-0 calibration and return a summary payload."""
    output_dir = Path(output_dir)
    run_event0_calibration_core(
        raw_chunks_dir=raw_chunks_dir,
        raw_profile_file=raw_profile_file,
        output_dir=output_dir,
        event_label=event_label,
        drift_mode=drift_mode,
    )

    metrics_dir = output_dir / "metrics"
    results_path = metrics_dir / "calibration_results.json"
    rows = 0
    evaluated_rows = 0
    active_statuses = {"supported", "fallback_profile"}
    status_counts: dict[str, int] = {}
    if results_path.exists():
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        rows = len(records)
        for rec in records:
            status = str(rec.get("status", ""))
            status_counts[status] = status_counts.get(status, 0) + 1
        evaluated_rows = sum(status_counts.get(s, 0) for s in active_statuses)

    return {
        "output_dir": str(output_dir),
        "metrics_dir": str(metrics_dir),
        "event_label": int(event_label),
        "drift_mode": drift_mode,
        "record_count": rows,
        "evaluated_row_count": evaluated_rows,
        "status_counts": status_counts,
    }
