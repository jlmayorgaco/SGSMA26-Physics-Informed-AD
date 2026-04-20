from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.calibration.calibration_outputs import write_outputs
from src.calibration.signal_specs import build_signal_specs


def _workspace_dir(prefix: str) -> Path:
    root = Path("output") / "_test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _records() -> list[dict]:
    return [
        {
            "bus_id": "10",
            "signal_key": "VA_mag",
            "status": "supported",
            "scope": "aggregate",
            "signal_family": "voltage_mag",
            "ks_stat": 0.05,
            "composite_score": 0.1,
            "relative_mean_error": 0.1,
            "relative_std_error": 0.1,
            "psd_lowfreq_mismatch": 0.1,
            "spectral_centroid_error": 0.1,
            "chunk_stability_penalty": 0.0,
            "quality_bucket": "usable",
        }
    ]


def test_write_outputs_creates_expected_files() -> None:
    out = _workspace_dir("m3_outputs")
    write_outputs(_records(), out, 0, "fft_residual_synthesis", {"supported", "fallback_profile"}, build_signal_specs())
    assert (out / "metrics" / "calibration_metrics_long.csv").exists()
    assert (out / "signal_support_matrix.csv").exists()


def test_calibration_results_json_contains_generated_at_utc_event_label_drift_mode_records() -> None:
    out = _workspace_dir("m3_outputs")
    write_outputs(_records(), out, 0, "fft_residual_synthesis", {"supported", "fallback_profile"}, build_signal_specs())
    payload = json.loads((out / "metrics" / "calibration_results.json").read_text(encoding="utf-8"))
    assert {"generated_at_utc", "event_label", "drift_mode", "records"}.issubset(payload.keys())


def test_support_matrix_written() -> None:
    out = _workspace_dir("m3_outputs")
    write_outputs(_records(), out, 0, "fft_residual_synthesis", {"supported", "fallback_profile"}, build_signal_specs())
    df = pd.read_csv(out / "signal_support_matrix.csv")
    assert not df.empty


def test_summary_csvs_exist() -> None:
    out = _workspace_dir("m3_outputs")
    write_outputs(_records(), out, 0, "fft_residual_synthesis", {"supported", "fallback_profile"}, build_signal_specs())
    assert (out / "metrics" / "calibration_summary_by_bus.csv").exists()
    assert (out / "metrics" / "calibration_summary_by_signal.csv").exists()
    assert (out / "metrics" / "calibration_summary_chunk0.csv").exists()
    assert (out / "metrics" / "calibration_summary_signal_family.csv").exists()
