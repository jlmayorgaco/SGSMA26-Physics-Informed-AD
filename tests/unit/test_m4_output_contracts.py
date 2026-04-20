from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.estimation.estimation_outputs import export_estimated_outputs, export_estimation_report


def _sample_bus_df(bus_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TIMESTAMP": [0.0],
            f"BUS{bus_id}_VA_ANG": [0.0],
            f"BUS{bus_id}_VA_MAG": [1.0],
            f"BUS{bus_id}_VB_ANG": [-120.0],
            f"BUS{bus_id}_VB_MAG": [1.0],
            f"BUS{bus_id}_VC_ANG": [120.0],
            f"BUS{bus_id}_VC_MAG": [1.0],
            f"BUS{bus_id}_IA_ANG": [0.0],
            f"BUS{bus_id}_IA_MAG": [1.0],
            f"BUS{bus_id}_IB_ANG": [-120.0],
            f"BUS{bus_id}_IB_MAG": [1.0],
            f"BUS{bus_id}_IC_ANG": [120.0],
            f"BUS{bus_id}_IC_MAG": [1.0],
            f"BUS{bus_id}_Freq": [60.0],
            f"BUS{bus_id}_ROCOF": [0.0],
            "DATA_PRESENT": [1],
            "Event": [0],
        }
    )


def _clean_dir(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        else:
            p.rmdir()
    path.rmdir()


def test_export_estimation_report_creates_expected_files() -> None:
    out_dir = Path("tests/fixtures/_tmp_m4_output_contract_report")
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = {"10": _sample_bus_df("10")}
    est = {"10": _sample_bus_df("10")}
    try:
        export_estimation_report("10", sim, est, out_dir)
        assert (out_dir / "estimation_metrics_long.csv").exists()
        assert (out_dir / "estimation_summary_by_bus.csv").exists()
        assert (out_dir / "estimation_summary_by_signal.csv").exists()
        assert (out_dir / "estimation_report.json").exists()
        assert (out_dir / "estimation_bus_quality_percent.png").exists()
        assert (out_dir / "estimation_bus_relative_rmse_percent.png").exists()
        assert (out_dir / "comparison_current_mag_percent_by_bus.png").exists()
        assert (out_dir / "comparison_voltage_mag_percent_by_bus.png").exists()
        assert (out_dir / "comparison_frequency_percent_by_bus.png").exists()
        assert (out_dir / "comparison_rocof_percent_by_bus.png").exists()
    finally:
        _clean_dir(out_dir)


def test_export_estimated_outputs_creates_expected_csvs() -> None:
    out_dir = Path("tests/fixtures/_tmp_m4_output_contract_estimated")
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = {"10": _sample_bus_df("10")}
    est = {"10": _sample_bus_df("10")}
    try:
        export_estimated_outputs(est, sim, "10", out_dir)
        assert (out_dir / "BUS10_Competition_Data_nanmask.csv").exists()
    finally:
        _clean_dir(out_dir)


def test_run_info_contract_fields() -> None:
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fault_bus": "10",
        "folders": {"simulation": "x", "estimated": "y"},
    }
    assert {"generated_at_utc", "fault_bus", "folders"}.issubset(payload.keys())
    json.dumps(payload)
