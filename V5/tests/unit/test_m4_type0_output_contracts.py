from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.application.use_cases.simulate_normal_event import _force_event_zero, validate_type0_output_contract


def _clean_dir(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        else:
            p.rmdir()
    path.rmdir()


def test_type0_output_contract_paths_and_reports_exist() -> None:
    root = Path("tests/fixtures/_tmp_m4_type0_contract")
    sim_dir = root / "simulation"
    est_dir = root / "estimated"
    sim_dir.mkdir(parents=True, exist_ok=True)
    est_dir.mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame({"TIMESTAMP": [0.0], "Event": [0], "DATA_PRESENT": [1]}).to_csv(
            sim_dir / "BUS1_Competition_Data_nanmask.csv", index=False
        )
        pd.DataFrame({"TIMESTAMP": [0.0], "Event": [0], "DATA_PRESENT": [1]}).to_csv(
            est_dir / "BUS1_Competition_Data_nanmask.csv", index=False
        )
        pd.DataFrame({"bus_id": ["1"], "rmse": [0.0], "mae": [0.0], "relative_rmse": [0.0], "corr": [1.0]}).to_csv(
            est_dir / "estimation_summary_by_bus.csv", index=False
        )
        pd.DataFrame({"signal": ["VA_MAG"], "rmse": [0.0], "mae": [0.0], "relative_rmse": [0.0], "corr": [1.0]}).to_csv(
            est_dir / "estimation_summary_by_signal.csv", index=False
        )
        pd.DataFrame(
            [{"bus_id": "1", "signal": "VA_MAG", "relative_rmse": 0.0, "corr": 1.0, "rmse": 0.0, "mae": 0.0, "source": "x", "max_abs_error": 0.0}]
        ).to_csv(est_dir / "estimation_metrics_long.csv", index=False)
        (est_dir / "estimation_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        (root / "run_info.json").write_text(json.dumps({"scenario_type": "type0"}), encoding="utf-8")
        out = validate_type0_output_contract(root)
        assert out["simulation_bus_csv_count"] == 1
        assert out["estimated_bus_csv_count"] == 1
    finally:
        _clean_dir(root)


def test_type0_event_column_all_zero_enforcement() -> None:
    df = pd.DataFrame({"Event": [1, 1, 0], "DATA_PRESENT": [0, 1, 1]})
    out = _force_event_zero(df)
    assert out["Event"].tolist() == [0, 0, 0]
    assert out["DATA_PRESENT"].tolist() == [1, 1, 1]
