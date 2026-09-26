from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.simulation.comparison_reports import compare_runs


def _clean_dir(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        else:
            p.rmdir()
    path.rmdir()


def _write_run(root: Path, scale: float) -> None:
    est = root / "estimated"
    est.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "bus_id": ["2", "10"],
            "rmse": [0.1 * scale, 0.2 * scale],
            "mae": [0.1 * scale, 0.2 * scale],
            "relative_rmse": [0.1 * scale, 0.2 * scale],
            "corr": [0.9, 0.8],
        }
    ).to_csv(est / "estimation_summary_by_bus.csv", index=False)
    pd.DataFrame(
        {
            "signal": ["VA_MAG", "IA_MAG", "Freq", "ROCOF"],
            "rmse": [0.1 * scale, 0.2 * scale, 0.05 * scale, 0.03 * scale],
            "mae": [0.1 * scale, 0.2 * scale, 0.05 * scale, 0.03 * scale],
            "relative_rmse": [0.1 * scale, 0.2 * scale, 0.05 * scale, 0.03 * scale],
            "corr": [0.9, 0.8, 0.95, 0.9],
        }
    ).to_csv(est / "estimation_summary_by_signal.csv", index=False)
    pd.DataFrame(
        [
            {"bus_id": "2", "signal": "VA_MAG", "relative_rmse": 0.1 * scale, "corr": 0.9, "rmse": 0.1, "mae": 0.1, "source": "pmu_passthrough", "max_abs_error": 0.1},
            {"bus_id": "2", "signal": "IA_MAG", "relative_rmse": 0.2 * scale, "corr": 0.9, "rmse": 0.1, "mae": 0.1, "source": "pmu_passthrough", "max_abs_error": 0.1},
            {"bus_id": "2", "signal": "Freq", "relative_rmse": 0.05 * scale, "corr": 0.9, "rmse": 0.1, "mae": 0.1, "source": "pmu_passthrough", "max_abs_error": 0.1},
            {"bus_id": "2", "signal": "ROCOF", "relative_rmse": 0.03 * scale, "corr": 0.9, "rmse": 0.1, "mae": 0.1, "source": "pmu_passthrough", "max_abs_error": 0.1},
            {"bus_id": "10", "signal": "VA_MAG", "relative_rmse": 0.2 * scale, "corr": 0.8, "rmse": 0.2, "mae": 0.2, "source": "ybus_estimated", "max_abs_error": 0.2},
            {"bus_id": "10", "signal": "IA_MAG", "relative_rmse": 0.3 * scale, "corr": 0.8, "rmse": 0.2, "mae": 0.2, "source": "ybus_estimated", "max_abs_error": 0.2},
            {"bus_id": "10", "signal": "Freq", "relative_rmse": 0.06 * scale, "corr": 0.8, "rmse": 0.2, "mae": 0.2, "source": "ybus_estimated", "max_abs_error": 0.2},
            {"bus_id": "10", "signal": "ROCOF", "relative_rmse": 0.04 * scale, "corr": 0.8, "rmse": 0.2, "mae": 0.2, "source": "ybus_estimated", "max_abs_error": 0.2},
        ]
    ).to_csv(est / "estimation_metrics_long.csv", index=False)


def test_compare_runs_generates_summaries_and_plots() -> None:
    root = Path("tests/fixtures/_tmp_m4_compare")
    run_a = root / "type0"
    run_b = root / "type1"
    out = root / "comparison"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)
    try:
        _write_run(run_a, scale=1.0)
        _write_run(run_b, scale=1.2)
        summary = compare_runs(run_a, run_b, out)
        assert summary["bus_rows"] > 0
        assert (out / "comparison_summary_by_bus.csv").exists()
        assert (out / "comparison_summary_by_signal.csv").exists()
        assert (out / "comparison_voltage_mag_percent_by_bus.png").exists()
        assert (out / "comparison_current_mag_percent_by_bus.png").exists()
        assert (out / "comparison_frequency_percent_by_bus.png").exists()
        assert (out / "comparison_rocof_percent_by_bus.png").exists()
        assert (out / "estimation_bus_quality_percent.png").exists()
        assert (out / "estimation_bus_relative_rmse_percent.png").exists()
    finally:
        _clean_dir(root)
