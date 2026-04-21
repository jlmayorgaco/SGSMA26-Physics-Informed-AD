"""Use case orchestration for m4 type-1 physical fault simulation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.estimation.estimation_outputs import (
    generate_bus_comparison_plots_from_csv,
    generate_signal_family_comparison_plots_from_csv,
)
from src.infrastructure.legacy.m4_adapter import run_type1_fault_simulation
from src.simulation.type1_config import Type1Config, validate_fault_bus


def _require_file(path: str | Path, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing required {label}: {p}")
    return p


def _validate_output_contract(run_dir: Path) -> dict[str, int]:
    sim_dir = run_dir / "simulation"
    est_dir = run_dir / "estimated"
    run_info = run_dir / "run_info.json"
    if not sim_dir.exists():
        raise FileNotFoundError(f"Missing simulation folder: {sim_dir}")
    if not est_dir.exists():
        raise FileNotFoundError(f"Missing estimated folder: {est_dir}")
    if not run_info.exists():
        raise FileNotFoundError(f"Missing run_info.json: {run_info}")
    sim_bus = len(list(sim_dir.glob("BUS*_Competition_Data_nanmask.csv")))
    est_bus = len(list(est_dir.glob("BUS*_Competition_Data_nanmask.csv")))
    if sim_bus == 0 or est_bus == 0:
        raise RuntimeError("Output contract failed: expected BUS*_Competition_Data_nanmask.csv files in simulation/ and estimated/.")
    return {"simulation_bus_csv_count": sim_bus, "estimated_bus_csv_count": est_bus}


def run_simulate_type1_fault_use_case(
    fault_bus: str,
    output_root: str | Path,
    event0_profile_path: str | Path,
    event0_current_mapping_csv: str | Path,
    event0_support_matrix_csv: str | Path,
    event0_calibration_json: str | Path,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Run parity-first m4 type-1 simulation + estimation and return summary."""
    bus = validate_fault_bus(fault_bus)
    output_root_path = Path(output_root)

    profile_path = _require_file(event0_profile_path, "event0 profile JSON")
    mapping_csv = _require_file(event0_current_mapping_csv, "event0 current_mapping_selection.csv")
    support_csv = _require_file(event0_support_matrix_csv, "event0 signal_support_matrix.csv")
    calibration_json = _require_file(event0_calibration_json, "event0 calibration_results.json")

    run_type1_fault_simulation(
        fault_bus=bus,
        output_root=output_root_path,
        event0_profile_path=profile_path,
        event0_current_mapping_csv=mapping_csv,
        event0_support_matrix_csv=support_csv,
        event0_calibration_json=calibration_json,
        generate_plots=generate_plots,
    )

    cfg = Type1Config(fault_bus=bus, base_output_dir=str(output_root_path))
    run_dir = output_root_path / cfg.run_folder_name()
    counts = _validate_output_contract(run_dir)
    comparison_plots = generate_bus_comparison_plots_from_csv(
        run_dir / "estimated" / "estimation_summary_by_bus.csv",
        run_dir / "estimated",
    )
    family_plots = generate_signal_family_comparison_plots_from_csv(
        run_dir / "estimated" / "estimation_metrics_long.csv",
        run_dir / "estimated",
    )
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    return {
        "fault_bus": bus,
        "run_dir": str(run_dir),
        "simulation_dir": str(run_dir / "simulation"),
        "estimated_dir": str(run_dir / "estimated"),
        **counts,
        "comparison_plot_count": len(comparison_plots) + len(family_plots),
        "generated_at_utc": run_info.get("generated_at_utc"),
    }
