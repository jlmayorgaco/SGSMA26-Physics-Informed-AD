"""Use case orchestration for m4 type0 (normal/no-fault) simulation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.estimation.estimation_outputs import export_estimated_outputs
from src.estimation.temporal_regularized_estimator import build_estimated_bus_dataframes
from src.simulation.andes_normal_runtime import run_normal_simulation_andes
from src.simulation.competition_export import build_competition_dataframe_for_bus
from src.simulation.event0_artifacts import load_event0_artifacts
from src.simulation.network_extraction import extract_all_bus_signals
from src.simulation.scenario_labels import event_series_type0
from src.simulation.simulation_plots import generate_bus_plots
from src.simulation.type0_config import Type0Config


def _require_file(path: str | Path, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing required {label}: {p}")
    return p


def _force_event_zero(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Event" in out.columns:
        out["Event"] = 0
    if "DATA_PRESENT" in out.columns:
        out["DATA_PRESENT"] = 1
    return out


def _export_type0_simulation_outputs(
    run_dir: Path,
    signals_by_bus: dict[str, dict],
    artifacts: dict,
    generate_plots: bool,
) -> dict[str, pd.DataFrame]:
    sim_dir = run_dir / "simulation"
    sim_dir.mkdir(parents=True, exist_ok=True)
    simulation_dfs: dict[str, pd.DataFrame] = {}

    for bus_id in sorted(signals_by_bus.keys(), key=lambda x: int(x)):
        rng = np.random.default_rng(20260418 + int(bus_id))
        df, clean_map, noisy_map = build_competition_dataframe_for_bus(str(bus_id), signals_by_bus[bus_id], artifacts, rng)
        df = _force_event_zero(df)
        simulation_dfs[bus_id] = df.copy()
        df.to_csv(sim_dir / f"BUS{bus_id}_Competition_Data_nanmask.csv", index=False)

        if generate_plots and "TIMESTAMP" in df.columns:
            ev = event_series_type0(df["TIMESTAMP"].to_numpy(dtype=float))
            generate_bus_plots(
                bus_id=str(bus_id),
                t=df["TIMESTAMP"].to_numpy(dtype=float),
                event_arr=ev,
                clean_map=clean_map,
                noisy_map=noisy_map,
                base_out_dir=str(sim_dir),
            )

    if not simulation_dfs:
        raise RuntimeError("Type0 simulation produced zero bus outputs.")
    return simulation_dfs


def validate_type0_output_contract(run_dir: Path) -> dict[str, int]:
    sim_dir = run_dir / "simulation"
    est_dir = run_dir / "estimated"
    run_info = run_dir / "run_info.json"
    if not sim_dir.exists():
        raise FileNotFoundError(f"Missing simulation folder: {sim_dir}")
    if not est_dir.exists():
        raise FileNotFoundError(f"Missing estimated folder: {est_dir}")
    if not run_info.exists():
        raise FileNotFoundError(f"Missing run_info.json: {run_info}")
    required_est = [
        est_dir / "estimation_metrics_long.csv",
        est_dir / "estimation_summary_by_bus.csv",
        est_dir / "estimation_summary_by_signal.csv",
        est_dir / "estimation_report.json",
    ]
    for p in required_est:
        if not p.exists():
            raise FileNotFoundError(f"Missing estimated artifact: {p}")
    sim_bus = len(list(sim_dir.glob("BUS*_Competition_Data_nanmask.csv")))
    est_bus = len(list(est_dir.glob("BUS*_Competition_Data_nanmask.csv")))
    if sim_bus == 0 or est_bus == 0:
        raise RuntimeError("Type0 contract failed: simulation/estimated BUS csv count is zero.")
    return {"simulation_bus_csv_count": sim_bus, "estimated_bus_csv_count": est_bus}


def build_type0_run_info(
    cfg: Type0Config,
    run_dir: Path,
    profile_path: Path,
    mapping_csv: Path,
    support_csv: Path,
    calibration_json: Path,
) -> dict[str, Any]:
    """Build deterministic run-info payload for type0 run folder."""
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_type": "type0",
        "fault_bus": None,
        "sim_tf": cfg.sim_tf,
        "sim_tstep": cfg.sim_tstep,
        "event_label_mode": cfg.event_label_mode,
        "data_present_default": cfg.data_present_default,
        "folders": {
            "root": str(run_dir.resolve()),
            "simulation": str((run_dir / "simulation").resolve()),
            "estimated": str((run_dir / "estimated").resolve()),
        },
        "event0_artifacts": {
            "profile_path": str(profile_path),
            "current_mapping_csv": str(mapping_csv),
            "support_matrix_csv": str(support_csv),
            "calibration_json": str(calibration_json),
        },
    }


def run_simulate_type0_use_case(
    output_root: str | Path,
    event0_profile_path: str | Path,
    event0_current_mapping_csv: str | Path,
    event0_support_matrix_csv: str | Path,
    event0_calibration_json: str | Path,
    generate_plots: bool = True,
    sim_tf: float = 10.1,
    sim_tstep: float = 1.0 / 30.0,
) -> dict[str, Any]:
    """Run type0 normal scenario with shared m4 architecture and export artifacts."""
    output_root_path = Path(output_root)
    cfg = Type0Config(base_output_dir=str(output_root_path), sim_tf=sim_tf, sim_tstep=sim_tstep)
    run_dir = output_root_path / cfg.run_folder_name()
    run_dir.mkdir(parents=True, exist_ok=True)

    profile_path = _require_file(event0_profile_path, "event0 profile JSON")
    mapping_csv = _require_file(event0_current_mapping_csv, "event0 current_mapping_selection.csv")
    support_csv = _require_file(event0_support_matrix_csv, "event0 signal_support_matrix.csv")
    calibration_json = _require_file(event0_calibration_json, "event0 calibration_results.json")

    artifacts = load_event0_artifacts(profile_path, mapping_csv, support_csv, calibration_json)
    system = run_normal_simulation_andes(tf=cfg.sim_tf, tstep=cfg.sim_tstep)
    signals_by_bus, meta = extract_all_bus_signals(system, "0")
    simulation_dfs = _export_type0_simulation_outputs(run_dir, signals_by_bus, artifacts, generate_plots=generate_plots)

    estimated_dfs = build_estimated_bus_dataframes(simulation_dfs, meta, "0")
    estimated_dfs = {k: _force_event_zero(v) for k, v in estimated_dfs.items()}
    export_estimated_outputs(
        estimated_dfs=estimated_dfs,
        simulation_dfs=simulation_dfs,
        fault_bus="0",
        output_dir=run_dir / "estimated",
        generate_plots=generate_plots,
    )

    run_info = build_type0_run_info(cfg, run_dir, profile_path, mapping_csv, support_csv, calibration_json)
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    counts = validate_type0_output_contract(run_dir)
    return {
        "run_dir": str(run_dir),
        "simulation_dir": str(run_dir / "simulation"),
        "estimated_dir": str(run_dir / "estimated"),
        **counts,
        "scenario_type": "type0",
        "generated_at_utc": run_info["generated_at_utc"],
    }
