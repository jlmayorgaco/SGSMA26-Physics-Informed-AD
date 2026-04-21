from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from src.infrastructure.legacy.m4_adapter import run_type1_fault_simulation
from src.simulation.type1_config import Type1Config


@pytest.mark.integration
@pytest.mark.andes
def test_legacy_m4_pipeline_callable() -> None:
    if os.environ.get("RUN_ANDES_TESTS") != "1":
        pytest.skip("Set RUN_ANDES_TESTS=1 to run ANDES integration tests")
    required = [
        Path("output/M2_RAW0001_NOISE_PROFILE_NEWARCH/dataset_profiles_raw.json"),
        Path("output/ANDES_CALIBRATION_RAW_NEWARCH/current_mapping_selection.csv"),
        Path("output/ANDES_CALIBRATION_RAW_NEWARCH/signal_support_matrix.csv"),
        Path("output/ANDES_CALIBRATION_RAW_NEWARCH/metrics/calibration_results.json"),
    ]
    if any(not p.exists() for p in required):
        pytest.skip("Missing required event0 artifacts for legacy m4 run")

    out_root = Path("output") / "_test_runs" / f"m4_legacy_{uuid4().hex[:8]}"
    run_type1_fault_simulation(
        fault_bus="39",
        output_root=out_root,
        event0_profile_path=required[0],
        event0_current_mapping_csv=required[1],
        event0_support_matrix_csv=required[2],
        event0_calibration_json=required[3],
        generate_plots=False,
    )
    run_dir = out_root / Type1Config(fault_bus="39").run_folder_name()
    assert (run_dir / "run_info.json").exists()
