from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from src.application.use_cases.simulate_physical_event import run_simulate_type1_fault_use_case


def _workspace_dir(prefix: str) -> Path:
    root = Path("output") / "_test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _required_event0_paths() -> dict[str, Path]:
    return {
        "profile": Path("output/M2_RAW0001_NOISE_PROFILE_NEWARCH/dataset_profiles_raw.json"),
        "mapping": Path("output/ANDES_CALIBRATION_RAW_NEWARCH/current_mapping_selection.csv"),
        "support": Path("output/ANDES_CALIBRATION_RAW_NEWARCH/signal_support_matrix.csv"),
        "calibration": Path("output/ANDES_CALIBRATION_RAW_NEWARCH/metrics/calibration_results.json"),
    }


@pytest.mark.integration
@pytest.mark.andes
def test_m4_type1_fault_use_case() -> None:
    if os.environ.get("RUN_ANDES_TESTS") != "1":
        pytest.skip("Set RUN_ANDES_TESTS=1 to run ANDES integration tests")
    paths = _required_event0_paths()
    missing = [p for p in paths.values() if not p.exists()]
    if missing:
        pytest.skip(f"Required event0 artifacts missing: {missing}")

    out_root = _workspace_dir("m4_integration")
    summary = run_simulate_type1_fault_use_case(
        fault_bus="39",
        output_root=out_root,
        event0_profile_path=paths["profile"],
        event0_current_mapping_csv=paths["mapping"],
        event0_support_matrix_csv=paths["support"],
        event0_calibration_json=paths["calibration"],
        generate_plots=False,
    )
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "simulation").exists()
    assert (run_dir / "estimated").exists()
    assert (run_dir / "run_info.json").exists()
    assert len(list((run_dir / "simulation").glob("BUS*_Competition_Data_nanmask.csv"))) > 0
    assert len(list((run_dir / "estimated").glob("BUS*_Competition_Data_nanmask.csv"))) > 0
