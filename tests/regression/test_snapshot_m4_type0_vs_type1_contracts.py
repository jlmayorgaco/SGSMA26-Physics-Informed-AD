from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest

from src.application.use_cases.simulate_normal_event import run_simulate_type0_use_case
from src.application.use_cases.simulate_physical_event import run_simulate_type1_fault_use_case


@pytest.mark.regression
@pytest.mark.andes
def test_type0_vs_type1_contract_shapes() -> None:
    if os.environ.get("RUN_ANDES_TESTS") != "1":
        pytest.skip("Set RUN_ANDES_TESTS=1 to run ANDES regression tests")

    paths = {
        "profile": Path("output/M2_RAW0001_NOISE_PROFILE_NEWARCH/dataset_profiles_raw.json"),
        "mapping": Path("output/ANDES_CALIBRATION_RAW_NEWARCH/current_mapping_selection.csv"),
        "support": Path("output/ANDES_CALIBRATION_RAW_NEWARCH/signal_support_matrix.csv"),
        "calibration": Path("output/ANDES_CALIBRATION_RAW_NEWARCH/metrics/calibration_results.json"),
    }
    if any(not p.exists() for p in paths.values()):
        pytest.skip("Missing required event0 artifacts for m4 type0/type1 contract regression")

    root = Path("output") / "_test_runs" / f"m4_t0_t1_contracts_{uuid4().hex[:8]}"
    t0_root = root / "type0"
    t1_root = root / "type1"

    t0 = run_simulate_type0_use_case(
        output_root=t0_root,
        event0_profile_path=paths["profile"],
        event0_current_mapping_csv=paths["mapping"],
        event0_support_matrix_csv=paths["support"],
        event0_calibration_json=paths["calibration"],
        generate_plots=True,
    )
    t1 = run_simulate_type1_fault_use_case(
        fault_bus="39",
        output_root=t1_root,
        event0_profile_path=paths["profile"],
        event0_current_mapping_csv=paths["mapping"],
        event0_support_matrix_csv=paths["support"],
        event0_calibration_json=paths["calibration"],
        generate_plots=True,
    )

    run0 = Path(t0["run_dir"])
    run1 = Path(t1["run_dir"])
    required_rels = [
        "simulation",
        "estimated",
        "run_info.json",
        "estimated/estimation_metrics_long.csv",
        "estimated/estimation_summary_by_bus.csv",
        "estimated/estimation_summary_by_signal.csv",
        "estimated/estimation_report.json",
    ]
    for rel in required_rels:
        assert (run0 / rel).exists()
        assert (run1 / rel).exists()

    # Same plot folder structure availability
    assert (run0 / "simulation" / "plots").exists() == (run1 / "simulation" / "plots").exists()
    assert (run0 / "estimated" / "plots").exists() == (run1 / "estimated" / "plots").exists()

    info0 = json.loads((run0 / "run_info.json").read_text(encoding="utf-8"))
    info1 = json.loads((run1 / "run_info.json").read_text(encoding="utf-8"))
    assert info0.get("scenario_type") == "type0"
    assert info1.get("fault_bus") is not None
    assert info0.get("fault_bus") is None
