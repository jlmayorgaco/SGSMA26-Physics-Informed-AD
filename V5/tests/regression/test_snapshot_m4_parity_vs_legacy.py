from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.application.use_cases.simulate_physical_event import run_simulate_type1_fault_use_case
from src.infrastructure.legacy.m4_adapter import run_type1_fault_simulation
from src.simulation.type1_config import Type1Config


@pytest.mark.regression
@pytest.mark.andes
def test_m4_parity_vs_legacy_key_artifacts() -> None:
    if os.environ.get("RUN_ANDES_TESTS") != "1":
        pytest.skip("Set RUN_ANDES_TESTS=1 to run ANDES regression tests")
    paths = {
        "profile": Path("output/M2_RAW0001_NOISE_PROFILE_NEWARCH/dataset_profiles_raw.json"),
        "mapping": Path("output/ANDES_CALIBRATION_RAW_NEWARCH/current_mapping_selection.csv"),
        "support": Path("output/ANDES_CALIBRATION_RAW_NEWARCH/signal_support_matrix.csv"),
        "calibration": Path("output/ANDES_CALIBRATION_RAW_NEWARCH/metrics/calibration_results.json"),
    }
    if any(not p.exists() for p in paths.values()):
        pytest.skip("Missing required event0 artifacts for m4 parity regression")

    root = Path("output") / "_test_runs" / f"m4_regression_{uuid4().hex[:8]}"
    migrated_root = root / "migrated"
    legacy_root = root / "legacy"
    summary = run_simulate_type1_fault_use_case(
        fault_bus="39",
        output_root=migrated_root,
        event0_profile_path=paths["profile"],
        event0_current_mapping_csv=paths["mapping"],
        event0_support_matrix_csv=paths["support"],
        event0_calibration_json=paths["calibration"],
        generate_plots=False,
    )
    run_type1_fault_simulation(
        fault_bus="39",
        output_root=legacy_root,
        event0_profile_path=paths["profile"],
        event0_current_mapping_csv=paths["mapping"],
        event0_support_matrix_csv=paths["support"],
        event0_calibration_json=paths["calibration"],
        generate_plots=False,
    )
    migrated_run = Path(summary["run_dir"])
    legacy_run = legacy_root / Type1Config(fault_bus="39").run_folder_name()

    for rel in [
        "estimated/estimation_metrics_long.csv",
        "estimated/estimation_summary_by_bus.csv",
        "estimated/estimation_summary_by_signal.csv",
    ]:
        assert_frame_equal(
            pd.read_csv(migrated_run / rel),
            pd.read_csv(legacy_run / rel),
            check_dtype=False,
            check_like=True,
            rtol=1e-6,
            atol=1e-8,
        )

    m_info = json.loads((migrated_run / "run_info.json").read_text(encoding="utf-8"))
    l_info = json.loads((legacy_run / "run_info.json").read_text(encoding="utf-8"))
    for key in ["fault_bus", "folders", "event_label_mode"]:
        assert key in m_info
        assert key in l_info

    assert len(list((migrated_run / "simulation").glob("BUS*_Competition_Data_nanmask.csv"))) == len(
        list((legacy_run / "simulation").glob("BUS*_Competition_Data_nanmask.csv"))
    )
    assert len(list((migrated_run / "estimated").glob("BUS*_Competition_Data_nanmask.csv"))) == len(
        list((legacy_run / "estimated").glob("BUS*_Competition_Data_nanmask.csv"))
    )

    migrated_plots = list((migrated_run / "simulation" / "plots").rglob("*.png")) if (migrated_run / "simulation" / "plots").exists() else []
    legacy_plots = list((legacy_run / "simulation" / "plots").rglob("*.png")) if (legacy_run / "simulation" / "plots").exists() else []
    assert len(migrated_plots) == len(legacy_plots)
