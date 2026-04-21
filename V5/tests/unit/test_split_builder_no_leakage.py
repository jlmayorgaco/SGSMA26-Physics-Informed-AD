from __future__ import annotations

from src.simulation.m9.hardening import build_splits_no_leakage, generate_balanced_batch
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_split_builder_no_leakage(tmp_path) -> None:
    generate_balanced_batch(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        batch_mode="balanced_core",
        n_scenarios=14,
        difficulty_mode="uniform",
        save_plots=False,
    )
    out = build_splits_no_leakage(tmp_path / "data" / "scenarios", tmp_path)
    assert out["report"]["no_scenario_leakage"]
    assert out["report"]["no_near_duplicate_family_leakage"]

