from __future__ import annotations

from src.simulation.m9.final_polish import build_splits_v2
from src.simulation.m9.hardening import generate_balanced_batch
from tests.helpers.m9_test_utils import PMU_LOCATION_PATH, RAW_PATH, REFERENCE_PMU_DIR


def test_m9_2_split_rebuilder_pipeline(tmp_path) -> None:
    generate_balanced_batch(
        raw_path=RAW_PATH,
        pmu_location_path=PMU_LOCATION_PATH,
        reference_pmu_dir=REFERENCE_PMU_DIR,
        output_root=tmp_path,
        n_scenarios=12,
        batch_mode="balanced_core",
        difficulty_mode="curriculum_easy_to_hard",
        save_plots=False,
    )
    out = build_splits_v2(tmp_path / "data" / "scenarios", tmp_path, split_strategy="leakage_safe_balanced", seed=19)
    assert (tmp_path / "split_manifest_v2.json").exists()
    assert out["report"]["no_scenario_leakage"] is True

