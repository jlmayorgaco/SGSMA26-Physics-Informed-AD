from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.helpers.detector_phase1_utils import create_synthetic_scenario


def create_m10_1_family_split_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "data"
    s_normal = create_synthetic_scenario(root, scenario_id="M101_N", abnormal_start_idx=9999)
    s_physical = create_synthetic_scenario(root, scenario_id="M101_P", abnormal_start_idx=20)
    s_cyber = create_synthetic_scenario(root, scenario_id="M101_C", abnormal_start_idx=24)
    s_conc = create_synthetic_scenario(root, scenario_id="M101_CC", abnormal_start_idx=28)
    s_extra = create_synthetic_scenario(root, scenario_id="M101_E", abnormal_start_idx=34)

    train = pd.DataFrame(
        [
            {"scenario_id": "M101_N", "scenario_dir": str(s_normal), "split": "train", "template_name": "TEMPLATE_EVENT0_NORMAL", "event_coarse": 0, "difficulty_level": "easy", "scenario_family": "normal_f", "seed_family": "N1"},
            {"scenario_id": "M101_P", "scenario_dir": str(s_physical), "split": "train", "template_name": "TEMPLATE_EVENT2_LINE_OUTAGE", "event_coarse": 2, "difficulty_level": "easy", "scenario_family": "physical_f", "seed_family": "P1"},
            {"scenario_id": "M101_C", "scenario_dir": str(s_cyber), "split": "train", "template_name": "TEMPLATE_EVENT5_MISSING_ONLY", "event_coarse": 5, "difficulty_level": "medium", "scenario_family": "cyber_f", "seed_family": "C1"},
            {"scenario_id": "M101_CC", "scenario_dir": str(s_conc), "split": "train", "template_name": "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL", "event_coarse": 6, "difficulty_level": "hard", "scenario_family": "concurrent_f", "seed_family": "CC1"},
            {"scenario_id": "M101_E", "scenario_dir": str(s_extra), "split": "train", "template_name": "TEMPLATE_EVENT7_BAD_DATA", "event_coarse": 7, "difficulty_level": "hard", "scenario_family": "cyber_f2", "seed_family": "C2"},
        ]
    )
    val = train.iloc[[0]].assign(split="val").copy()
    test = train.iloc[[1]].assign(split="test").copy()

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"
    train.to_csv(train_csv, index=False)
    val.to_csv(val_csv, index=False)
    test.to_csv(test_csv, index=False)
    return train_csv, val_csv, test_csv, s_cyber
