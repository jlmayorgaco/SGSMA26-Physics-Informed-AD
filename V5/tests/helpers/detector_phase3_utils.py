from __future__ import annotations

from pathlib import Path

import pandas as pd

from tests.helpers.detector_phase1_utils import create_synthetic_scenario


def build_phase3_splits(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    data_root = tmp_path / "data"
    s0 = create_synthetic_scenario(data_root, scenario_id="P3S0", abnormal_start_idx=70)
    s1 = create_synthetic_scenario(data_root, scenario_id="P3S1", abnormal_start_idx=22)
    s2 = create_synthetic_scenario(data_root, scenario_id="P3S2", abnormal_start_idx=28)
    s3 = create_synthetic_scenario(data_root, scenario_id="P3S3", abnormal_start_idx=35)

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    test_csv = tmp_path / "test.csv"

    pd.DataFrame(
        [
            {"scenario_id": "P3S0", "scenario_dir": str(s0), "split": "train", "template_name": "TEMPLATE_EVENT0_NORMAL", "event_coarse": 0, "difficulty_level": "easy", "scenario_family": "normal_family", "seed_family": "N1"},
            {"scenario_id": "P3S1", "scenario_dir": str(s1), "split": "train", "template_name": "TEMPLATE_EVENT5_MISSING_ONLY", "event_coarse": 5, "difficulty_level": "medium", "scenario_family": "cyber_family", "seed_family": "C1"},
        ]
    ).to_csv(train_csv, index=False)

    pd.DataFrame(
        [
            {"scenario_id": "P3S2", "scenario_dir": str(s2), "split": "validation", "template_name": "TEMPLATE_EVENT2_LINE_TRIP", "event_coarse": 2, "difficulty_level": "hard", "scenario_family": "physical_family", "seed_family": "P1"},
        ]
    ).to_csv(val_csv, index=False)

    pd.DataFrame(
        [
            {"scenario_id": "P3S1", "scenario_dir": str(s1), "split": "test", "template_name": "TEMPLATE_EVENT5_MISSING_ONLY", "event_coarse": 5, "difficulty_level": "medium", "scenario_family": "cyber_family", "seed_family": "C1"},
            {"scenario_id": "P3S3", "scenario_dir": str(s3), "split": "test", "template_name": "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL", "event_coarse": 6, "difficulty_level": "hard", "scenario_family": "concurrent_family", "seed_family": "CP1"},
        ]
    ).to_csv(test_csv, index=False)

    return train_csv, val_csv, test_csv, s3
