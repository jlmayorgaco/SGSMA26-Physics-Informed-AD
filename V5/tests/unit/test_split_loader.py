from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.detectors.training.datasets.detector_split_loader import load_split_csv, validate_no_split_leakage


def test_split_loader_reads_csv_and_detects_no_leakage(tmp_path: Path) -> None:
    train = pd.DataFrame(
        [
            {"scenario_id": "A", "scenario_dir": "data/scenarios/A", "split": "train", "scenario_family": "F1"},
            {"scenario_id": "B", "scenario_dir": "data/scenarios/B", "split": "train", "scenario_family": "F2"},
        ]
    )
    val = pd.DataFrame([{"scenario_id": "C", "scenario_dir": "data/scenarios/C", "split": "val", "scenario_family": "F3"}])
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    train.to_csv(train_csv, index=False)
    val.to_csv(val_csv, index=False)
    train_records = load_split_csv(train_csv, split_name="train", workspace_root=Path(".").resolve())
    val_records = load_split_csv(val_csv, split_name="val", workspace_root=Path(".").resolve())
    checks = validate_no_split_leakage({"train": train_records, "val": val_records})
    assert checks["no_scenario_leakage"] is True

