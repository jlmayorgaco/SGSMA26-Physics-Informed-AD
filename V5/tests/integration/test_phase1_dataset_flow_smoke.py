from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.detectors.domain.models.detector_config import DetectorConfigV2
from src.detectors.training.datasets.detector_dataset_builder import DetectorDatasetBuilder
from src.detectors.training.datasets.detector_split_loader import load_split_csv
from tests.helpers.detector_phase1_utils import create_synthetic_scenario


def test_phase1_dataset_flow_smoke(tmp_path: Path) -> None:
    scenario_dir = create_synthetic_scenario(tmp_path, scenario_id="SIMFLOW", abnormal_start_idx=28)
    split_csv = tmp_path / "train.csv"
    pd.DataFrame(
        [
            {
                "scenario_id": "SIMFLOW",
                "scenario_dir": str(scenario_dir),
                "split": "train",
                "template_name": "TEMPLATE_EVENT5_MISSING_ONLY",
                "event_coarse": 5,
                "difficulty_level": "easy",
                "scenario_family": "EVENT5",
                "seed_family": "S1",
            }
        ]
    ).to_csv(split_csv, index=False)
    records = load_split_csv(split_csv, split_name="train", workspace_root=Path(".").resolve())
    dataset = DetectorDatasetBuilder(DetectorConfigV2()).build_for_records(records, fit=True)
    assert len(dataset.samples) > 0
    assert "SIMFLOW" in dataset.by_scenario

