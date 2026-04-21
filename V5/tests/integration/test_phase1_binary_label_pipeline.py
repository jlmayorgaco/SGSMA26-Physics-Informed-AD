from __future__ import annotations

from pathlib import Path

from src.detectors.domain.models.detector_config import DetectorConfigV2
from src.detectors.training.datasets.detector_dataset_builder import DetectorDatasetBuilder
from src.detectors.training.datasets.detector_split_loader import SplitRecord
from tests.helpers.detector_phase1_utils import create_synthetic_scenario


def test_phase1_binary_label_pipeline(tmp_path: Path) -> None:
    scenario_dir = create_synthetic_scenario(tmp_path, scenario_id="SIMBIN", abnormal_start_idx=30)
    record = SplitRecord(
        scenario_id="SIMBIN",
        scenario_dir=scenario_dir,
        split="train",
        template_name="TEMPLATE_EVENT5_MISSING_ONLY",
        event_coarse=5,
        difficulty_level="medium",
        scenario_family="EVENT5",
    )
    dataset = DetectorDatasetBuilder(DetectorConfigV2()).build_for_records([record], fit=True)
    labels = [s.y_binary for s in dataset.samples]
    assert 1 in labels
    assert all(v in {0, 1} for v in labels)
    assert any(s.is_cyber_event for s in dataset.samples)

