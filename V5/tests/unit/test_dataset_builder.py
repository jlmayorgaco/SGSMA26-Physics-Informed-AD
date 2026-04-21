from __future__ import annotations

from pathlib import Path

from src.detectors.domain.models.detector_config import DetectorConfigV2
from src.detectors.training.datasets.detector_dataset_builder import DetectorDatasetBuilder
from src.detectors.training.datasets.detector_split_loader import SplitRecord
from tests.helpers.detector_phase1_utils import create_synthetic_scenario


def test_dataset_builder_creates_window_samples(tmp_path: Path) -> None:
    scenario_dir = create_synthetic_scenario(tmp_path, scenario_id="SIMA", abnormal_start_idx=24)
    record = SplitRecord(
        scenario_id="SIMA",
        scenario_dir=scenario_dir,
        split="train",
        template_name="TEMPLATE_EVENT5_MISSING_ONLY",
        event_coarse=5,
        difficulty_level="easy",
        scenario_family="EVENT5",
    )
    builder = DetectorDatasetBuilder(DetectorConfigV2())
    dataset = builder.build_for_records([record], fit=True)
    assert len(dataset.samples) > 0
    assert "SIMA" in dataset.by_scenario
    assert dataset.by_scenario["SIMA"].x_windows.shape[0] == len(dataset.samples)
    assert any(sample.y_binary == 1 for sample in dataset.samples)

