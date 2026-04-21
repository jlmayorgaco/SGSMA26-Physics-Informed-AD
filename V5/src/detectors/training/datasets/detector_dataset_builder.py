from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.domain.models.detector_config import DetectorConfigV2
from src.detectors.domain.models.window_sample import WindowSample
from src.detectors.shared.preprocessing.angle_features import add_angle_derived_features
from src.detectors.shared.preprocessing.estimator_feature_adapter import EstimatorFeatureAdapter
from src.detectors.shared.preprocessing.nan_masking import apply_nan_masking, numeric_feature_columns
from src.detectors.shared.preprocessing.normalization import (
    NormalizationState,
    apply_normalization,
    fit_normalization_state,
)
from src.detectors.shared.preprocessing.pmu_alignment import align_scenario_pmu_dir
from src.detectors.shared.preprocessing.pmu_window_builder import (
    PMUWindowBuilder,
    PMUWindowConfig,
    collate_samples_to_detection_input,
)
from src.detectors.shared.preprocessing.positive_sequence_features import add_positive_sequence_features
from src.detectors.training.datasets.detector_split_loader import SplitRecord


@dataclass(slots=True)
class BuiltDataset:
    samples: list[WindowSample]
    feature_names: list[str]
    by_scenario: dict[str, DetectionInput]


class DetectorDatasetBuilder:
    def __init__(self, config: DetectorConfigV2) -> None:
        self.config = config
        self.window_builder = PMUWindowBuilder(
            PMUWindowConfig(
                window_size=config.window.size,
                stride=config.window.stride,
                positive_ratio_threshold=config.window.positive_ratio_threshold,
            )
        )
        self.estimator_adapter = EstimatorFeatureAdapter(enabled=config.preprocessing.include_estimator_features)
        self.normalization_state: NormalizationState | None = None
        self.feature_columns: list[str] = []
        self.nan_count_before: int = 0
        self.nan_count_after: int = 0

    def _load_scenario_frame(self, record: SplitRecord) -> pd.DataFrame:
        pmu_dir = record.scenario_dir / "pmu"
        if not pmu_dir.exists():
            pmu_dir = record.scenario_dir
        frame = align_scenario_pmu_dir(pmu_dir)
        labels_path = record.scenario_dir / "labels" / "event_frame_labels.csv"
        if labels_path.exists():
            labels = pd.read_csv(labels_path)
            if "TIMESTAMP" in labels.columns:
                labels = labels.rename(columns={"EVENT": "EVENT_LABELS"})
                frame = frame.merge(labels, on="TIMESTAMP", how="left")
                if "EVENT_LABELS" in frame.columns:
                    frame["EVENT"] = pd.to_numeric(frame["EVENT_LABELS"], errors="coerce").fillna(frame["EVENT"]).astype(int)
                    frame = frame.drop(columns=["EVENT_LABELS"])
        frame["SCENARIO_ID"] = record.scenario_id
        frame["SPLIT"] = record.split
        frame["EVENT_COARSE"] = record.event_coarse if record.event_coarse is not None else np.nan
        frame["DIFFICULTY_LEVEL"] = record.difficulty_level
        frame["SCENARIO_FAMILY"] = record.scenario_family
        frame["SEED_FAMILY"] = record.seed_family
        frame["TEMPLATE_NAME"] = record.template_name
        frame = self.estimator_adapter.enrich_frame(frame, scenario_dir=record.scenario_dir)
        return frame

    def _prepare_features(self, frame: pd.DataFrame, *, fit: bool) -> tuple[pd.DataFrame, list[str], int, int]:
        out = frame.copy()
        if self.config.preprocessing.include_angle_features:
            out = add_angle_derived_features(out)
        if self.config.preprocessing.include_positive_sequence:
            out = add_positive_sequence_features(out)
        blocked = {
            "EVENT",
            "EVENT_COARSE",
            "DATA_PRESENT",
            "SCENARIO_ID",
            "SPLIT",
            "DIFFICULTY_LEVEL",
            "SCENARIO_FAMILY",
            "SEED_FAMILY",
            "TEMPLATE_NAME",
            "TIMESTAMP",
        }
        candidate_features = numeric_feature_columns(out, exclude=blocked)
        nan_before = int(out[candidate_features].isna().sum().sum()) if candidate_features else 0
        out, nan_mask = apply_nan_masking(
            out,
            feature_columns=candidate_features,
            fill_method=self.config.preprocessing.fill_method,
        )
        out = pd.concat([out, nan_mask], axis=1)
        features = candidate_features + list(nan_mask.columns)
        if "DATA_PRESENT" in out.columns and "DATA_PRESENT" not in features:
            features.append("DATA_PRESENT")
        if fit or self.normalization_state is None:
            self.normalization_state = fit_normalization_state(
                out, features, clip_quantile=self.config.preprocessing.scaling_clip_quantile
            )
            self.feature_columns = list(features)
        assert self.normalization_state is not None
        normalized = apply_normalization(out, self.normalization_state)
        for col in self.feature_columns:
            if col not in normalized.columns:
                normalized[col] = 0.0
        nan_after = int(normalized[self.feature_columns].isna().sum().sum()) if self.feature_columns else 0
        return normalized, list(self.feature_columns), nan_before, nan_after

    def build_for_records(self, records: Sequence[SplitRecord], *, fit: bool) -> BuiltDataset:
        samples: list[WindowSample] = []
        by_scenario: dict[str, DetectionInput] = {}
        for idx, record in enumerate(records):
            frame = self._load_scenario_frame(record)
            prepared, feature_names, nan_before, nan_after = self._prepare_features(frame, fit=(fit and idx == 0))
            self.nan_count_before += int(nan_before)
            self.nan_count_after += int(nan_after)
            scenario_samples = self.window_builder.build(
                prepared,
                feature_columns=feature_names,
                scenario_id=record.scenario_id,
                split=record.split,
                default_metadata={
                    "template_name": record.template_name,
                    "difficulty_level": record.difficulty_level,
                    "scenario_family": record.scenario_family,
                },
            )
            samples.extend(scenario_samples)
            by_scenario[record.scenario_id] = collate_samples_to_detection_input(
                scenario_samples,
                feature_names=feature_names,
                scenario_id=record.scenario_id,
                split=record.split,
            )
        return BuiltDataset(samples=samples, feature_names=list(self.feature_columns), by_scenario=by_scenario)


def merge_detection_inputs(inputs: Sequence[DetectionInput], *, scenario_id: str, split: str) -> DetectionInput:
    valid = [item for item in inputs if item.x_windows.shape[0] > 0]
    if not valid:
        return DetectionInput(
            scenario_id=scenario_id,
            split=split,
            x_windows=np.zeros((0, 1, 1), dtype=float),
            timestamps=np.zeros((0,), dtype=float),
            feature_names=[],
            metadata=pd.DataFrame(),
        )
    feature_names = list(valid[0].feature_names)
    x = np.concatenate([item.x_windows for item in valid], axis=0)
    ts = np.concatenate([item.timestamps for item in valid], axis=0)
    meta = pd.concat([item.metadata for item in valid], ignore_index=True)
    return DetectionInput(
        scenario_id=scenario_id,
        split=split,
        x_windows=x,
        timestamps=ts,
        feature_names=feature_names,
        metadata=meta,
    )
