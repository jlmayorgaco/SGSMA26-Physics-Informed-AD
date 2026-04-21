from __future__ import annotations

from pathlib import Path
from dataclasses import asdict

import numpy as np
import pandas as pd

from src.detectors.configs import DetectorConfig
from src.detectors.data.ingestion import load_raw_frame, load_scenario_frame, load_split_records
from src.detectors.data.preprocessing import SharedPreprocessor
from src.detectors.data.window_builder import SlidingWindowBuilder, concat_batches
from src.detectors.models import ScenarioRecord, WindowedBatch


def build_batches_for_records(records: list[ScenarioRecord], config: DetectorConfig, preprocessor: SharedPreprocessor | None = None, fit: bool = True) -> WindowedBatch:
    prep = preprocessor or SharedPreprocessor(config.preprocessing)
    builder = SlidingWindowBuilder(config.window)
    batches: list[WindowedBatch] = []
    for idx, rec in enumerate(records):
        frame = load_scenario_frame(rec)
        clean, features = prep.fit_transform(frame) if (fit and idx == 0 and not prep.feature_columns) else prep.transform(frame)
        # Include aggregated DATA_PRESENT feature for cyber rules.
        if "DATA_PRESENT" in clean.columns and "DATA_PRESENT" not in features:
            features = list(features) + ["DATA_PRESENT"]
        batch = builder.build(clean, feature_columns=features, scenario_id=rec.scenario_id)
        if len(batch.y):
            batches.append(batch)
    return concat_batches(batches)


def build_batch_for_raw(raw_dir: Path, config: DetectorConfig, preprocessor: SharedPreprocessor) -> WindowedBatch:
    frame = load_raw_frame(raw_dir)
    clean, features = preprocessor.transform(frame)
    if "DATA_PRESENT" in clean.columns and "DATA_PRESENT" not in features:
        features = list(features) + ["DATA_PRESENT"]
    builder = SlidingWindowBuilder(config.window)
    return builder.build(clean, feature_columns=features, scenario_id=raw_dir.name.upper())


def load_records_from_split(split_csv: Path, workspace_root: Path | None = None) -> list[ScenarioRecord]:
    return load_split_records(split_csv=split_csv, workspace_root=workspace_root)


def save_detection_artifacts(output_dir: Path, run_name: str, artifacts) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts.frame_predictions.to_csv(output_dir / f"{run_name}_frame_predictions.csv", index=False)
    pd.DataFrame([asdict(chunk) for chunk in artifacts.event_chunks]).to_csv(output_dir / f"{run_name}_event_chunks.csv", index=False)
    metrics_frame = pd.DataFrame([asdict(artifacts.metrics)])
    metrics_frame.to_csv(output_dir / f"{run_name}_metrics.csv", index=False)
    pd.DataFrame([artifacts.branch_summary]).to_csv(output_dir / f"{run_name}_branch_summary.csv", index=False)
