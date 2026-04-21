from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ScenarioRecord:
    scenario_id: str
    scenario_dir: Path
    template_name: str = ""
    event_coarse: int | None = None
    difficulty_level: str = ""
    scenario_family: str = ""
    seed_family: str = ""
    split: str = ""


@dataclass(slots=True)
class WindowedBatch:
    x: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    metadata: pd.DataFrame
    feature_names: list[str]


@dataclass(slots=True)
class BranchScores:
    probabilities: np.ndarray
    details: dict[str, Any]


@dataclass(slots=True)
class EventChunk:
    start_time_s: float
    end_time_s: float
    duration_s: float
    mean_score: float
    max_score: float
    frame_count: int


@dataclass(slots=True)
class DetectorMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    balanced_accuracy: float
    roc_auc: float | None
    pr_auc: float | None
    support_positive: int
    support_negative: int


@dataclass(slots=True)
class DetectionRunArtifacts:
    metrics: DetectorMetrics
    frame_predictions: pd.DataFrame
    event_chunks: list[EventChunk]
    branch_summary: dict[str, Any]

