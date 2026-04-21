from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from src.detectors.models import BranchScores, WindowedBatch


class Preprocessor(Protocol):
    def fit_transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]: ...

    def transform(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]: ...


class WindowBuilder(Protocol):
    def build(self, frame: pd.DataFrame, feature_columns: list[str], scenario_id: str) -> WindowedBatch: ...


class Branch(Protocol):
    def fit(self, batch: WindowedBatch) -> None: ...

    def predict(self, batch: WindowedBatch) -> BranchScores: ...


class Fusion(Protocol):
    def fuse(
        self,
        cyber_probabilities: np.ndarray,
        physical_probabilities: np.ndarray,
        metadata: pd.DataFrame,
    ) -> np.ndarray: ...


class Postprocessor(Protocol):
    def smooth(self, scores: np.ndarray) -> np.ndarray: ...

