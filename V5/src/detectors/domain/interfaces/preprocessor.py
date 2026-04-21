from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd

from src.detectors.domain.models.window_sample import WindowSample


class Preprocessor(Protocol):
    """Convert PMU frames into deterministic window samples."""

    def fit(self, frames: Sequence[pd.DataFrame]) -> None: ...

    def transform(self, frame: pd.DataFrame, scenario_id: str, split: str) -> list[WindowSample]: ...

