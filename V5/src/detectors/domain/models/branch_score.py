from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class BranchScore:
    probability: np.ndarray
    raw_score: np.ndarray | None = None
    backend: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.probability = np.asarray(self.probability, dtype=float)
        if self.raw_score is not None:
            self.raw_score = np.asarray(self.raw_score, dtype=float)
            if len(self.raw_score) != len(self.probability):
                raise ValueError("raw_score length must match probability length")

