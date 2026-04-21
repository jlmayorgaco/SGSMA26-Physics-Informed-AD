from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ResidualChangeDetector:
    baseline_median: float = 0.0
    baseline_iqr: float = 1.0
    score_scale: float = 2.5

    def fit(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=float)
        if x.size == 0:
            self.baseline_median = 0.0
            self.baseline_iqr = 1.0
            return
        self.baseline_median = float(np.nanmedian(x))
        q1 = float(np.nanquantile(x, 0.25))
        q3 = float(np.nanquantile(x, 0.75))
        self.baseline_iqr = max(1e-6, q3 - q1)

    def score(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        z = np.abs(x - self.baseline_median) / self.baseline_iqr
        return np.clip(z / max(self.score_scale, 1e-6), 0.0, 1.0)

