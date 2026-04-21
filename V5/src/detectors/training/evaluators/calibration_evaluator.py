from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.detectors.shared.utils.calibration import brier_score, expected_calibration_error


@dataclass(slots=True)
class CalibrationResult:
    bins: pd.DataFrame
    ece: float
    brier: float


@dataclass(slots=True)
class CalibrationEvaluator:
    n_bins: int = 10

    def evaluate(self, y_true: np.ndarray, y_prob: np.ndarray) -> CalibrationResult:
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_prob, dtype=float)
        edges = np.linspace(0.0, 1.0, int(self.n_bins) + 1)
        rows: list[dict[str, float | int]] = []
        for i in range(len(edges) - 1):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            if i == len(edges) - 2:
                mask = (yp >= lo) & (yp <= hi)
            else:
                mask = (yp >= lo) & (yp < hi)
            if not np.any(mask):
                rows.append({"bin_start": lo, "bin_end": hi, "count": 0, "mean_pred": np.nan, "empirical": np.nan})
                continue
            rows.append(
                {
                    "bin_start": lo,
                    "bin_end": hi,
                    "count": int(mask.sum()),
                    "mean_pred": float(yp[mask].mean()),
                    "empirical": float(yt[mask].mean()),
                }
            )
        return CalibrationResult(
            bins=pd.DataFrame(rows),
            ece=float(expected_calibration_error(yt, yp, bins=self.n_bins)),
            brier=float(brier_score(yt, yp)),
        )
