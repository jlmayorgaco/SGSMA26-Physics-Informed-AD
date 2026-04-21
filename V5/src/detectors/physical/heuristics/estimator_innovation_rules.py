from __future__ import annotations

import numpy as np
import pandas as pd


def estimator_innovation_evidence(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(metadata)
    if n == 0:
        zeros = np.zeros((0,), dtype=float)
        return {"innovation_score": zeros, "score": zeros}
    if "ESTIMATOR_DIFFICULTY_SCORE" in metadata.columns:
        innovation = pd.to_numeric(metadata["ESTIMATOR_DIFFICULTY_SCORE"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    elif "estimator_difficulty_score" in metadata.columns:
        innovation = pd.to_numeric(metadata["estimator_difficulty_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        innovation = np.zeros((n,), dtype=float)
    score = np.clip(innovation, 0.0, 1.0)
    return {"innovation_score": innovation, "score": score}

