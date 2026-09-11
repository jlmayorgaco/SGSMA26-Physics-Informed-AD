"""Explicit data-boundary APIs for E04-A leakage checks."""
from __future__ import annotations
import numpy as np

def observed_measurements(frame):
    """Return only the 32 configured PMU channels from an input frame."""
    return np.asarray([frame[f"pmu_{i}"] for i in range(1,33)], dtype=float)

def evaluation_ground_truth(frame):
    """Evaluation-only API; estimator modules must not import this function."""
    return np.asarray([frame[f"hidden_{i}"] for i in range(1,63)], dtype=float)
