"""Small physics-based label guards for event classification.

These rules are intentionally sparse and parameter-free. They correct cases
where a compact boosted tree predicts a class that conflicts with hard PMU or
power-system evidence.
"""
from __future__ import annotations

import numpy as np

from src.classifier.features import FEATURE_NAMES


def _feat(features: np.ndarray, name: str, default: float = 0.0) -> float:
    try:
        return float(features[FEATURE_NAMES.index(name)])
    except (ValueError, IndexError):
        return default


def apply_physics_label_overrides(label: int, features: np.ndarray) -> int:
    """Return a corrected label when hard physics evidence contradicts ML."""
    missing = _feat(features, "pmu_missing_count")
    va_max = _feat(features, "VA_MAG_max")
    ia_max = _feat(features, "IA_MAG_max")
    state_top = max(_feat(features, "state_top1_energy"), 1e-12)
    bus7 = _feat(features, "state_bus7_energy")

    # A missing-data or cyber+physical label requires an actual missing PMU,
    # unless the classifier is reacting to a severe physical fault.
    if label in {5, 6} and missing < 0.5:
        if va_max > 50_000.0 and ia_max > 1_000.0:
            return 1

    # Bus 7 load changes are non-PMU events: they look like a local voltage /
    # current depression around PMUs 5/6, not a generator/inertia event. The
    # Ybus state proxy catches this when Bus7 energy is near the maximum.
    if label in {0, 3} and missing < 0.5 and bus7 / state_top > 0.85:
        return 4

    return int(label)
