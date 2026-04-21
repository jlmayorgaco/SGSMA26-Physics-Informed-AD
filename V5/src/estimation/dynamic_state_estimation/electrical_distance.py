"""Electrical distance utilities for M8 dynamic estimators."""

from __future__ import annotations

import numpy as np

from src.metadata.electrical_distance import build_electrical_distance_matrix
from src.metadata.zbus_builder import build_zbus


def build_electrical_distance_from_ybus(ybus: np.ndarray) -> tuple[np.ndarray, dict]:
    """Build electrical-distance matrix from Ybus via Zbus."""
    zbus, z_meta = build_zbus(ybus)
    dist = build_electrical_distance_matrix(zbus)
    meta = {
        "zbus": z_meta,
        "distance_shape": [int(dist.shape[0]), int(dist.shape[1])],
        "distance_finite": bool(np.isfinite(dist).all()),
    }
    return dist, meta

