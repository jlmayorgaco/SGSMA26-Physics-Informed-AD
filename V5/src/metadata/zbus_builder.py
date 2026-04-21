"""Zbus builder with invertibility diagnostics."""

from __future__ import annotations

import numpy as np


def build_zbus(ybus: np.ndarray) -> tuple[np.ndarray, dict]:
    """Attempt Zbus = inv(Ybus); report condition/invertibility details."""
    y = np.asarray(ybus, dtype=complex)
    if y.ndim != 2 or y.shape[0] != y.shape[1]:
        raise ValueError("Ybus must be a square matrix.")

    cond = float(np.linalg.cond(y))
    invertible = np.isfinite(cond) and cond < 1e12
    notes: list[str] = []
    if invertible:
        z = np.linalg.inv(y)
    else:
        z = np.linalg.pinv(y)
        notes.append("Ybus ill-conditioned or singular; using pseudo-inverse for Zbus.")

    meta = {
        "condition_number": cond,
        "invertible": bool(invertible),
        "shape": [int(y.shape[0]), int(y.shape[1])],
        "notes": notes,
    }
    return z, meta
