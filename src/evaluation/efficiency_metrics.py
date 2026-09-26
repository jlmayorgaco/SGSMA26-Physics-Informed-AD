"""Efficiency metric placeholders for phase-1 wiring."""

from __future__ import annotations

from typing import Any


def parameter_count_placeholder(model: Any) -> int:
    """Best-effort parameter count for model-like objects."""
    if hasattr(model, "count_params") and callable(model.count_params):
        return int(model.count_params())
    if hasattr(model, "parameters") and callable(model.parameters):
        return int(sum(int(p.size) for p in model.parameters()))
    return 0


def model_size_mb_placeholder(model: Any) -> float:
    """Placeholder returning 0.0 until phase-2 model serialization is added."""
    _ = model
    return 0.0
