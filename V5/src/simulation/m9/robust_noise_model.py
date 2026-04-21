"""Shared robust heavy-tailed noise model for RAW-informed cyber simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def _suffix_from_column(column: str) -> str:
    return str(column).split("_", maxsplit=1)[-1].upper()


@dataclass(slots=True)
class RobustNoiseModel:
    """Per-channel Student-t residual noise model."""

    baseline: dict[str, Any]
    seed: int = 12345
    _rng: np.random.Generator = field(init=False, repr=False)
    _by_channel: dict[str, Any] = field(init=False, repr=False)
    _by_family: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._by_channel = dict(self.baseline.get("per_pmu_channel", {}))
        self._by_family = dict(self.baseline.get("channel_family_summary", {}))

    def channel_params(self, *, bus: str, column: str) -> tuple[float, float]:
        suffix = _suffix_from_column(column)
        key = f"{bus}::{suffix}"
        stats = self._by_channel.get(key, {})
        if stats:
            sigma = max(float(stats.get("robust_sigma", stats.get("std", 1.0))), 1e-6)
            dof = float(stats.get("student_t_df", 8.0))
            return sigma, float(np.clip(dof, 3.5, 50.0))
        family = "other"
        if suffix == "FREQ":
            family = "frequency"
        elif suffix == "ROCOF":
            family = "rocof"
        elif suffix.startswith("V"):
            family = "voltage"
        elif suffix.startswith("I"):
            family = "current"
        family_stats = self._by_family.get(family, {})
        sigma = max(float(family_stats.get("robust_sigma_mean", family_stats.get("std_mean", 1.0))), 1e-6)
        dof = float(family_stats.get("student_t_df_mean", 8.0))
        return sigma, float(np.clip(dof, 3.5, 50.0))

    def sample_noise(self, *, sigma: float, dof: float, size: int) -> np.ndarray:
        draws = self._rng.standard_t(df=max(3.5, dof), size=int(size))
        return np.asarray(draws, dtype=float) * float(max(sigma, 1e-6))

    def apply_to_frame(
        self,
        *,
        bus: str,
        frame: pd.DataFrame,
        columns: list[str],
        indices: np.ndarray | None = None,
        scale_multiplier: float = 1.0,
    ) -> pd.DataFrame:
        out = frame.copy()
        if indices is None:
            idx = np.arange(len(out), dtype=int)
        else:
            idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            return out
        for column in columns:
            if column not in out.columns:
                continue
            values = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
            sigma, dof = self.channel_params(bus=bus, column=column)
            noise = self.sample_noise(sigma=sigma * float(scale_multiplier), dof=dof, size=idx.size)
            values[idx] = values[idx] + noise
            out[column] = values
        return out
