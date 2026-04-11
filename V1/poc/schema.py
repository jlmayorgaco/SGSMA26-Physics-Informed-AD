"""Shared schema and reproducibility helpers for the SGSMA POC."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PMU_BUSES: list[int] = [2, 5, 6, 10, 19, 22, 29, 39]
ALL_BUSES: list[int] = list(range(1, 40))
CHANNEL_SUFFIXES: list[str] = [
    "VA_ANG",
    "VA_MAG",
    "VB_ANG",
    "VB_MAG",
    "VC_ANG",
    "VC_MAG",
    "IA_ANG",
    "IA_MAG",
    "IB_ANG",
    "IB_MAG",
    "IC_ANG",
    "IC_MAG",
    "Freq",
    "ROCOF",
]


class EventLabel(IntEnum):
    """Nine SGSMA labels used by both real and synthetic data."""

    NORMAL = 0
    FAULT_3LG = 1
    LINE_OUTAGE = 2
    GEN_CHANGE = 3
    LOAD_CHANGE = 4
    PMU_DROPOUT = 5
    CYBER_PHYSICAL = 6
    BAD_DATA = 7
    UNKNOWN = 8


@dataclass(frozen=True)
class Scenario:
    """Declarative synthetic event specification.

    Parameters are intentionally loose because ANDES events, PMU dropouts, and
    artificial bad-data injections need different fields while sharing one grid.
    """

    scenario_id: str
    label: int
    event_bus: int
    t_event: float
    duration: float
    category: str
    params: dict[str, Any] = field(default_factory=dict)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and optional ML stacks from one config value."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import lightgbm as lgb  # noqa: F401
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def measurement_columns() -> list[str]:
    """Return the 112 measurement columns in competition order."""

    return [f"BUS{bus}_{suffix}" for bus in PMU_BUSES for suffix in CHANNEL_SUFFIXES]


def data_present_columns() -> list[str]:
    """Return the eight PMU data-present columns after merge."""

    return [f"BUS{bus}_DATA_PRESENT" for bus in PMU_BUSES]


def single_bus_columns(bus: int) -> list[str]:
    """Return one bus CSV header in the real competition schema."""

    return ["TIMESTAMP"] + [f"BUS{bus}_{suffix}" for suffix in CHANNEL_SUFFIXES] + [
        "DATA_PRESENT",
        "Event",
    ]


def observations_from_merged(df: pd.DataFrame) -> np.ndarray:
    """Extract a ``(T, 112)`` PMU observation matrix from a merged DataFrame."""

    cols = measurement_columns()
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing measurement columns: {missing[:5]}")
    return df[cols].to_numpy(dtype=float)


def observations_from_scenario_dir(path: str | Path) -> pd.DataFrame:
    """Load one scenario directory through the production CSV loader."""

    from src.io.load_csv import load_all

    return load_all(Path(path))


def robust_mean_std(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-safe center and scale vectors with finite fallbacks."""

    mean = np.nanmean(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    centered = np.where(np.isnan(x), mean[None, :], x) - mean[None, :]
    std = np.nanstd(centered, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-9), std, 1.0)
    return mean.astype(float), std.astype(float)

