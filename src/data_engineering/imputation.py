"""Phase-1 scaffold module for imputation."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

import pandas as pd


CANONICAL_MAP = {
    "TIMESTAMP": "TIMESTAMP",
    "DATA_PRESENT": "DATA_PRESENT",
    "EVENT": "EVENT",
    "VA_ANG": "VA_ANG",
    "VA_MAG": "VA_MAG",
    "VB_ANG": "VB_ANG",
    "VB_MAG": "VB_MAG",
    "VC_ANG": "VC_ANG",
    "VC_MAG": "VC_MAG",
    "IA_ANG": "IA_ANG",
    "IA_MAG": "IA_MAG",
    "IB_ANG": "IB_ANG",
    "IB_MAG": "IB_MAG",
    "IC_ANG": "IC_ANG",
    "IC_MAG": "IC_MAG",
    "FREQ": "FREQ",
    "ROCOF": "ROCOF",
}

ALIASES = {
    "FREQUENCY": "FREQ",
    "FREQ": "FREQ",
}


@dataclass(frozen=True)
class PMUSchemaInfo:
    bus_token: str | None
    original_to_canonical: Dict[str, str]


def _norm(name: str) -> str:
    return str(name).strip().upper().replace(" ", "").replace("-", "_")


def infer_bus_token(columns: list[str]) -> str | None:
    for col in columns:
        m = re.match(r"^(BUS\d+)_", _norm(col))
        if m:
            return m.group(1)
    return None


def canonicalize_pmu_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, PMUSchemaInfo]:
    rename_map: Dict[str, str] = {}
    bus_token = infer_bus_token(list(df.columns))

    for col in df.columns:
        norm = _norm(col)

        if bus_token and norm.startswith(f"{bus_token}_"):
            norm = norm[len(bus_token) + 1 :]

        norm = ALIASES.get(norm, norm)

        if norm in CANONICAL_MAP:
            rename_map[col] = CANONICAL_MAP[norm]

    out = df.rename(columns=rename_map).copy()

    expected = [
        "TIMESTAMP",
        "VA_ANG", "VA_MAG",
        "VB_ANG", "VB_MAG",
        "VC_ANG", "VC_MAG",
        "IA_ANG", "IA_MAG",
        "IB_ANG", "IB_MAG",
        "IC_ANG", "IC_MAG",
        "FREQ", "ROCOF",
        "DATA_PRESENT", "EVENT",
    ]

    for col in expected:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[expected]
    return out, PMUSchemaInfo(bus_token=bus_token, original_to_canonical=rename_map)