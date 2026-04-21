"""ANDES helper for M6 validation truth generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.calibration.andes_runtime import get_timeseries, run_andes_normal


@dataclass(slots=True)
class AndesTruth:
    timestamps: np.ndarray
    bus_ids: list[str]
    voltage_complex_pu: np.ndarray


def run_andes_ieee39_truth(tf: float = 10.0, tstep: float = 1.0 / 30.0, stride: int = 1) -> AndesTruth:
    """Run short ANDES simulation and return full-bus voltage phasor truth in p.u."""
    system = run_andes_normal(tf=tf, tstep=tstep)
    t, v_df, a_df = get_timeseries(system)
    if stride > 1:
        t = t[::stride]
        v_df = v_df.iloc[::stride]
        a_df = a_df.iloc[::stride]
    bus_ids = [str(c) if str(c).startswith("BUS") else f"BUS{c}" for c in v_df.columns]
    v_complex = np.asarray(v_df.to_numpy(float), dtype=float) * np.exp(1j * np.deg2rad(a_df.to_numpy(float)))
    return AndesTruth(timestamps=np.asarray(t, dtype=float), bus_ids=bus_ids, voltage_complex_pu=v_complex)

