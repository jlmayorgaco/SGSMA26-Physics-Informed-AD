from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES
from src.simulation.m9.raw_informed_parameters import extract_event7_parameters


def _table() -> pd.DataFrame:
    channels = [suffix.upper() for suffix in PMU_MEASUREMENT_SUFFIXES]
    rows = []
    for idx in range(80):
        event = 0 if idx < 50 else 7
        row = {
            "TIMESTAMP": float(idx) * 0.033,
            "BUS": "BUS10",
            "CHUNK_NAME": "chunk02_event_7_bad_data",
            "EVENT": event,
            "DATA_PRESENT": 1,
        }
        for channel in channels:
            base = 1.0 + 0.005 * idx
            if channel in {"FREQ", "ROCOF"}:
                base = 60.0 + 0.002 * idx if channel == "FREQ" else 0.01 * np.sin(0.2 * idx)
            row[channel] = base
        if event == 7:
            row["VA_MAG"] = row["VA_MAG"] + (4.5 if idx % 4 == 0 else 0.0)
            row["FREQ"] = row["FREQ"] + 0.07 * np.sin(0.6 * idx)
            if 60 <= idx <= 64:
                row["IA_MAG"] = 2.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_event7_parameter_extraction_produces_mode_priors() -> None:
    params = extract_event7_parameters(_table())
    assert "mode_prior" in params
    prior = params["mode_prior"]
    assert {"SPIKE", "BIAS_DRIFT", "STUCK", "REPLAY_LIKE"}.issubset(prior.keys())
    total = float(sum(float(value) for value in prior.values()))
    assert total > 0.99
    assert total < 1.01
    assert params["spike_amplitude_distribution"]["count"] >= 1

