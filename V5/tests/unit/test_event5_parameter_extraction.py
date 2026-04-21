from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES
from src.simulation.m9.raw_informed_parameters import extract_event5_parameters


def _table() -> pd.DataFrame:
    rows = []
    channels = [suffix.upper() for suffix in PMU_MEASUREMENT_SUFFIXES]
    for idx in range(40):
        row = {
            "TIMESTAMP": float(idx) * 0.033,
            "BUS": "BUS29",
            "CHUNK_NAME": "chunk08_event_5_missing_data",
            "EVENT": 5,
            "DATA_PRESENT": 1,
        }
        for channel in channels:
            row[channel] = 1.0 + 0.01 * idx
        if 8 <= idx <= 14:
            row["DATA_PRESENT"] = 0
            for channel in channels:
                row[channel] = np.nan
        if 25 <= idx <= 30:
            row["DATA_PRESENT"] = 1
            row["VA_MAG"] = np.nan
            row["IA_MAG"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def test_event5_parameter_extraction_has_dropout_signatures() -> None:
    params = extract_event5_parameters(_table())
    assert "state_transition_matrix" in params
    assert "dropout_burst_length_distribution" in params
    assert params["dropout_burst_length_distribution"]["count"] >= 1
    full_fraction = float(params["full_vs_partial_dropout_fraction"]["full"])
    partial_fraction = float(params["full_vs_partial_dropout_fraction"]["partial"])
    assert full_fraction > 0.0
    assert partial_fraction > 0.0
    assert abs((full_fraction + partial_fraction) - 1.0) < 1e-6

