from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES
from src.simulation.m9.event5_dropout_process import Event5DropoutProcess


def _frame() -> pd.DataFrame:
    bus = "BUS29"
    data = {
        "TIMESTAMP": np.arange(0, 120, dtype=float) * 0.033,
        "DATA_PRESENT": np.ones((120,), dtype=int),
        "Event": np.zeros((120,), dtype=int),
    }
    for suffix in PMU_MEASUREMENT_SUFFIXES:
        data[f"{bus}_{suffix}"] = np.linspace(1.0, 2.0, 120)
    return pd.DataFrame(data)


def test_event5_dropout_process_injects_missingness() -> None:
    params = {
        "state_transition_matrix": {
            "NORMAL": {"NORMAL": 0.55, "PARTIAL_DROPOUT": 0.25, "FULL_DROPOUT": 0.20},
            "PARTIAL_DROPOUT": {"NORMAL": 0.30, "PARTIAL_DROPOUT": 0.40, "FULL_DROPOUT": 0.30},
            "FULL_DROPOUT": {"NORMAL": 0.30, "PARTIAL_DROPOUT": 0.20, "FULL_DROPOUT": 0.50},
        },
        "dropout_burst_length_distribution": {"mean": 8.0, "p50": 6.0},
        "inter_burst_interval_distribution": {"mean": 10.0},
        "full_vs_partial_dropout_fraction": {"full": 0.7, "partial": 0.3},
        "channel_family_dropout_tendencies": {
            "voltage": {"nan_fraction_mean": 0.4},
            "current": {"nan_fraction_mean": 0.4},
            "frequency": {"nan_fraction_mean": 0.2},
            "rocof": {"nan_fraction_mean": 0.2},
        },
        "pmu_specific_dropout_tendencies": {"BUS29": {"dropout_burst_mean_frames": 8.0, "interburst_mean_frames": 9.0}},
    }
    process = Event5DropoutProcess(params=params, seed=7)
    frame = _frame()
    updated, latent = process.apply(
        bus="BUS29",
        frame=frame,
        start_idx=20,
        end_idx=80,
        physical_event_mask=np.zeros((len(frame),), dtype=bool),
    )
    assert len(latent) > 0
    assert int((updated["Event"] == 5).sum()) > 0
    nan_rows = updated.filter(like="BUS29_").isna().any(axis=1)
    assert int(nan_rows.sum()) > 0

