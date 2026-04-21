from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES
from src.simulation.m9.event7_corruption_process import Event7CorruptionProcess


def _frame() -> pd.DataFrame:
    bus = "BUS10"
    data = {
        "TIMESTAMP": np.arange(0, 140, dtype=float) * 0.033,
        "DATA_PRESENT": np.ones((140,), dtype=int),
        "Event": np.zeros((140,), dtype=int),
    }
    for suffix in PMU_MEASUREMENT_SUFFIXES:
        if suffix == "Freq":
            data[f"{bus}_{suffix}"] = 60.0 + 0.005 * np.sin(np.arange(140) / 4.0)
        elif suffix == "ROCOF":
            data[f"{bus}_{suffix}"] = 0.02 * np.cos(np.arange(140) / 6.0)
        else:
            data[f"{bus}_{suffix}"] = 1.0 + 0.002 * np.arange(140)
    return pd.DataFrame(data)


def _noise_baseline() -> dict:
    payload = {"per_pmu_channel": {}, "channel_family_summary": {}}
    for suffix in [s.upper() for s in PMU_MEASUREMENT_SUFFIXES]:
        payload["per_pmu_channel"][f"BUS10::{suffix}"] = {
            "robust_sigma": 0.01 if suffix not in {"FREQ", "ROCOF"} else 0.002,
            "std": 0.01,
            "student_t_df": 8.0,
        }
    payload["channel_family_summary"] = {
        "voltage": {"robust_sigma_mean": 0.01, "student_t_df_mean": 8.0},
        "current": {"robust_sigma_mean": 0.01, "student_t_df_mean": 8.0},
        "frequency": {"robust_sigma_mean": 0.002, "student_t_df_mean": 8.0},
        "rocof": {"robust_sigma_mean": 0.002, "student_t_df_mean": 8.0},
    }
    return payload


def test_event7_corruption_process_changes_values_without_missingness() -> None:
    params = {
        "mode_prior": {"SPIKE": 0.4, "BIAS_DRIFT": 0.3, "STUCK": 0.2, "REPLAY_LIKE": 0.1},
        "spike_amplitude_distribution": {"mean": 4.0, "p95": 8.0},
        "spike_duration_distribution": {"mean": 3.0},
        "drift_slope_distribution": {"mean": 0.02, "std": 0.01},
        "stuck_run_length_distribution": {"mean": 4.0},
        "replay_segment_length_distribution": {"mean": 5.0},
        "channel_family_corruption_tendencies": {
            "voltage": {"jump_rate_mean": 0.06, "outlier_rate_mean": 0.02},
            "current": {"jump_rate_mean": 0.05, "outlier_rate_mean": 0.02},
            "frequency": {"jump_rate_mean": 0.04, "outlier_rate_mean": 0.02},
            "rocof": {"jump_rate_mean": 0.04, "outlier_rate_mean": 0.02},
        },
        "pmu_specific_corruption_tendencies": {"BUS10": {"jump_rate_mean": 0.07}},
    }
    process = Event7CorruptionProcess(params=params, noise_baseline=_noise_baseline(), seed=9)
    frame = _frame()
    original = frame.copy()
    updated, latent = process.apply(
        bus="BUS10",
        frame=frame,
        start_idx=30,
        end_idx=100,
        physical_event_mask=np.zeros((len(frame),), dtype=bool),
    )
    assert len(latent) > 0
    assert int((updated["Event"] == 7).sum()) > 0
    assert int((updated["DATA_PRESENT"] == 0).sum()) == 0
    changed = np.abs(updated["BUS10_VA_MAG"].to_numpy() - original["BUS10_VA_MAG"].to_numpy()) > 1e-12
    assert int(np.sum(changed)) > 0

