from __future__ import annotations

import numpy as np
import pandas as pd

from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES
from src.simulation.m9.raw_informed_cyber_layer import CyberIntervalSpec, RawInformedCyberLayer


def _frame(bus: str, n: int = 120) -> pd.DataFrame:
    data = {
        "TIMESTAMP": np.arange(n, dtype=float) * 0.033,
        "DATA_PRESENT": np.ones((n,), dtype=int),
        "Event": np.zeros((n,), dtype=int),
    }
    for suffix in PMU_MEASUREMENT_SUFFIXES:
        data[f"{bus}_{suffix}"] = 1.0 + 0.001 * np.arange(n)
    return pd.DataFrame(data)


def _event5_params() -> dict:
    return {
        "state_transition_matrix": {
            "NORMAL": {"NORMAL": 0.6, "PARTIAL_DROPOUT": 0.2, "FULL_DROPOUT": 0.2},
            "PARTIAL_DROPOUT": {"NORMAL": 0.3, "PARTIAL_DROPOUT": 0.4, "FULL_DROPOUT": 0.3},
            "FULL_DROPOUT": {"NORMAL": 0.3, "PARTIAL_DROPOUT": 0.2, "FULL_DROPOUT": 0.5},
        },
        "dropout_burst_length_distribution": {"mean": 8.0, "p50": 6.0},
        "inter_burst_interval_distribution": {"mean": 12.0},
        "full_vs_partial_dropout_fraction": {"full": 0.7, "partial": 0.3},
        "channel_family_dropout_tendencies": {
            "voltage": {"nan_fraction_mean": 0.4},
            "current": {"nan_fraction_mean": 0.4},
            "frequency": {"nan_fraction_mean": 0.2},
            "rocof": {"nan_fraction_mean": 0.2},
        },
        "pmu_specific_dropout_tendencies": {"BUS29": {"dropout_burst_mean_frames": 8.0, "interburst_mean_frames": 10.0}},
    }


def _event7_params() -> dict:
    return {
        "mode_prior": {"SPIKE": 0.35, "BIAS_DRIFT": 0.25, "STUCK": 0.20, "REPLAY_LIKE": 0.20},
        "spike_amplitude_distribution": {"mean": 4.0, "p95": 7.0},
        "spike_duration_distribution": {"mean": 3.0},
        "drift_slope_distribution": {"mean": 0.015, "std": 0.008},
        "stuck_run_length_distribution": {"mean": 5.0},
        "replay_segment_length_distribution": {"mean": 4.0},
        "channel_family_corruption_tendencies": {
            "voltage": {"jump_rate_mean": 0.05, "outlier_rate_mean": 0.01},
            "current": {"jump_rate_mean": 0.05, "outlier_rate_mean": 0.01},
            "frequency": {"jump_rate_mean": 0.05, "outlier_rate_mean": 0.01},
            "rocof": {"jump_rate_mean": 0.05, "outlier_rate_mean": 0.01},
        },
        "pmu_specific_corruption_tendencies": {"BUS10": {"jump_rate_mean": 0.06}},
    }


def _event0_noise() -> dict:
    per_channel = {}
    for bus in ["BUS29", "BUS10"]:
        for suffix in [s.upper() for s in PMU_MEASUREMENT_SUFFIXES]:
            per_channel[f"{bus}::{suffix}"] = {"robust_sigma": 0.01, "std": 0.01, "student_t_df": 8.0}
    return {
        "per_pmu_channel": per_channel,
        "channel_family_summary": {
            "voltage": {"robust_sigma_mean": 0.01, "student_t_df_mean": 8.0},
            "current": {"robust_sigma_mean": 0.01, "student_t_df_mean": 8.0},
            "frequency": {"robust_sigma_mean": 0.01, "student_t_df_mean": 8.0},
            "rocof": {"robust_sigma_mean": 0.01, "student_t_df_mean": 8.0},
        },
    }


def test_raw_informed_cyber_layer_applies_event5_to_target_bus() -> None:
    layer = RawInformedCyberLayer(
        event5_params=_event5_params(),
        event7_params=_event7_params(),
        event0_noise_baseline=_event0_noise(),
        seed=21,
        apply_shared_noise=False,
    )
    frames = {"BUS29": _frame("BUS29"), "BUS10": _frame("BUS10")}
    intervals = [
        CyberIntervalSpec(
            kind="event5",
            start_time_s=1.0,
            end_time_s=2.0,
            target_pmus=["BUS29"],
            subtype="dropout",
            event_label_if_no_physical=5,
            event_label_if_physical=6,
        )
    ]
    updated, latent = layer.apply_to_frames(pmu_frames=frames, intervals=intervals)
    assert not latent.empty
    assert int((updated["BUS29"]["Event"] == 5).sum()) > 0
    assert int((updated["BUS29"].filter(like="BUS29_").isna().any(axis=1)).sum()) > 0
    assert int((updated["BUS10"]["Event"] != 0).sum()) == 0

