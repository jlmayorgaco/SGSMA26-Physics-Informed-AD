from __future__ import annotations

import pandas as pd

from src.simulation.m9.raw_informed_validation import compare_feature_tables


def test_raw_vs_sim_event7_metrics_table_has_required_fields() -> None:
    raw = pd.DataFrame(
        [
            {
                "data_present_zero_rate": 0.0,
                "nan_fraction_mean": 0.0,
                "full_dropout_rate": 0.0,
                "partial_dropout_rate": 0.0,
                "burst_length_mean": 0.0,
                "interburst_length_mean": 0.0,
                "outlier_rate_mean": 0.06,
                "jump_rate_mean": 0.15,
                "stuck_rate_mean": 0.04,
                "psd_lowfreq_ratio_mean": 0.62,
                "freq_std": 0.08,
                "rocof_std": 0.22,
                "angle_circular_var_mean": 0.16,
            }
        ]
    )
    sim = pd.DataFrame(
        [
            {
                "data_present_zero_rate": 0.0,
                "nan_fraction_mean": 0.0,
                "full_dropout_rate": 0.0,
                "partial_dropout_rate": 0.0,
                "burst_length_mean": 0.0,
                "interburst_length_mean": 0.0,
                "outlier_rate_mean": 0.05,
                "jump_rate_mean": 0.11,
                "stuck_rate_mean": 0.03,
                "psd_lowfreq_ratio_mean": 0.55,
                "freq_std": 0.06,
                "rocof_std": 0.19,
                "angle_circular_var_mean": 0.12,
            }
        ]
    )
    metrics = compare_feature_tables(raw_features=raw, sim_features=sim, event_id=7)
    assert not metrics.empty
    assert {"metric", "raw_mean", "sim_mean", "abs_delta", "pass_metric"}.issubset(metrics.columns)
    assert (metrics["event"] == 7).all()

