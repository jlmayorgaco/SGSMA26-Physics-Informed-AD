from __future__ import annotations

import pandas as pd

from src.simulation.m9.raw_informed_validation import compare_feature_tables


def test_raw_vs_sim_event5_metrics_table_has_required_fields() -> None:
    raw = pd.DataFrame(
        [
            {
                "data_present_zero_rate": 0.95,
                "nan_fraction_mean": 0.96,
                "full_dropout_rate": 0.90,
                "partial_dropout_rate": 0.05,
                "burst_length_mean": 12.0,
                "interburst_length_mean": 20.0,
                "outlier_rate_mean": 0.0,
                "jump_rate_mean": 0.0,
                "stuck_rate_mean": 0.0,
                "psd_lowfreq_ratio_mean": 0.0,
                "freq_std": 0.0,
                "rocof_std": 0.0,
                "angle_circular_var_mean": 0.0,
            }
        ]
    )
    sim = pd.DataFrame(
        [
            {
                "data_present_zero_rate": 0.88,
                "nan_fraction_mean": 0.89,
                "full_dropout_rate": 0.85,
                "partial_dropout_rate": 0.08,
                "burst_length_mean": 10.0,
                "interburst_length_mean": 18.0,
                "outlier_rate_mean": 0.0,
                "jump_rate_mean": 0.0,
                "stuck_rate_mean": 0.0,
                "psd_lowfreq_ratio_mean": 0.0,
                "freq_std": 0.0,
                "rocof_std": 0.0,
                "angle_circular_var_mean": 0.0,
            }
        ]
    )
    metrics = compare_feature_tables(raw_features=raw, sim_features=sim, event_id=5)
    assert not metrics.empty
    assert {"metric", "raw_mean", "sim_mean", "abs_delta", "pass_metric"}.issubset(metrics.columns)
    assert (metrics["event"] == 5).all()

