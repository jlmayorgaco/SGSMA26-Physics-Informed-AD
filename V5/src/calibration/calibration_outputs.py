"""Writers for m3 calibration artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.support_matrix import write_support_matrix


def summarize_active(active: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if active.empty:
        return pd.DataFrame()
    return (
        active.groupby(group_cols, dropna=False)
        .agg(
            n=("signal_key", "count"),
            ks_mean=("ks_stat", "mean"),
            ks_median=("ks_stat", "median"),
            composite_score_mean=("composite_score", "mean"),
            relative_mean_error_mean=("relative_mean_error", "mean"),
            relative_std_error_mean=("relative_std_error", "mean"),
            psd_lowfreq_mismatch_mean=("psd_lowfreq_mismatch", "mean"),
            spectral_centroid_error_mean=("spectral_centroid_error", "mean"),
            chunk_stability_penalty_mean=("chunk_stability_penalty", "mean"),
            very_good_pct=("quality_bucket", lambda s: float(np.mean(s == "very_good") * 100.0)),
            usable_pct=("quality_bucket", lambda s: float(np.mean(s == "usable") * 100.0)),
            needs_tuning_pct=("quality_bucket", lambda s: float(np.mean(s == "needs_tuning") * 100.0)),
            poor_pct=("quality_bucket", lambda s: float(np.mean(s == "poor") * 100.0)),
        )
        .reset_index()
    )


def write_outputs(records, output_dir, event_label, drift_mode, active_statuses, signal_specs):
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    long_path = metrics_dir / "calibration_metrics_long.csv"
    df.to_csv(long_path, index=False)

    active = df[df["status"].isin(active_statuses)].copy()
    summarize_active(active, ["bus_id"]).to_csv(metrics_dir / "calibration_summary_by_bus.csv", index=False)
    summarize_active(active, ["signal_key"]).to_csv(metrics_dir / "calibration_summary_by_signal.csv", index=False)
    summarize_active(active[active["scope"] == "per_chunk"], ["chunk_id", "bus_id", "signal_key"]).to_csv(
        metrics_dir / "calibration_summary_chunk0.csv", index=False
    )
    summarize_active(active, ["signal_family"]).to_csv(metrics_dir / "calibration_summary_signal_family.csv", index=False)
    results_path = metrics_dir / "calibration_results.json"
    results_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "event_label": event_label,
                "drift_mode": drift_mode,
                "records": records,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_support_matrix(df, signal_specs, output_dir, active_statuses)
    return df
