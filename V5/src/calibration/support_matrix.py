"""Support-matrix helpers for m3 artifact generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def unsupported_record(bus_id, spec, event_label=0):
    from src.calibration.calibration_metrics import signal_family_from_key
    from src.calibration.raw_chunk_loader import raw_col

    return {
        "bus_id": bus_id,
        "event_label": event_label,
        "event_name": "normal_operation",
        "signal_key": spec["signal_key"],
        "raw_column_used": raw_col(bus_id, spec["raw_suffix"]),
        "chunk_id": "ALL_CHUNK0",
        "scope": "aggregate",
        "signal_family": signal_family_from_key(spec["signal_key"]),
        "support_status": spec["support_status"],
        "profile_status": "not_used",
        "status": "unsupported",
        "quality_bucket": "invalid_mapping",
        "reason": spec["notes"],
    }


def write_support_matrix(df: pd.DataFrame, signal_specs: list[dict], output_dir: str | Path, active_statuses: set[str]) -> Path:
    active_agg = df[(df["status"].isin(active_statuses)) & (df["scope"] == "aggregate")]
    rows = []
    for spec in signal_specs:
        sig_rows = active_agg[active_agg["signal_key"] == spec["signal_key"]]
        mean_ks = float(sig_rows["ks_stat"].mean()) if not sig_rows.empty else np.nan
        mean_score = float(sig_rows["composite_score"].mean()) if not sig_rows.empty else np.nan
        supported = spec["support_status"] not in {"unsupported_raw_absolute", "unsupported_physics", "unsupported_mapping"}
        recommended = bool(supported and np.isfinite(mean_ks) and mean_ks < 0.15 and mean_score < 0.30 and spec["recommended"])
        rows.append(
            {
                "signal_name": spec["signal_key"],
                "raw_representation": spec["raw_representation"],
                "supported": "yes" if supported else "no",
                "support_status": spec["support_status"],
                "reason": spec["notes"],
                "recommended_for_training": "yes" if recommended else "no",
                "mean_ks_event0": mean_ks,
                "mean_composite_score_event0": mean_score,
                "notes": spec["notes"],
            }
        )
    out_path = Path(output_dir) / "signal_support_matrix.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
