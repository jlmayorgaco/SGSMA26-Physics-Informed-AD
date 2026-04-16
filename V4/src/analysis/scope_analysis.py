
from typing import Any

from analysis.channel_analysis import compute_channel_analysis
from analysis.events import event_spans, summarize_event_spans
from analysis.integrity import timestamp_integrity
from analysis.missing import missing_profile
from analysis.power import compute_three_phase_power
from analysis.stats import basic_stats, phase_balance_metrics
from config.config import DEFAULT_EVENT_LABELS, AnalysisConfig


def analyze_scope(df: pd.DataFrame, scope_name: str, config: AnalysisConfig) -> dict[str, Any]:
    if df.empty:
        return {
            "scope": scope_name,
            "row_count": 0,
            "channels": {},
            "phase_balance": {},
            "missing_data": {},
            "integrity": {},
            "event_spans": [],
            "event_summary": {},
            "power_summary": {},
        }

    label_map = {int(k): v for k, v in DEFAULT_EVENT_LABELS.items()}
    dt = compute_dt(df)
    power_df = compute_three_phase_power(df)
    power_summary = {
        col: basic_stats(power_df[col].to_numpy(dtype=float), dt, col)
        for col in power_df.columns
        if col != "TIMESTAMP"
    }

    spans = event_spans(df, label_map)

    return {
        "scope": scope_name,
        "row_count": int(len(df)),
        "integrity": timestamp_integrity(df),
        "missing_data": missing_profile(df),
        "phase_balance": phase_balance_metrics(df),
        "event_spans": spans,
        "event_summary": summarize_event_spans(spans),
        "channels": compute_channel_analysis(
            df=df,
            label_map=label_map,
            event_baseline_seconds=config.event_baseline_seconds,
            event_post_seconds=config.event_post_seconds,
            trend_window_seconds=config.trend_window_seconds,
        ),
        "power_summary": power_summary,
    }
