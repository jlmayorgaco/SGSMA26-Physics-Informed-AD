"""RAW-vs-SIM validation for RAW-informed Event 5/Event 7 cyber processes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES


ANGLE_CHANNELS = [suffix.upper() for suffix in PMU_MEASUREMENT_SUFFIXES if suffix.endswith("ANG")]
ALL_CHANNELS = [suffix.upper() for suffix in PMU_MEASUREMENT_SUFFIXES]


def _safe_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    return float(np.mean(array))


def _safe_std(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    return float(np.std(array))


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    flags = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(flags):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def _run_lengths(mask: np.ndarray) -> list[int]:
    return [end - start + 1 for start, end in _contiguous_runs(mask)]


def _interburst_lengths(mask: np.ndarray) -> list[int]:
    runs = _contiguous_runs(mask)
    if len(runs) <= 1:
        return []
    return [max(0, right[0] - left[1] - 1) for left, right in zip(runs[:-1], runs[1:])]


def _wrapped_deg(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return ((array + 180.0) % 360.0) - 180.0


def _series_jump_rate(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 3:
        return 0.0
    diff = np.abs(np.diff(array))
    threshold = float(np.quantile(diff, 0.95))
    if threshold <= 1e-12:
        return 0.0
    return float(np.mean(diff >= threshold))


def _series_outlier_rate(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 3:
        return 0.0
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    sigma = max(1e-6, 1.4826 * mad)
    z = np.abs((array - median) / sigma)
    return float(np.mean(z > 3.5))


def _series_stuck_rate(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 3:
        return 0.0
    diffs = np.abs(np.diff(array))
    scale = float(np.quantile(np.abs(array - np.median(array)), 0.75)) if array.size else 1.0
    tol = max(1e-9, 0.001 * max(scale, 1e-6))
    return float(np.mean(diffs <= tol))


def _psd_lowfreq_ratio(values: np.ndarray, dt: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 8 or dt <= 0.0 or not math.isfinite(dt):
        return 0.0
    centered = array - np.mean(array)
    spec = np.fft.rfft(centered)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(array.size, d=dt)
    total = float(np.sum(power))
    if total <= 1e-12:
        return 0.0
    low_mask = freqs <= min(0.5, 0.5 / dt)
    if not np.any(low_mask):
        return 0.0
    return float(np.sum(power[low_mask]) / total)


def _angle_circular_variance(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 3:
        return 0.0
    radians = np.deg2rad(_wrapped_deg(array))
    sin_mean = float(np.mean(np.sin(radians)))
    cos_mean = float(np.mean(np.cos(radians)))
    resultant = math.sqrt(sin_mean * sin_mean + cos_mean * cos_mean)
    return float(1.0 - resultant)


def _parse_chunk_name(name: str) -> tuple[int, int]:
    parts = name.split("_")
    if len(parts) < 3:
        return 0, 0
    try:
        chunk_index = int(parts[0].replace("chunk", ""))
        event = int(parts[2])
        return chunk_index, event
    except Exception:
        return 0, 0


def _canonicalize_bus_frame(frame: pd.DataFrame, *, bus: str) -> pd.DataFrame:
    out = frame.copy()
    prefix = f"{bus.upper()}_"
    renamed: dict[str, str] = {}
    for column in out.columns:
        upper = str(column).upper()
        if upper.startswith(prefix):
            renamed[column] = upper[len(prefix) :]
        else:
            renamed[column] = upper
    out = out.rename(columns=renamed)
    required = ["TIMESTAMP", *ALL_CHANNELS, "DATA_PRESENT", "EVENT"]
    for column in required:
        if column not in out.columns:
            out[column] = np.nan
    out = out[required].copy()
    out["TIMESTAMP"] = pd.to_numeric(out["TIMESTAMP"], errors="coerce")
    out = out.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP").reset_index(drop=True)
    for channel in ALL_CHANNELS:
        out[channel] = pd.to_numeric(out[channel], errors="coerce")
    out["DATA_PRESENT"] = pd.to_numeric(out["DATA_PRESENT"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["EVENT"] = pd.to_numeric(out["EVENT"], errors="coerce").fillna(0).astype(int)
    return out


def load_raw_event_table(chunks_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for chunk_dir in sorted([path for path in chunks_root.iterdir() if path.is_dir()]):
        chunk_idx, event_folder = _parse_chunk_name(chunk_dir.name)
        for csv_path in sorted(chunk_dir.glob("Bus*.csv")):
            bus = csv_path.stem.upper()
            frame = _canonicalize_bus_frame(pd.read_csv(csv_path), bus=bus)
            frame["BUS"] = bus
            frame["CHUNK_NAME"] = chunk_dir.name
            frame["CHUNK_INDEX"] = chunk_idx
            frame["EVENT_FOLDER"] = event_folder
            rows.append(frame)
    if not rows:
        raise RuntimeError(f"No raw chunk CSVs found under {chunks_root}")
    return pd.concat(rows, ignore_index=True)


def load_sim_event_table(*, generated_manifest_path: Path, workspace_root: Path) -> pd.DataFrame:
    payload = json.loads(generated_manifest_path.read_text(encoding="utf-8"))
    rows: list[pd.DataFrame] = []
    for row in payload.get("scenarios", []):
        scenario_id = str(row.get("scenario_id"))
        scenario_dir = workspace_root / str(row.get("scenario_dir", ""))
        pmu_dir = scenario_dir / "pmu"
        for csv_path in sorted(pmu_dir.glob("Bus*_Competition_Data_sim.csv")):
            bus_token = "".join(ch for ch in csv_path.stem.split("_", maxsplit=1)[0] if ch.isdigit())
            bus = f"BUS{int(bus_token)}" if bus_token else "BUSUNK"
            frame = _canonicalize_bus_frame(pd.read_csv(csv_path), bus=bus)
            frame["BUS"] = bus
            frame["CHUNK_NAME"] = scenario_id
            frame["CHUNK_INDEX"] = 0
            frame["EVENT_FOLDER"] = int(row.get("event_coarse", 0))
            frame["TARGETED_FAMILY"] = str(row.get("targeted_family", ""))
            frame["TARGETED_VARIANT"] = str(row.get("targeted_variant", ""))
            rows.append(frame)
    if not rows:
        raise RuntimeError(f"No generated scenario PMU files found from manifest {generated_manifest_path}")
    return pd.concat(rows, ignore_index=True)


def compute_feature_rows(table: pd.DataFrame, *, event_id: int) -> pd.DataFrame:
    selected = table.loc[table["EVENT"] == int(event_id)].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "chunk_name",
                "bus",
                "data_present_zero_rate",
                "nan_fraction_mean",
                "full_dropout_rate",
                "partial_dropout_rate",
                "burst_length_mean",
                "interburst_length_mean",
                "outlier_rate_mean",
                "jump_rate_mean",
                "stuck_rate_mean",
                "psd_lowfreq_ratio_mean",
                "freq_std",
                "rocof_std",
                "angle_circular_var_mean",
            ]
        )
    rows: list[dict[str, Any]] = []
    for (chunk_name, bus), group in selected.groupby(["CHUNK_NAME", "BUS"], sort=True):
        ordered = group.sort_values("TIMESTAMP").reset_index(drop=True)
        ts = ordered["TIMESTAMP"].to_numpy(dtype=float)
        dt = np.diff(ts)
        dt = dt[np.isfinite(dt) & (dt > 0.0)]
        dt_median = float(np.median(dt)) if dt.size else 0.033
        row_nan = ordered[ALL_CHANNELS].isna().mean(axis=1).to_numpy(dtype=float)
        full_mask = row_nan >= 0.95
        partial_mask = (row_nan > 0.0) & (row_nan < 0.95)
        dp_zero = ordered["DATA_PRESENT"].to_numpy(dtype=float) < 0.5
        bursts = _run_lengths(dp_zero)
        inter = _interburst_lengths(dp_zero)
        outlier_rates: list[float] = []
        jump_rates: list[float] = []
        stuck_rates: list[float] = []
        psd_low: list[float] = []
        angle_vars: list[float] = []
        for channel in ALL_CHANNELS:
            values = ordered[channel].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size < 4:
                continue
            outlier_rates.append(_series_outlier_rate(values))
            jump_rates.append(_series_jump_rate(values))
            stuck_rates.append(_series_stuck_rate(values))
            psd_low.append(_psd_lowfreq_ratio(values, dt=dt_median))
            if channel in ANGLE_CHANNELS:
                angle_vars.append(_angle_circular_variance(values))
        rows.append(
            {
                "chunk_name": chunk_name,
                "bus": bus,
                "data_present_zero_rate": _safe_mean(dp_zero.astype(float)),
                "nan_fraction_mean": _safe_mean(row_nan),
                "full_dropout_rate": _safe_mean(full_mask.astype(float)),
                "partial_dropout_rate": _safe_mean(partial_mask.astype(float)),
                "burst_length_mean": _safe_mean(np.asarray(bursts, dtype=float)),
                "interburst_length_mean": _safe_mean(np.asarray(inter, dtype=float)),
                "outlier_rate_mean": _safe_mean(np.asarray(outlier_rates, dtype=float)),
                "jump_rate_mean": _safe_mean(np.asarray(jump_rates, dtype=float)),
                "stuck_rate_mean": _safe_mean(np.asarray(stuck_rates, dtype=float)),
                "psd_lowfreq_ratio_mean": _safe_mean(np.asarray(psd_low, dtype=float)),
                "freq_std": _safe_std(ordered["FREQ"].to_numpy(dtype=float)),
                "rocof_std": _safe_std(ordered["ROCOF"].to_numpy(dtype=float)),
                "angle_circular_var_mean": _safe_mean(np.asarray(angle_vars, dtype=float)),
            }
        )
    return pd.DataFrame(rows)


EVENT5_THRESHOLDS = {
    "data_present_zero_rate": 0.20,
    "nan_fraction_mean": 0.20,
    "full_dropout_rate": 0.20,
    "partial_dropout_rate": 0.20,
    "burst_length_mean": 8.0,
    "interburst_length_mean": 12.0,
}
EVENT7_THRESHOLDS = {
    "outlier_rate_mean": 0.06,
    "jump_rate_mean": 0.10,
    "stuck_rate_mean": 0.10,
    "psd_lowfreq_ratio_mean": 0.25,
    "freq_std": 0.08,
    "rocof_std": 0.25,
    "angle_circular_var_mean": 0.10,
}


def compare_feature_tables(
    *,
    raw_features: pd.DataFrame,
    sim_features: pd.DataFrame,
    event_id: int,
) -> pd.DataFrame:
    if event_id == 5:
        metric_names = [
            "data_present_zero_rate",
            "nan_fraction_mean",
            "full_dropout_rate",
            "partial_dropout_rate",
            "burst_length_mean",
            "interburst_length_mean",
            "outlier_rate_mean",
            "jump_rate_mean",
            "stuck_rate_mean",
            "psd_lowfreq_ratio_mean",
            "freq_std",
            "rocof_std",
            "angle_circular_var_mean",
        ]
        thresholds = EVENT5_THRESHOLDS
    else:
        metric_names = [
            "data_present_zero_rate",
            "nan_fraction_mean",
            "full_dropout_rate",
            "partial_dropout_rate",
            "burst_length_mean",
            "interburst_length_mean",
            "outlier_rate_mean",
            "jump_rate_mean",
            "stuck_rate_mean",
            "psd_lowfreq_ratio_mean",
            "freq_std",
            "rocof_std",
            "angle_circular_var_mean",
        ]
        thresholds = EVENT7_THRESHOLDS

    rows: list[dict[str, Any]] = []
    for metric in metric_names:
        raw_vals = raw_features.get(metric, pd.Series(dtype=float)).to_numpy(dtype=float)
        sim_vals = sim_features.get(metric, pd.Series(dtype=float)).to_numpy(dtype=float)
        raw_mean = _safe_mean(raw_vals)
        sim_mean = _safe_mean(sim_vals)
        abs_delta = abs(sim_mean - raw_mean)
        rel_delta = abs_delta / max(abs(raw_mean), 1e-6)
        threshold = float(thresholds.get(metric, np.nan))
        passes = bool(abs_delta <= threshold) if np.isfinite(threshold) else True
        rows.append(
            {
                "event": int(event_id),
                "metric": metric,
                "raw_mean": raw_mean,
                "sim_mean": sim_mean,
                "abs_delta": abs_delta,
                "rel_delta": rel_delta,
                "acceptance_threshold_abs": threshold,
                "pass_metric": passes,
                "raw_std": _safe_std(raw_vals),
                "sim_std": _safe_std(sim_vals),
            }
        )
    frame = pd.DataFrame(rows).sort_values("abs_delta").reset_index(drop=True)
    return frame


def _plot_event_panels(
    *,
    event_id: int,
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    top = metrics.nsmallest(8, "abs_delta")
    bars = np.arange(len(top))
    axes[0].bar(bars - 0.2, top["raw_mean"], width=0.4, label="RAW", color="#1f77b4")
    axes[0].bar(bars + 0.2, top["sim_mean"], width=0.4, label="SIM", color="#ff7f0e")
    axes[0].set_xticks(bars, top["metric"], rotation=45, ha="right")
    axes[0].set_title("Closest Match Metrics")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    worst = metrics.nlargest(8, "abs_delta")
    bars2 = np.arange(len(worst))
    axes[1].bar(bars2 - 0.2, worst["raw_mean"], width=0.4, label="RAW", color="#1f77b4")
    axes[1].bar(bars2 + 0.2, worst["sim_mean"], width=0.4, label="SIM", color="#ff7f0e")
    axes[1].set_xticks(bars2, worst["metric"], rotation=45, ha="right")
    axes[1].set_title("Largest Gap Metrics")
    axes[1].grid(alpha=0.2)

    pass_rate = float(metrics["pass_metric"].mean()) if not metrics.empty else 0.0
    axes[2].bar(["pass", "fail"], [pass_rate, 1.0 - pass_rate], color=["#2ca02c", "#d62728"])
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("Metric Pass Rate")
    axes[2].grid(alpha=0.2)

    axes[3].hist(metrics["abs_delta"].to_numpy(dtype=float), bins=12, color="#9467bd", alpha=0.85)
    axes[3].set_title("Absolute Delta Distribution")
    axes[3].set_xlabel("abs(raw_mean - sim_mean)")
    axes[3].grid(alpha=0.2)

    fig.suptitle(f"RAW vs SIM Event {event_id} Match Panels")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@dataclass(slots=True)
class EventMatchSummary:
    event_id: int
    pass_rate: float
    acceptable_for_training: bool
    strongest_matches: list[str]
    largest_gaps: list[str]


def _build_event_match_summary(event_id: int, metrics: pd.DataFrame) -> EventMatchSummary:
    pass_rate = float(metrics["pass_metric"].mean()) if not metrics.empty else 0.0
    strongest = metrics.nsmallest(5, "abs_delta")["metric"].astype(str).tolist()
    gaps = metrics.nlargest(5, "abs_delta")["metric"].astype(str).tolist()
    # Event 7 is harder because RAW support is sparse; use a conservative threshold.
    threshold = 0.60 if event_id == 5 else 0.50
    acceptable = bool(pass_rate >= threshold)
    return EventMatchSummary(
        event_id=int(event_id),
        pass_rate=pass_rate,
        acceptable_for_training=acceptable,
        strongest_matches=strongest,
        largest_gaps=gaps,
    )


def _write_event_report(
    *,
    event_id: int,
    metrics: pd.DataFrame,
    summary: EventMatchSummary,
    output_path: Path,
) -> None:
    lines = [
        f"# Event {event_id} RAW vs SIM Match Report",
        "",
        f"- pass_rate: `{summary.pass_rate:.4f}`",
        f"- acceptable_for_training_realism: `{summary.acceptable_for_training}`",
        "",
        "## What Matches RAW Well",
    ]
    for metric in summary.strongest_matches:
        row = metrics.loc[metrics["metric"] == metric].iloc[0]
        lines.append(
            "- "
            + f"{metric}: raw_mean={float(row['raw_mean']):.6f}, sim_mean={float(row['sim_mean']):.6f}, abs_delta={float(row['abs_delta']):.6f}"
        )
    lines.extend(["", "## What Still Differs"])
    for metric in summary.largest_gaps:
        row = metrics.loc[metrics["metric"] == metric].iloc[0]
        lines.append(
            "- "
            + f"{metric}: raw_mean={float(row['raw_mean']):.6f}, sim_mean={float(row['sim_mean']):.6f}, abs_delta={float(row['abs_delta']):.6f}"
        )
    lines.extend(
        [
            "",
            "## Acceptability Statement",
            "- "
            + (
                "Process appears acceptable for detector training realism on this event family."
                if summary.acceptable_for_training
                else "Process still differs from RAW and needs further calibration before relying on transfer."
            ),
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_raw_vs_sim(
    *,
    chunks_root: Path,
    generated_manifest_path: Path,
    workspace_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    metrics_dir = output_root / "metrics"
    report_dir = output_root / "report"
    plots_dir = output_root / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw_table = load_raw_event_table(chunks_root)
    sim_table = load_sim_event_table(generated_manifest_path=generated_manifest_path, workspace_root=workspace_root)
    sim_e5 = sim_table.loc[sim_table["TARGETED_FAMILY"] == "RAW_INFORMED_EVENT5"].copy()
    sim_e7 = sim_table.loc[sim_table["TARGETED_FAMILY"] == "RAW_INFORMED_EVENT7"].copy()

    raw_event5_features = compute_feature_rows(raw_table, event_id=5)
    raw_event7_features = compute_feature_rows(raw_table, event_id=7)
    sim_event5_features = compute_feature_rows(sim_e5, event_id=5)
    sim_event7_features = compute_feature_rows(sim_e7, event_id=7)

    event5_metrics = compare_feature_tables(raw_features=raw_event5_features, sim_features=sim_event5_features, event_id=5)
    event7_metrics = compare_feature_tables(raw_features=raw_event7_features, sim_features=sim_event7_features, event_id=7)

    event5_path = metrics_dir / "raw_vs_sim_event5_metrics.csv"
    event7_path = metrics_dir / "raw_vs_sim_event7_metrics.csv"
    event5_metrics.to_csv(event5_path, index=False)
    event7_metrics.to_csv(event7_path, index=False)

    event5_summary = _build_event_match_summary(5, event5_metrics)
    event7_summary = _build_event_match_summary(7, event7_metrics)
    _write_event_report(
        event_id=5,
        metrics=event5_metrics,
        summary=event5_summary,
        output_path=report_dir / "event5_raw_match_report.md",
    )
    _write_event_report(
        event_id=7,
        metrics=event7_metrics,
        summary=event7_summary,
        output_path=report_dir / "event7_raw_match_report.md",
    )
    _plot_event_panels(event_id=5, metrics=event5_metrics, output_path=plots_dir / "event5_raw_vs_sim_panels.png")
    _plot_event_panels(event_id=7, metrics=event7_metrics, output_path=plots_dir / "event7_raw_vs_sim_panels.png")

    return {
        "event5_metrics_path": str(event5_path),
        "event7_metrics_path": str(event7_path),
        "event5_summary": {
            "pass_rate": event5_summary.pass_rate,
            "acceptable_for_training": event5_summary.acceptable_for_training,
            "strongest_matches": event5_summary.strongest_matches,
            "largest_gaps": event5_summary.largest_gaps,
        },
        "event7_summary": {
            "pass_rate": event7_summary.pass_rate,
            "acceptable_for_training": event7_summary.acceptable_for_training,
            "strongest_matches": event7_summary.strongest_matches,
            "largest_gaps": event7_summary.largest_gaps,
        },
    }

