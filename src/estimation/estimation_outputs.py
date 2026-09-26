"""Export helpers for m4 estimated outputs and reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import pandas as pd

from src.estimation.estimation_metrics import compute_estimation_metrics
from src.estimation.estimation_plots import generate_estimated_bus_plots

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PMU_BUSES = {"39", "29", "10", "22", "19", "2", "5", "6"}


def _quality_percent_from_summary(summary_bus: pd.DataFrame) -> pd.Series:
    """Compute bus quality percentage from relative RMSE and correlation."""
    rel = pd.to_numeric(summary_bus["relative_rmse"], errors="coerce").fillna(1.0)
    corr = pd.to_numeric(summary_bus["corr"], errors="coerce").fillna(0.0)
    rel_score = (1.0 - rel).clip(lower=0.0, upper=1.0) * 100.0
    corr_score = ((corr + 1.0) / 2.0).clip(lower=0.0, upper=1.0) * 100.0
    return 0.7 * rel_score + 0.3 * corr_score


def generate_bus_comparison_plots(summary_bus: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Generate % quality and % relative error bar plots per bus."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if summary_bus.empty:
        return []

    frame = summary_bus.copy()
    frame["bus_id"] = frame["bus_id"].astype(str)
    frame = frame.sort_values("bus_id", key=lambda s: s.astype(int))
    frame["quality_percent"] = _quality_percent_from_summary(frame)
    rel_error_percent = (pd.to_numeric(frame["relative_rmse"], errors="coerce").fillna(1.0).clip(lower=0.0) * 100.0).clip(
        upper=300.0
    )

    paths: list[Path] = []

    fig1, ax1 = plt.subplots(figsize=(12, 4.8))
    ax1.bar(frame["bus_id"], frame["quality_percent"], color="#2a9d8f")
    ax1.set_ylim(0.0, 100.0)
    ax1.set_xlabel("Bus ID")
    ax1.set_ylabel("Quality (%)")
    ax1.set_title("Estimated vs Simulated: Bus Quality Score (%)")
    ax1.grid(axis="y", alpha=0.25)
    q_path = out_dir / "estimation_bus_quality_percent.png"
    fig1.tight_layout()
    fig1.savefig(q_path, dpi=140)
    plt.close(fig1)
    paths.append(q_path)

    fig2, ax2 = plt.subplots(figsize=(12, 4.8))
    ax2.bar(frame["bus_id"], rel_error_percent, color="#e76f51")
    ax2.set_xlabel("Bus ID")
    ax2.set_ylabel("Relative RMSE (%)")
    ax2.set_title("Estimated vs Simulated: Relative Error by Bus (%)")
    ax2.grid(axis="y", alpha=0.25)
    e_path = out_dir / "estimation_bus_relative_rmse_percent.png"
    fig2.tight_layout()
    fig2.savefig(e_path, dpi=140)
    plt.close(fig2)
    paths.append(e_path)

    return paths


def _signal_error_percent(series: pd.Series) -> pd.Series:
    rel = pd.to_numeric(series, errors="coerce").fillna(1.0)
    return (rel.clip(lower=0.0) * 100.0).clip(upper=300.0)


def _plot_family_grouped_percent(
    metrics_long: pd.DataFrame,
    signals: list[str],
    title: str,
    filename: str,
    output_dir: Path,
) -> Path | None:
    subset = metrics_long[metrics_long["signal"].isin(signals)].copy()
    if subset.empty:
        return None
    subset["bus_id"] = subset["bus_id"].astype(str)
    subset["error_percent"] = _signal_error_percent(subset["relative_rmse"])
    pivot = (
        subset.pivot_table(index="bus_id", columns="signal", values="error_percent", aggfunc="mean")
        .reindex(columns=signals)
        .sort_index(key=lambda s: s.astype(int))
    )
    if pivot.empty:
        return None

    buses = list(pivot.index)
    x = range(len(buses))
    n = len(signals)
    width = 0.8 / max(n, 1)

    fig, ax = plt.subplots(figsize=(13, 5.2))
    for i, sig in enumerate(signals):
        offsets = [xi - 0.4 + width / 2 + i * width for xi in x]
        values = pivot[sig].fillna(0.0).to_numpy(dtype=float)
        colors = ["#d62828" if b in PMU_BUSES else "#457b9d" for b in buses]
        ax.bar(offsets, values, width=width, label=sig, color=colors, alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(buses)
    ax.set_xlabel("Bus ID")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=min(4, n), fontsize=8)

    # simple legend note for node coloring
    ax.text(0.01, 0.98, "Red = PMU bus, Blue = non-PMU bus", transform=ax.transAxes, va="top", fontsize=8)

    out_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def generate_signal_family_comparison_plots(metrics_long: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    """Generate per-family by-node error-percent comparison plots."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if metrics_long.empty:
        return []
    required = {"bus_id", "signal", "relative_rmse"}
    if not required.issubset(metrics_long.columns):
        return []

    plots: list[Path] = []
    specs = [
        (["IA_MAG", "IB_MAG", "IC_MAG"], "Current Magnitude Error by Bus (%)", "comparison_current_mag_percent_by_bus.png"),
        (["VA_MAG", "VB_MAG", "VC_MAG"], "Voltage Magnitude Error by Bus (%)", "comparison_voltage_mag_percent_by_bus.png"),
        (["Freq"], "Frequency Error by Bus (%)", "comparison_frequency_percent_by_bus.png"),
        (["ROCOF"], "ROCOF Error by Bus (%)", "comparison_rocof_percent_by_bus.png"),
    ]
    for signals, title, filename in specs:
        path = _plot_family_grouped_percent(metrics_long, signals, title, filename, out_dir)
        if path is not None:
            plots.append(path)
    return plots


def generate_bus_comparison_plots_from_csv(summary_bus_csv: str | Path, output_dir: str | Path) -> list[Path]:
    """Load summary-by-bus CSV and generate comparison plots."""
    csv_path = Path(summary_bus_csv)
    if not csv_path.exists():
        return []
    summary_bus = pd.read_csv(csv_path)
    required = {"bus_id", "relative_rmse", "corr"}
    if not required.issubset(summary_bus.columns):
        return []
    return generate_bus_comparison_plots(summary_bus, output_dir)


def generate_signal_family_comparison_plots_from_csv(metrics_long_csv: str | Path, output_dir: str | Path) -> list[Path]:
    """Load metrics-long CSV and generate family comparison plots."""
    csv_path = Path(metrics_long_csv)
    if not csv_path.exists():
        return []
    metrics_long = pd.read_csv(csv_path)
    return generate_signal_family_comparison_plots(metrics_long, output_dir)


def export_estimation_report(
    fault_bus: str,
    simulation_dfs: dict[str, pd.DataFrame],
    estimated_dfs: dict[str, pd.DataFrame],
    output_dir: str | Path,
    estimation_mode: str = "ybus_temporal_regularized",
) -> Path:
    """Write estimation metrics and JSON report."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_long, summary_bus, summary_signal = compute_estimation_metrics(simulation_dfs, estimated_dfs)
    metrics_long.to_csv(out_dir / "estimation_metrics_long.csv", index=False)
    summary_bus.to_csv(out_dir / "estimation_summary_by_bus.csv", index=False)
    summary_signal.to_csv(out_dir / "estimation_summary_by_signal.csv", index=False)
    generate_bus_comparison_plots(summary_bus, out_dir)
    generate_signal_family_comparison_plots(metrics_long, out_dir)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fault_bus": str(fault_bus),
        "estimation_mode": estimation_mode,
        "summary_by_bus": summary_bus.to_dict(orient="records"),
        "summary_by_signal": summary_signal.to_dict(orient="records"),
    }
    report_path = out_dir / "estimation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def export_estimated_outputs(
    estimated_dfs: dict[str, pd.DataFrame],
    simulation_dfs: dict[str, pd.DataFrame],
    fault_bus: str,
    output_dir: str | Path,
    generate_plots: bool = True,
) -> Path:
    """Write estimated BUS csv outputs plus estimation report artifacts."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for bus_id in sorted(estimated_dfs.keys(), key=lambda x: int(x)):
        df = estimated_dfs[bus_id]
        df.to_csv(out_dir / f"BUS{bus_id}_Competition_Data_nanmask.csv", index=False)
        if generate_plots and bus_id in simulation_dfs and "TIMESTAMP" in df.columns and "Event" in df.columns:
            generate_estimated_bus_plots(
                bus_id=str(bus_id),
                t=df["TIMESTAMP"].to_numpy(dtype=float),
                event_arr=df["Event"].to_numpy(dtype=int),
                simulation_df=simulation_dfs[bus_id],
                estimated_df=df,
                base_out_dir=str(out_dir),
            )
    export_estimation_report(fault_bus, simulation_dfs, estimated_dfs, out_dir)
    return out_dir
