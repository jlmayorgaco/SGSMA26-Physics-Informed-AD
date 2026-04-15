from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.physics.ybus_baseline import run_ybus_voltage_baseline


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


def _save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(data), file, indent=2, ensure_ascii=False)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _metric_block(df: pd.DataFrame) -> dict[str, float]:
    mag_true = df["mag_true"].to_numpy()
    mag_hat = df["mag_hat"].to_numpy()
    mag_error = df["mag_error"].to_numpy()
    mag_abs_error = df["mag_abs_error"].to_numpy()

    angle_error_deg = df["angle_error_deg"].to_numpy()
    angle_abs_error_deg = df["angle_abs_error_deg"].to_numpy()

    mean_mag_true = float(np.mean(mag_true))
    mag_rmse = float(np.sqrt(np.mean(np.square(mag_error))))
    mag_mae = float(np.mean(mag_abs_error))
    mag_nrmse_pct = float(100.0 * mag_rmse / mean_mag_true) if not np.isclose(mean_mag_true, 0.0) else float("nan")
    mag_mape_pct = float(100.0 * np.mean(mag_abs_error / np.maximum(np.abs(mag_true), 1e-12)))
    mag_r2 = _safe_r2(mag_true, mag_hat)

    angle_mae_deg = float(np.mean(angle_abs_error_deg))
    angle_rmse_deg = float(np.sqrt(np.mean(np.square(angle_error_deg))))

    return {
        "n_samples": int(len(df)),
        "mag_mean_true": mean_mag_true,
        "mag_rmse": mag_rmse,
        "mag_mae": mag_mae,
        "mag_nrmse_pct": mag_nrmse_pct,
        "mag_mape_pct": mag_mape_pct,
        "mag_r2": mag_r2,
        "angle_mae_deg": angle_mae_deg,
        "angle_rmse_deg": angle_rmse_deg,
    }


def _build_event_spans(time_df: pd.DataFrame) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    in_event = False
    start_time = 0.0

    timestamps = time_df["TIMESTAMP"].to_numpy()
    events = time_df["Event"].to_numpy()

    for idx, event_value in enumerate(events):
        is_event = int(event_value) != 0

        if is_event and not in_event:
            in_event = True
            start_time = float(timestamps[idx])

        if in_event and not is_event:
            end_time = float(timestamps[idx - 1])
            spans.append((start_time, end_time))
            in_event = False

    if in_event:
        spans.append((start_time, float(timestamps[-1])))

    return spans


def _add_event_shading(ax: plt.Axes, time_df: pd.DataFrame) -> None:
    for start_time, end_time in _build_event_spans(time_df):
        ax.axvspan(start_time, end_time, alpha=0.15)


def _save_per_bus_bar_plot(
    per_bus_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plot_df = per_bus_df.sort_values(metric_col).copy()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(plot_df["bus_id"].astype(str), plot_df[metric_col].to_numpy())
    ax.set_xlabel("Hidden bus")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_time_error_plot(
    results_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    time_df = (
        results_df.groupby("TIMESTAMP", as_index=False)
        .agg(
            Event=("Event", "first"),
            mean_value=(value_col, "mean"),
            median_value=(value_col, "median"),
            p90_value=(value_col, lambda x: float(np.percentile(x, 90))),
        )
        .sort_values("TIMESTAMP")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time_df["TIMESTAMP"], time_df["mean_value"], label="Mean")
    ax.plot(time_df["TIMESTAMP"], time_df["median_value"], label="Median")
    ax.plot(time_df["TIMESTAMP"], time_df["p90_value"], label="P90")
    _add_event_shading(ax, time_df)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_regime_boxplot(
    results_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    normal_values = results_df.loc[results_df["Event"] == 0, value_col].to_numpy()
    event_values = results_df.loc[results_df["Event"] != 0, value_col].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([normal_values, event_values], tick_labels=["Normal", "Event"], showfliers=False)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_scatter_plot(
    results_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    sample_size: int = 5000,
    random_seed: int = 42,
) -> None:
    plot_df = results_df[[x_col, y_col]].dropna().copy()

    if len(plot_df) > sample_size:
        plot_df = plot_df.sample(n=sample_size, random_state=random_seed)

    x = plot_df[x_col].to_numpy()
    y = plot_df[y_col].to_numpy()

    xy_min = float(min(np.min(x), np.min(y)))
    xy_max = float(max(np.max(x), np.max(y)))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=10, alpha=0.5)
    ax.plot([xy_min, xy_max], [xy_min, xy_max], linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_example_bus_plots(
    results_df: pd.DataFrame,
    per_bus_df: pd.DataFrame,
    output_dir: Path,
    n_best: int = 2,
    n_worst: int = 2,
) -> list[str]:
    saved_files: list[str] = []

    best_buses = per_bus_df.nsmallest(n_best, "mag_nrmse_pct")["bus_id"].tolist()
    worst_buses = per_bus_df.nlargest(n_worst, "mag_nrmse_pct")["bus_id"].tolist()
    example_buses = list(dict.fromkeys(best_buses + worst_buses))

    for bus_id in example_buses:
        bus_df = (
            results_df.loc[results_df["bus_id"] == bus_id]
            .sort_values("TIMESTAMP")
            .reset_index(drop=True)
        )

        time_df = bus_df[["TIMESTAMP", "Event"]].copy()

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        axes[0].plot(bus_df["TIMESTAMP"], bus_df["mag_true"], label="True magnitude")
        axes[0].plot(bus_df["TIMESTAMP"], bus_df["mag_hat"], label="Estimated magnitude")
        _add_event_shading(axes[0], time_df)
        axes[0].set_ylabel("Voltage magnitude")
        axes[0].set_title(f"Bus {bus_id} voltage magnitude")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(bus_df["TIMESTAMP"], bus_df["angle_true_deg"], label="True angle")
        axes[1].plot(bus_df["TIMESTAMP"], bus_df["angle_hat_deg"], label="Estimated angle")
        _add_event_shading(axes[1], time_df)
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Angle [deg]")
        axes[1].set_title(f"Bus {bus_id} voltage angle")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()

        output_path = output_dir / f"bus_{bus_id}_timeseries.png"
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        saved_files.append(output_path.name)

    return saved_files


def _build_per_bus_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for bus_id, bus_df in results_df.groupby("bus_id"):
        metrics = _metric_block(bus_df)
        metrics["bus_id"] = int(bus_id)
        rows.append(metrics)

    return (
        pd.DataFrame(rows)
        .sort_values(["mag_nrmse_pct", "angle_mae_deg"], ascending=[True, True])
        .reset_index(drop=True)
    )


def _build_report(
    scenario_id: str,
    representation: str,
    results_df: pd.DataFrame,
    per_bus_df: pd.DataFrame,
    baseline_summary: Any,
    output_dir: Path,
    saved_plot_files: list[str],
) -> dict[str, Any]:
    overall_metrics = _metric_block(results_df)
    normal_metrics = _metric_block(results_df.loc[results_df["Event"] == 0])
    event_metrics = _metric_block(results_df.loc[results_df["Event"] != 0])

    best_buses = per_bus_df.nsmallest(5, "mag_nrmse_pct")[["bus_id", "mag_nrmse_pct", "angle_mae_deg"]]
    worst_buses = per_bus_df.nlargest(5, "mag_nrmse_pct")[["bus_id", "mag_nrmse_pct", "angle_mae_deg"]]

    report = {
        "scenario_id": scenario_id,
        "representation": representation,
        "baseline_summary": asdict(baseline_summary),
        "global_metrics": {
            "overall": overall_metrics,
            "normal_only": normal_metrics,
            "event_only": event_metrics,
        },
        "per_bus_metrics": per_bus_df.to_dict(orient="records"),
        "ranking": {
            "best_buses_by_mag_nrmse_pct": best_buses.to_dict(orient="records"),
            "worst_buses_by_mag_nrmse_pct": worst_buses.to_dict(orient="records"),
        },
        "artifacts": {
            "output_dir": str(output_dir),
            "plots": saved_plot_files,
            "results_csv": "results_long.csv",
            "per_bus_csv": "per_bus_metrics.csv",
            "report_json": "report.json",
        },
    }

    return report


def run_and_save_ybus_baseline_report(
    scenario_id: str = "SIM_0001",
    representation: str = "positive_sequence",
    output_root: str | Path = "artifacts/ybus_baseline",
) -> dict[str, str]:
    """
    Run the Ybus baseline, compute statistical analysis, save PNG plots and JSON report.

    Returns
    -------
    dict[str, str]
        Paths to the main generated artifacts.
    """
    results_df, _, baseline_summary = run_ybus_voltage_baseline(
        scenario_id=scenario_id,
        representation=representation,
    )

    per_bus_df = _build_per_bus_metrics(results_df)

    output_dir = _ensure_dir(Path(output_root) / scenario_id / representation)
    plots_dir = _ensure_dir(output_dir / "plots")

    results_csv_path = output_dir / "results_long.csv"
    per_bus_csv_path = output_dir / "per_bus_metrics.csv"
    report_json_path = output_dir / "report.json"

    results_df.to_csv(results_csv_path, index=False)
    per_bus_df.to_csv(per_bus_csv_path, index=False)

    saved_plot_files: list[str] = []

    plot_specs = [
        (
            lambda: _save_per_bus_bar_plot(
                per_bus_df=per_bus_df,
                metric_col="mag_nrmse_pct",
                ylabel="Magnitude NRMSE [%]",
                title=f"{scenario_id} - Hidden-bus magnitude NRMSE",
                output_path=plots_dir / "per_bus_mag_nrmse_pct.png",
            ),
            "per_bus_mag_nrmse_pct.png",
        ),
        (
            lambda: _save_per_bus_bar_plot(
                per_bus_df=per_bus_df,
                metric_col="angle_mae_deg",
                ylabel="Angle MAE [deg]",
                title=f"{scenario_id} - Hidden-bus angle MAE",
                output_path=plots_dir / "per_bus_angle_mae_deg.png",
            ),
            "per_bus_angle_mae_deg.png",
        ),
        (
            lambda: _save_time_error_plot(
                results_df=results_df,
                value_col="mag_abs_error",
                ylabel="Absolute magnitude error",
                title=f"{scenario_id} - Magnitude error over time",
                output_path=plots_dir / "time_mag_abs_error.png",
            ),
            "time_mag_abs_error.png",
        ),
        (
            lambda: _save_time_error_plot(
                results_df=results_df,
                value_col="angle_abs_error_deg",
                ylabel="Absolute angle error [deg]",
                title=f"{scenario_id} - Angle error over time",
                output_path=plots_dir / "time_angle_abs_error_deg.png",
            ),
            "time_angle_abs_error_deg.png",
        ),
        (
            lambda: _save_regime_boxplot(
                results_df=results_df,
                value_col="mag_abs_error",
                ylabel="Absolute magnitude error",
                title=f"{scenario_id} - Magnitude error by regime",
                output_path=plots_dir / "boxplot_mag_abs_error_by_regime.png",
            ),
            "boxplot_mag_abs_error_by_regime.png",
        ),
        (
            lambda: _save_regime_boxplot(
                results_df=results_df,
                value_col="angle_abs_error_deg",
                ylabel="Absolute angle error [deg]",
                title=f"{scenario_id} - Angle error by regime",
                output_path=plots_dir / "boxplot_angle_abs_error_deg_by_regime.png",
            ),
            "boxplot_angle_abs_error_deg_by_regime.png",
        ),
        (
            lambda: _save_scatter_plot(
                results_df=results_df,
                x_col="mag_true",
                y_col="mag_hat",
                xlabel="True magnitude",
                ylabel="Estimated magnitude",
                title=f"{scenario_id} - True vs estimated magnitude",
                output_path=plots_dir / "scatter_mag_true_vs_hat.png",
            ),
            "scatter_mag_true_vs_hat.png",
        ),
        (
            lambda: _save_scatter_plot(
                results_df=results_df,
                x_col="angle_true_deg",
                y_col="angle_hat_deg",
                xlabel="True angle [deg]",
                ylabel="Estimated angle [deg]",
                title=f"{scenario_id} - True vs estimated angle",
                output_path=plots_dir / "scatter_angle_true_vs_hat.png",
            ),
            "scatter_angle_true_vs_hat.png",
        ),
    ]

    for plot_fn, filename in plot_specs:
        plot_fn()
        saved_plot_files.append(str(Path("plots") / filename))

    example_plot_files = _save_example_bus_plots(
        results_df=results_df,
        per_bus_df=per_bus_df,
        output_dir=plots_dir,
    )
    saved_plot_files.extend(str(Path("plots") / name) for name in example_plot_files)

    report = _build_report(
        scenario_id=scenario_id,
        representation=representation,
        results_df=results_df,
        per_bus_df=per_bus_df,
        baseline_summary=baseline_summary,
        output_dir=output_dir,
        saved_plot_files=saved_plot_files,
    )
    _save_json(report, report_json_path)

    return {
        "output_dir": str(output_dir),
        "report_json": str(report_json_path),
        "results_csv": str(results_csv_path),
        "per_bus_csv": str(per_bus_csv_path),
    }