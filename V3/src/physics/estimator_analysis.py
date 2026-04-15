from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.physics.global_wls_angle_residual_ekf_estimator import (
    GlobalWLSAngleResidualEKFConfig,
    run_global_wls_angle_residual_ekf_estimator,
)
from src.physics.ybus_baseline import run_ybus_voltage_baseline

REQUIRED_RESULT_COLUMNS = {
    "TIMESTAMP",
    "Event",
    "bus_id",
    "mag_true",
    "mag_hat",
    "mag_error",
    "mag_abs_error",
    "angle_true_deg",
    "angle_hat_deg",
    "angle_error_deg",
    "angle_abs_error_deg",
}

EVENT_SUBSETS: dict[str, set[int] | None] = {
    "global": None,
    "topology_fixed_physical_only": {0, 1, 3, 4},
    "fault_only_vs_normal": {0, 1},
}


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    return value


def _save_json(data: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(data), file, indent=2, ensure_ascii=False)


def _validate_results_df(results_df: pd.DataFrame) -> None:
    missing = REQUIRED_RESULT_COLUMNS - set(results_df.columns)
    if missing:
        raise ValueError(
            "Estimator results dataframe is missing required columns: "
            f"{sorted(missing)}"
        )


def _filter_results_by_subset(
    results_df: pd.DataFrame,
    allowed_events: set[int] | None,
) -> pd.DataFrame:
    if allowed_events is None:
        return results_df.copy()
    return results_df.loc[results_df["Event"].isin(allowed_events)].copy()


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if np.isclose(ss_tot, 0.0):
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


def _metric_block(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {
            "n_samples": 0,
            "mag_mean_true": float("nan"),
            "mag_rmse": float("nan"),
            "mag_mae": float("nan"),
            "mag_nrmse_pct": float("nan"),
            "mag_mape_pct": float("nan"),
            "mag_r2": float("nan"),
            "angle_mae_deg": float("nan"),
            "angle_rmse_deg": float("nan"),
        }

    mag_true = df["mag_true"].to_numpy()
    mag_hat = df["mag_hat"].to_numpy()
    mag_error = df["mag_error"].to_numpy()
    mag_abs_error = df["mag_abs_error"].to_numpy()

    angle_error_deg = df["angle_error_deg"].to_numpy()
    angle_abs_error_deg = df["angle_abs_error_deg"].to_numpy()

    mean_mag_true = float(np.mean(mag_true))
    mag_rmse = float(np.sqrt(np.mean(np.square(mag_error))))
    mag_mae = float(np.mean(mag_abs_error))
    mag_nrmse_pct = (
        float(100.0 * mag_rmse / mean_mag_true)
        if not np.isclose(mean_mag_true, 0.0)
        else float("nan")
    )
    mag_mape_pct = float(
        100.0 * np.mean(mag_abs_error / np.maximum(np.abs(mag_true), 1e-12))
    )
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


def _build_regime_metrics(results_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        "overall": _metric_block(results_df),
        "normal_only": _metric_block(results_df.loc[results_df["Event"] == 0]),
        "event_only": _metric_block(results_df.loc[results_df["Event"] != 0]),
    }


def _build_per_bus_metrics(results_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for bus_id, bus_df in results_df.groupby("bus_id"):
        metrics = _metric_block(bus_df)
        metrics["bus_id"] = int(bus_id)
        rows.append(metrics)

    if not rows:
        return pd.DataFrame(columns=["bus_id"])

    return (
        pd.DataFrame(rows)
        .sort_values(["mag_nrmse_pct", "angle_mae_deg"], ascending=[True, True])
        .reset_index(drop=True)
    )


def _build_event_spans(time_df: pd.DataFrame) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []

    in_event = False
    start_time = 0.0

    timestamps = time_df["TIMESTAMP"].to_numpy()
    events = time_df["Event"].to_numpy()

    for idx, event_value in enumerate(events):
        is_event = int(event_value) != 0

        if is_event and not in_event:
            start_time = float(timestamps[idx])
            in_event = True

        if in_event and not is_event:
            end_time = float(timestamps[idx - 1])
            spans.append((start_time, end_time))
            in_event = False

    if in_event and len(timestamps) > 0:
        spans.append((start_time, float(timestamps[-1])))

    return spans


def _add_event_shading(ax: plt.Axes, time_df: pd.DataFrame) -> None:
    for start_time, end_time in _build_event_spans(time_df):
        ax.axvspan(start_time, end_time, alpha=0.15)


def _save_bar_plot(
    plot_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(plot_df[x_col].astype(str), plot_df[y_col].to_numpy())
    ax.set_xlabel(xlabel)
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
    if results_df.empty:
        return

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
    if results_df.empty:
        return

    normal_values = results_df.loc[results_df["Event"] == 0, value_col].to_numpy()
    event_values = results_df.loc[results_df["Event"] != 0, value_col].to_numpy()

    if len(normal_values) == 0 or len(event_values) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [normal_values, event_values],
        tick_labels=["Normal", "Event"],
        showfliers=False,
    )
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

    if plot_df.empty:
        return

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
    if per_bus_df.empty:
        return []

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

        if bus_df.empty:
            continue

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


def analyze_estimator_results(
    estimator_name: str,
    scenario_id: str,
    results_df: pd.DataFrame,
    estimator_summary: Any,
    output_root: str | Path = "artifacts/estimator_analysis",
    subset_name: str = "global",
) -> dict[str, str]:
    _validate_results_df(results_df)

    estimator_dir = _ensure_dir(Path(output_root) / scenario_id / subset_name / estimator_name)
    plots_dir = _ensure_dir(estimator_dir / "plots")

    subset_metrics = _build_regime_metrics(results_df)
    per_bus_df = _build_per_bus_metrics(results_df)

    results_csv_path = estimator_dir / "results_long.csv"
    per_bus_csv_path = estimator_dir / "per_bus_metrics.csv"
    report_json_path = estimator_dir / "report.json"

    results_df.to_csv(results_csv_path, index=False)
    per_bus_df.to_csv(per_bus_csv_path, index=False)

    saved_plot_files: list[str] = []

    plot_jobs = [
        (
            lambda: _save_bar_plot(
                plot_df=per_bus_df.sort_values("mag_nrmse_pct"),
                x_col="bus_id",
                y_col="mag_nrmse_pct",
                xlabel="Hidden bus",
                ylabel="Magnitude NRMSE [%]",
                title=f"{subset_name} - {estimator_name} - per-bus magnitude NRMSE",
                output_path=plots_dir / "per_bus_mag_nrmse_pct.png",
            ),
            "per_bus_mag_nrmse_pct.png",
        ),
        (
            lambda: _save_bar_plot(
                plot_df=per_bus_df.sort_values("angle_mae_deg"),
                x_col="bus_id",
                y_col="angle_mae_deg",
                xlabel="Hidden bus",
                ylabel="Angle MAE [deg]",
                title=f"{subset_name} - {estimator_name} - per-bus angle MAE",
                output_path=plots_dir / "per_bus_angle_mae_deg.png",
            ),
            "per_bus_angle_mae_deg.png",
        ),
        (
            lambda: _save_time_error_plot(
                results_df=results_df,
                value_col="mag_abs_error",
                ylabel="Absolute magnitude error",
                title=f"{subset_name} - {estimator_name} - magnitude error over time",
                output_path=plots_dir / "time_mag_abs_error.png",
            ),
            "time_mag_abs_error.png",
        ),
        (
            lambda: _save_time_error_plot(
                results_df=results_df,
                value_col="angle_abs_error_deg",
                ylabel="Absolute angle error [deg]",
                title=f"{subset_name} - {estimator_name} - angle error over time",
                output_path=plots_dir / "time_angle_abs_error_deg.png",
            ),
            "time_angle_abs_error_deg.png",
        ),
        (
            lambda: _save_regime_boxplot(
                results_df=results_df,
                value_col="mag_abs_error",
                ylabel="Absolute magnitude error",
                title=f"{subset_name} - {estimator_name} - magnitude error by regime",
                output_path=plots_dir / "boxplot_mag_abs_error_by_regime.png",
            ),
            "boxplot_mag_abs_error_by_regime.png",
        ),
        (
            lambda: _save_regime_boxplot(
                results_df=results_df,
                value_col="angle_abs_error_deg",
                ylabel="Absolute angle error [deg]",
                title=f"{subset_name} - {estimator_name} - angle error by regime",
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
                title=f"{subset_name} - {estimator_name} - true vs estimated magnitude",
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
                title=f"{subset_name} - {estimator_name} - true vs estimated angle",
                output_path=plots_dir / "scatter_angle_true_vs_hat.png",
            ),
            "scatter_angle_true_vs_hat.png",
        ),
    ]

    for job, filename in plot_jobs:
        job()
        saved_plot_files.append(str(Path("plots") / filename))

    example_plot_files = _save_example_bus_plots(
        results_df=results_df,
        per_bus_df=per_bus_df,
        output_dir=plots_dir,
    )
    saved_plot_files.extend(str(Path("plots") / name) for name in example_plot_files)

    best_buses = (
        per_bus_df.nsmallest(5, "mag_nrmse_pct")[["bus_id", "mag_nrmse_pct", "angle_mae_deg"]]
        if not per_bus_df.empty
        else pd.DataFrame()
    )
    worst_buses = (
        per_bus_df.nlargest(5, "mag_nrmse_pct")[["bus_id", "mag_nrmse_pct", "angle_mae_deg"]]
        if not per_bus_df.empty
        else pd.DataFrame()
    )

    report = {
        "scenario_id": scenario_id,
        "subset_name": subset_name,
        "estimator_name": estimator_name,
        "estimator_summary_full_run": _to_jsonable(estimator_summary),
        "subset_metrics": subset_metrics,
        "ranking": {
            "best_buses_by_mag_nrmse_pct": best_buses.to_dict(orient="records"),
            "worst_buses_by_mag_nrmse_pct": worst_buses.to_dict(orient="records"),
        },
        "per_bus_metrics": per_bus_df.to_dict(orient="records"),
        "artifacts": {
            "results_csv": str(results_csv_path),
            "per_bus_csv": str(per_bus_csv_path),
            "report_json": str(report_json_path),
            "plots": saved_plot_files,
        },
    }

    _save_json(report, report_json_path)

    return {
        "output_dir": str(estimator_dir),
        "results_csv": str(results_csv_path),
        "per_bus_csv": str(per_bus_csv_path),
        "report_json": str(report_json_path),
    }


def _build_comparison_df(
    static_results: pd.DataFrame,
    dynamic_results: pd.DataFrame,
) -> pd.DataFrame:
    static_metrics = _metric_block(static_results)
    dynamic_metrics = _metric_block(dynamic_results)

    metrics = [
        "mag_rmse",
        "mag_mae",
        "mag_nrmse_pct",
        "mag_mape_pct",
        "mag_r2",
        "angle_mae_deg",
        "angle_rmse_deg",
    ]

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        static_value = static_metrics.get(metric, np.nan)
        dynamic_value = dynamic_metrics.get(metric, np.nan)

        if np.isfinite(static_value) and not np.isclose(static_value, 0.0):
            delta_pct = 100.0 * (dynamic_value - static_value) / static_value
        else:
            delta_pct = float("nan")

        rows.append(
            {
                "metric": metric,
                "static": static_value,
                "dynamic": dynamic_value,
                "delta_dynamic_minus_static": dynamic_value - static_value,
                "delta_pct_dynamic_vs_static": delta_pct,
            }
        )

    return pd.DataFrame(rows)


def _save_comparison_group_plot(
    comparison_df: pd.DataFrame,
    metrics: list[str],
    title: str,
    output_path: Path,
) -> None:
    plot_df = comparison_df.loc[comparison_df["metric"].isin(metrics)].copy()
    if plot_df.empty:
        return

    x = np.arange(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, plot_df["static"].to_numpy(), width=width, label="Static")
    ax.bar(x + width / 2, plot_df["dynamic"].to_numpy(), width=width, label="Dynamic")

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["metric"].tolist(), rotation=20, ha="right")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_delta_plot(
    comparison_df: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    if comparison_df.empty:
        return

    x = np.arange(len(comparison_df))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, comparison_df["delta_pct_dynamic_vs_static"].to_numpy())
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["metric"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Delta [%] dynamic vs static")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_and_analyze_estimators(
    scenario_id: str = "SIM_0001",
    representation: str = "positive_sequence",
    dynamic_config: GlobalWLSAngleResidualEKFConfig | None = None,
    output_root: str | Path = "artifacts/estimator_analysis",
) -> dict[str, Any]:
    static_results_full, _, static_summary = run_ybus_voltage_baseline(
        scenario_id=scenario_id,
        representation=representation,
    )

    dynamic_results_full, _, dynamic_summary = run_global_wls_angle_residual_ekf_estimator(
        scenario_id=scenario_id,
        config=dynamic_config or GlobalWLSAngleResidualEKFConfig(representation=representation),
    )

    all_outputs: dict[str, Any] = {}

    for subset_name, allowed_events in EVENT_SUBSETS.items():
        static_results = _filter_results_by_subset(static_results_full, allowed_events)
        dynamic_results = _filter_results_by_subset(dynamic_results_full, allowed_events)

        static_artifacts = analyze_estimator_results(
            estimator_name="static_ybus",
            scenario_id=scenario_id,
            results_df=static_results,
            estimator_summary=static_summary,
            output_root=output_root,
            subset_name=subset_name,
        )

        dynamic_artifacts = analyze_estimator_results(
            estimator_name="global_wls_angle_residual_ekf",
            scenario_id=scenario_id,
            results_df=dynamic_results,
            estimator_summary=dynamic_summary,
            output_root=output_root,
            subset_name=subset_name,
        )

        comparison_dir = _ensure_dir(Path(output_root) / scenario_id / subset_name / "comparison")
        comparison_df = _build_comparison_df(static_results, dynamic_results)

        comparison_csv_path = comparison_dir / "comparison_metrics.csv"
        comparison_json_path = comparison_dir / "comparison_report.json"
        magnitude_plot_path = comparison_dir / "comparison_magnitude_metrics.png"
        angle_plot_path = comparison_dir / "comparison_angle_metrics.png"
        delta_plot_path = comparison_dir / "comparison_delta_pct.png"

        comparison_df.to_csv(comparison_csv_path, index=False)

        _save_comparison_group_plot(
            comparison_df=comparison_df,
            metrics=["mag_rmse", "mag_mae", "mag_nrmse_pct", "mag_mape_pct"],
            title=f"{scenario_id} - {subset_name} - magnitude comparison",
            output_path=magnitude_plot_path,
        )

        _save_comparison_group_plot(
            comparison_df=comparison_df,
            metrics=["angle_mae_deg", "angle_rmse_deg"],
            title=f"{scenario_id} - {subset_name} - angle comparison",
            output_path=angle_plot_path,
        )

        _save_delta_plot(
            comparison_df=comparison_df,
            title=f"{scenario_id} - {subset_name} - delta % dynamic vs static",
            output_path=delta_plot_path,
        )

        comparison_report = {
            "scenario_id": scenario_id,
            "subset_name": subset_name,
            "static_summary_full_run": _to_jsonable(static_summary),
            "dynamic_summary_full_run": _to_jsonable(dynamic_summary),
            "static_subset_metrics": _build_regime_metrics(static_results),
            "dynamic_subset_metrics": _build_regime_metrics(dynamic_results),
            "comparison_metrics": comparison_df.to_dict(orient="records"),
            "artifacts": {
                "comparison_csv": str(comparison_csv_path),
                "comparison_json": str(comparison_json_path),
                "magnitude_plot": str(magnitude_plot_path),
                "angle_plot": str(angle_plot_path),
                "delta_plot": str(delta_plot_path),
                "static_artifacts": static_artifacts,
                "dynamic_artifacts": dynamic_artifacts,
            },
        }
        _save_json(comparison_report, comparison_json_path)

        all_outputs[subset_name] = {
            "static": static_artifacts,
            "dynamic": dynamic_artifacts,
            "comparison": {
                "comparison_csv": str(comparison_csv_path),
                "comparison_json": str(comparison_json_path),
                "magnitude_plot": str(magnitude_plot_path),
                "angle_plot": str(angle_plot_path),
                "delta_plot": str(delta_plot_path),
            },
        }

    return all_outputs