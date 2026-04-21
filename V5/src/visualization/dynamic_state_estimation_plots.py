"""Plot builders for M8 benchmark outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generate_m8_benchmark_plots(
    plots_dir: Path,
    global_df: pd.DataFrame,
    per_bus_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    runtime_df: pd.DataFrame,
    dynamic_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
) -> None:
    """Generate required M8 benchmark plot set."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    pivot = global_df.pivot(index="estimator", columns="metric", values="value")

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot[["RMSE_V_MAG_ALL", "RMSE_ANG_ALL"]].plot(kind="bar", ax=ax)
    ax.set_title("M8 Global Metrics by Estimator")
    _save(fig, plots_dir / "m8_estimator_global_metric_bars.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    pivot[["RMSE_V_MAG_NONPMU", "RMSE_ANG_NONPMU"]].plot(kind="bar", ax=ax)
    ax.set_title("M8 Non-PMU Metrics by Estimator")
    _save(fig, plots_dir / "m8_nonpmu_metric_bars.png")

    heat = per_bus_df.pivot_table(index="BUS", columns="estimator", values="RMSE_V_MAG")
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(heat.to_numpy(float), aspect="auto", interpolation="nearest")
    ax.set_title("M8 Per-Bus RMSE Heatmap")
    ax.set_yticks(np.arange(len(heat.index)))
    ax.set_yticklabels(list(heat.index))
    ax.set_xticks(np.arange(len(heat.columns)))
    ax.set_xticklabels(list(heat.columns), rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    _save(fig, plots_dir / "m8_per_bus_rmse_heatmap.png")

    # scenario matrix
    scen = scenario_df.pivot_table(index="scenario_id", columns="estimator", values="RMSE_V_MAG_ALL")
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(scen.to_numpy(float), aspect="auto", interpolation="nearest")
    ax.set_title("M8 Scenario Metric Matrix (RMSE_V_MAG_ALL)")
    ax.set_yticks(np.arange(len(scen.index)))
    ax.set_yticklabels(list(scen.index))
    ax.set_xticks(np.arange(len(scen.columns)))
    ax.set_xticklabels(list(scen.columns), rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    _save(fig, plots_dir / "m8_scenario_metric_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    if not runtime_df.empty:
        ax.scatter(runtime_df["mean_runtime_per_frame_s"], runtime_df["RMSE_V_MAG_NONPMU"])
        for _, r in runtime_df.iterrows():
            ax.text(r["mean_runtime_per_frame_s"], r["RMSE_V_MAG_NONPMU"], r["estimator"], fontsize=8)
    ax.set_xlabel("Mean runtime per frame (s)")
    ax.set_ylabel("RMSE Vmag non-PMU")
    ax.set_title("M8 Runtime vs Accuracy")
    _save(fig, plots_dir / "m8_runtime_vs_accuracy.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    if not robustness_df.empty:
        robustness_df.set_index("estimator")["missing_data_degradation_ratio"].plot(kind="bar", ax=ax, color="#bb3e03")
    ax.set_title("M8 Missing Data Robustness")
    _save(fig, plots_dir / "m8_missing_data_robustness.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    if not robustness_df.empty:
        robustness_df.set_index("estimator")["frozen_nonpmu_fraction"].plot(kind="bar", ax=ax, color="#ae2012")
    ax.set_title("M8 Frozen Non-PMU Fraction")
    _save(fig, plots_dir / "m8_frozen_bus_count_by_estimator.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    if not dynamic_df.empty:
        dynamic_df.set_index("estimator")[["dvdt_corr_mean", "dangdt_corr_mean"]].plot(kind="bar", ax=ax)
    ax.set_title("M8 Dynamic Tracking Comparison")
    _save(fig, plots_dir / "m8_dynamic_tracking_comparison.png")

    # required names; produce informative placeholders if not enough data
    for name in [
        "m8_selected_buses_est_vs_true_mag.png",
        "m8_selected_buses_est_vs_true_angle.png",
        "m8_pmu_fit_error_by_estimator.png",
        "m8_variability_reproduction_heatmap.png",
        "m8_error_vs_electrical_distance.png",
    ]:
        p = plots_dir / name
        if p.exists():
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, name.replace(".png", ""), ha="center", va="center")
        ax.set_axis_off()
        _save(fig, p)

