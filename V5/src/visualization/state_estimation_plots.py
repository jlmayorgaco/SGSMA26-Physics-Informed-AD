"""Plot generation for M6 state-estimation diagnostics."""

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


def plot_selected_bus_voltage(
    timestamps: np.ndarray,
    bus_order: list[str],
    v_est_pu: np.ndarray,
    v_true_pu: np.ndarray,
    buses: list[str],
    out_dir: str | Path,
) -> dict[str, str]:
    """Create selected-bus |V| and angle overlay plots."""
    out = Path(out_dir)
    t = np.asarray(timestamps, dtype=float)
    paths: dict[str, str] = {}

    fig_mag, ax_mag = plt.subplots(figsize=(10, 5))
    fig_ang, ax_ang = plt.subplots(figsize=(10, 5))
    pos = {b: i for i, b in enumerate(bus_order)}
    for bus in buses:
        if bus not in pos:
            continue
        i = pos[bus]
        ax_mag.plot(t, np.abs(v_true_pu[:, i]), label=f"{bus} true")
        ax_mag.plot(t, np.abs(v_est_pu[:, i]), "--", label=f"{bus} est")
        ax_ang.plot(t, np.rad2deg(np.angle(v_true_pu[:, i])), label=f"{bus} true")
        ax_ang.plot(t, np.rad2deg(np.angle(v_est_pu[:, i])), "--", label=f"{bus} est")

    ax_mag.set_title("Estimated vs True Voltage Magnitude (p.u.)")
    ax_mag.set_xlabel("Time (s)")
    ax_mag.set_ylabel("|V| p.u.")
    ax_mag.grid(alpha=0.2)
    ax_mag.legend(fontsize=8, ncol=2)
    ax_ang.set_title("Estimated vs True Voltage Angle (deg)")
    ax_ang.set_xlabel("Time (s)")
    ax_ang.set_ylabel("Angle (deg)")
    ax_ang.grid(alpha=0.2)
    ax_ang.legend(fontsize=8, ncol=2)

    p1 = out / "voltage_mag_selected_buses.png"
    p2 = out / "voltage_ang_selected_buses.png"
    _save(fig_mag, p1)
    _save(fig_ang, p2)
    paths["voltage_mag_selected_buses"] = str(p1)
    paths["voltage_ang_selected_buses"] = str(p2)
    return paths


def plot_error_summaries(
    timestamps: np.ndarray,
    bus_order: list[str],
    v_est_pu: np.ndarray,
    v_true_pu: np.ndarray,
    per_bus_metrics: pd.DataFrame,
    diagnostics_df: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, str]:
    """Create error-over-time, per-bus RMSE and heatmap diagnostics."""
    out = Path(out_dir)
    t = np.asarray(timestamps, dtype=float)
    abs_err = np.abs(np.abs(v_est_pu) - np.abs(v_true_pu))
    mean_err_t = abs_err.mean(axis=1)

    paths: dict[str, str] = {}

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t, mean_err_t, color="#005f73")
    ax1.set_title("Mean Absolute Voltage Magnitude Error Over Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("MAE |V| (p.u.)")
    ax1.grid(alpha=0.2)
    p = out / "error_over_time.png"
    _save(fig1, p)
    paths["error_over_time"] = str(p)

    fig2, ax2 = plt.subplots(figsize=(11, 4))
    if not per_bus_metrics.empty:
        bus_col = "bus" if "bus" in per_bus_metrics.columns else "BUS"
        rmse_col = "rmse_mag_pu" if "rmse_mag_pu" in per_bus_metrics.columns else "RMSE_V_MAG"
        ax2.bar(per_bus_metrics[bus_col], per_bus_metrics[rmse_col], color="#0a9396")
    ax2.set_title("Per-Bus RMSE of |V|")
    ax2.set_xlabel("Bus")
    ax2.set_ylabel("RMSE |V| (p.u.)")
    ax2.tick_params(axis="x", rotation=90)
    ax2.grid(alpha=0.2)
    p = out / "per_bus_rmse.png"
    _save(fig2, p)
    paths["per_bus_rmse"] = str(p)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    im = ax3.imshow(abs_err.T, aspect="auto", origin="lower", cmap="viridis")
    ax3.set_title("Bus x Time |V| Absolute Error Heatmap")
    ax3.set_xlabel("Time Index")
    ax3.set_ylabel("Bus Index")
    fig3.colorbar(im, ax=ax3, label="|V| abs error (p.u.)")
    p = out / "heatmap_bus_time_error.png"
    _save(fig3, p)
    paths["heatmap_bus_time_error"] = str(p)

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    if not diagnostics_df.empty:
        tcol = "timestamp" if "timestamp" in diagnostics_df.columns else "TIMESTAMP"
        rcol = "residual_norm" if "residual_norm" in diagnostics_df.columns else "RESIDUAL_NORM"
        ax4.plot(diagnostics_df[tcol], diagnostics_df[rcol], color="#bb3e03")
    ax4.set_title("Residual Norm Over Time")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Residual norm")
    ax4.grid(alpha=0.2)
    p = out / "residual_norm.png"
    _save(fig4, p)
    paths["residual_norm"] = str(p)

    return paths


def generate_diagnostic_plots(
    frame_df: pd.DataFrame,
    pmu_fit_df: pd.DataFrame,
    residual_df: pd.DataFrame,
    variability_df: pd.DataFrame,
    dropped_df: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, str]:
    """Generate required M6 diagnostic plots."""
    out = Path(out_dir)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(10, 4))
    if not frame_df.empty:
        ax.plot(frame_df["TIMESTAMP"], frame_df["N_VALID_PMUS"], label="valid PMUs", color="#0a9396")
        ax.plot(frame_df["TIMESTAMP"], frame_df["N_PMUS_USED_IN_SOLVER"], label="used PMUs", color="#005f73")
    ax.set_title("Valid vs Used PMUs Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("PMU count")
    ax.legend()
    ax.grid(alpha=0.2)
    p = out / "n_valid_pmus_over_time.png"
    _save(fig, p)
    paths["n_valid_pmus_over_time"] = str(p)

    fig, ax = plt.subplots(figsize=(10, 4))
    if not pmu_fit_df.empty:
        for bus, grp in pmu_fit_df.groupby("PMU_BUS"):
            ax.plot(grp["TIMESTAMP"], grp["ABS_COMPLEX_ERROR"], label=bus)
        ax.legend(fontsize=8, ncol=3)
    ax.set_title("PMU Fit Absolute Complex Error Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Abs error")
    ax.grid(alpha=0.2)
    p = out / "pmu_fit_error_over_time.png"
    _save(fig, p)
    paths["pmu_fit_error_over_time"] = str(p)

    fig, ax = plt.subplots(figsize=(10, 4))
    if not residual_df.empty:
        ax.plot(residual_df["TIMESTAMP"], residual_df["RESIDUAL_NORM"], color="#bb3e03")
    ax.set_title("Residual Norm Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Residual norm")
    ax.grid(alpha=0.2)
    p = out / "residual_norm_over_time.png"
    _save(fig, p)
    paths["residual_norm_over_time"] = str(p)

    fig, ax = plt.subplots(figsize=(10, 4))
    if not residual_df.empty:
        ax.plot(residual_df["TIMESTAMP"], residual_df["DATA_TERM_VALUE"], label="data term")
        ax.plot(residual_df["TIMESTAMP"], residual_df["TEMPORAL_PRIOR_TERM_VALUE"], label="temporal prior")
        ax.plot(residual_df["TIMESTAMP"], residual_df["LOADFLOW_PRIOR_TERM_VALUE"], label="loadflow prior")
        ax.legend()
    ax.set_title("Data vs Prior Objective Terms")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Term value")
    ax.grid(alpha=0.2)
    p = out / "prior_vs_data_term_over_time.png"
    _save(fig, p)
    paths["prior_vs_data_term_over_time"] = str(p)

    fig, ax = plt.subplots(figsize=(12, 4))
    if not variability_df.empty:
        ax.bar(variability_df["BUS"], variability_df["STD_V_MAG"], color="#219ebc")
        ax.tick_params(axis="x", rotation=90)
    ax.set_title("Per-Bus Variability (STD |V|)")
    ax.set_xlabel("Bus")
    ax.set_ylabel("STD |V|")
    ax.grid(alpha=0.2)
    p = out / "per_bus_rmse_or_fit_bar.png"
    _save(fig, p)
    paths["per_bus_rmse_or_fit_bar"] = str(p)

    fig, ax = plt.subplots(figsize=(10, 6))
    if not variability_df.empty:
        mat = np.vstack([variability_df["STD_V_MAG"].to_numpy(float), variability_df["STD_V_ANG_DEG"].to_numpy(float)])
        im = ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set_yticks([0, 1], ["STD_V_MAG", "STD_V_ANG_DEG"])
        fig.colorbar(im, ax=ax)
    ax.set_title("Bus Variability Heatmap")
    p = out / "bus_variability_heatmap.png"
    _save(fig, p)
    paths["bus_variability_heatmap"] = str(p)

    fig, ax = plt.subplots(figsize=(10, 4))
    if not dropped_df.empty:
        dropped_df["REASON"].value_counts().plot(kind="bar", ax=ax, color="#ae2012")
        ax.tick_params(axis="x", rotation=45)
    ax.set_title("Dropped PMUs by Reason")
    ax.set_xlabel("Reason")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    p = out / "dropped_pmus_by_reason.png"
    _save(fig, p)
    paths["dropped_pmus_by_reason"] = str(p)
    return paths


def plot_estimator_performance(
    per_bus_metrics: pd.DataFrame,
    global_metrics: dict,
    pmu_fit_df: pd.DataFrame,
    out_dir: str | Path,
) -> dict[str, str]:
    """Create dedicated estimator-performance plots."""
    out = Path(out_dir)
    paths: dict[str, str] = {}

    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ["rmse_v_mag_all", "mae_v_mag_all", "rmse_ang_all", "mae_ang_all"]
    labels = ["RMSE |V|", "MAE |V|", "RMSE angle", "MAE angle"]
    vals = [float(global_metrics.get(k, np.nan)) for k in keys]
    ax.bar(labels, vals, color=["#0a9396", "#94d2bd", "#ee9b00", "#ca6702"])
    ax.set_title("Estimator Global Performance")
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", alpha=0.2)
    p = out / "estimator_performance_overview.png"
    _save(fig, p)
    paths["estimator_performance_overview"] = str(p)

    fig, ax = plt.subplots(figsize=(12, 4))
    if not per_bus_metrics.empty:
        bcol = "BUS" if "BUS" in per_bus_metrics.columns else "bus"
        ecol = "MAE_V_MAG" if "MAE_V_MAG" in per_bus_metrics.columns else "mae_mag_pu"
        ranked = per_bus_metrics.sort_values(ecol, ascending=False)
        ax.bar(ranked[bcol], ranked[ecol], color="#219ebc")
        ax.tick_params(axis="x", rotation=90)
    ax.set_title("Per-Bus Magnitude MAE (Ranked)")
    ax.set_ylabel("MAE |V|")
    ax.grid(axis="y", alpha=0.2)
    p = out / "per_bus_mae_ranked.png"
    _save(fig, p)
    paths["per_bus_mae_ranked"] = str(p)

    fig, ax = plt.subplots(figsize=(8, 4))
    if not pmu_fit_df.empty and "ABS_COMPLEX_ERROR" in pmu_fit_df.columns:
        vals = pmu_fit_df["ABS_COMPLEX_ERROR"].dropna().to_numpy(float)
        if vals.size > 0:
            ax.hist(vals, bins=30, color="#005f73", alpha=0.9)
    ax.set_title("PMU Fit Absolute Complex Error Distribution")
    ax.set_xlabel("Absolute complex error")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    p = out / "pmu_fit_error_distribution.png"
    _save(fig, p)
    paths["pmu_fit_error_distribution"] = str(p)
    return paths
