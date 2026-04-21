"""Reusable comparison reports for m4 run outputs (type0 vs type1, etc.)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _require_run_files(run_dir: Path) -> dict[str, Path]:
    required = {
        "summary_by_bus": run_dir / "estimated" / "estimation_summary_by_bus.csv",
        "summary_by_signal": run_dir / "estimated" / "estimation_summary_by_signal.csv",
        "metrics_long": run_dir / "estimated" / "estimation_metrics_long.csv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Comparison requires run artifacts, missing: {missing}")
    return required


def _quality_percent(summary_bus: pd.DataFrame) -> pd.Series:
    rel = pd.to_numeric(summary_bus["relative_rmse"], errors="coerce").fillna(1.0)
    corr = pd.to_numeric(summary_bus["corr"], errors="coerce").fillna(0.0)
    rel_score = (1.0 - rel).clip(lower=0.0, upper=1.0) * 100.0
    corr_score = ((corr + 1.0) / 2.0).clip(lower=0.0, upper=1.0) * 100.0
    return 0.7 * rel_score + 0.3 * corr_score


def _plot_pair_bars(
    buses: list[str],
    values_a: np.ndarray,
    values_b: np.ndarray,
    title: str,
    ylabel: str,
    out_path: Path,
    label_a: str,
    label_b: str,
) -> None:
    x = np.arange(len(buses))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.bar(x - width / 2, values_a, width=width, label=label_a, color="#577590")
    ax.bar(x + width / 2, values_b, width=width, label=label_b, color="#f3722c")
    ax.set_xticks(x)
    ax.set_xticklabels(buses)
    ax.set_xlabel("Bus ID")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _family_bus_error_percent(metrics_long: pd.DataFrame, signals: list[str]) -> pd.Series:
    subset = metrics_long[metrics_long["signal"].isin(signals)].copy()
    if subset.empty:
        return pd.Series(dtype=float)
    subset["bus_id"] = subset["bus_id"].astype(str)
    subset["err"] = pd.to_numeric(subset["relative_rmse"], errors="coerce").fillna(1.0).clip(lower=0.0) * 100.0
    return subset.groupby("bus_id")["err"].mean().sort_index(key=lambda s: s.astype(int))


def compare_runs(run_a_dir: str | Path, run_b_dir: str | Path, output_dir: str | Path) -> dict:
    """Compare two run directories and write comparison CSVs + plots."""
    run_a = Path(run_a_dir)
    run_b = Path(run_b_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files_a = _require_run_files(run_a)
    files_b = _require_run_files(run_b)

    sum_bus_a = pd.read_csv(files_a["summary_by_bus"])
    sum_bus_b = pd.read_csv(files_b["summary_by_bus"])
    sum_sig_a = pd.read_csv(files_a["summary_by_signal"])
    sum_sig_b = pd.read_csv(files_b["summary_by_signal"])
    metrics_a = pd.read_csv(files_a["metrics_long"])
    metrics_b = pd.read_csv(files_b["metrics_long"])

    bus_cmp = pd.merge(sum_bus_a, sum_bus_b, on="bus_id", how="outer", suffixes=("_run_a", "_run_b"))
    for col in ["rmse", "mae", "relative_rmse", "corr"]:
        a_col = f"{col}_run_a"
        b_col = f"{col}_run_b"
        if a_col in bus_cmp.columns and b_col in bus_cmp.columns:
            bus_cmp[f"{col}_delta_b_minus_a"] = pd.to_numeric(bus_cmp[b_col], errors="coerce") - pd.to_numeric(
                bus_cmp[a_col], errors="coerce"
            )
    bus_cmp = bus_cmp.sort_values("bus_id", key=lambda s: s.astype(int))
    bus_cmp.to_csv(out_dir / "comparison_summary_by_bus.csv", index=False)

    sig_cmp = pd.merge(sum_sig_a, sum_sig_b, on="signal", how="outer", suffixes=("_run_a", "_run_b"))
    for col in ["rmse", "mae", "relative_rmse", "corr"]:
        a_col = f"{col}_run_a"
        b_col = f"{col}_run_b"
        if a_col in sig_cmp.columns and b_col in sig_cmp.columns:
            sig_cmp[f"{col}_delta_b_minus_a"] = pd.to_numeric(sig_cmp[b_col], errors="coerce") - pd.to_numeric(
                sig_cmp[a_col], errors="coerce"
            )
    sig_cmp = sig_cmp.sort_values("signal")
    sig_cmp.to_csv(out_dir / "comparison_summary_by_signal.csv", index=False)

    # Comparison plots
    buses = sorted(set(bus_cmp["bus_id"].astype(str).tolist()), key=lambda x: int(x))
    a_map = sum_bus_a.copy()
    a_map["bus_id"] = a_map["bus_id"].astype(str)
    b_map = sum_bus_b.copy()
    b_map["bus_id"] = b_map["bus_id"].astype(str)
    a_map = a_map.set_index("bus_id")
    b_map = b_map.set_index("bus_id")

    q_a = _quality_percent(a_map.reindex(buses).reset_index()).to_numpy(dtype=float)
    q_b = _quality_percent(b_map.reindex(buses).reset_index()).to_numpy(dtype=float)
    _plot_pair_bars(
        buses,
        q_a,
        q_b,
        "Bus Quality Score (%) - Run A vs Run B",
        "Quality (%)",
        out_dir / "estimation_bus_quality_percent.png",
        "Run A",
        "Run B",
    )

    rel_a = (
        pd.to_numeric(a_map.reindex(buses)["relative_rmse"], errors="coerce").fillna(1.0).clip(lower=0.0).to_numpy(dtype=float)
        * 100.0
    )
    rel_b = (
        pd.to_numeric(b_map.reindex(buses)["relative_rmse"], errors="coerce").fillna(1.0).clip(lower=0.0).to_numpy(dtype=float)
        * 100.0
    )
    _plot_pair_bars(
        buses,
        rel_a,
        rel_b,
        "Relative Error by Bus (%) - Run A vs Run B",
        "Relative RMSE (%)",
        out_dir / "estimation_bus_relative_rmse_percent.png",
        "Run A",
        "Run B",
    )

    families = [
        ("comparison_voltage_mag_percent_by_bus.png", ["VA_MAG", "VB_MAG", "VC_MAG"], "Voltage Magnitude Error by Bus (%)"),
        ("comparison_current_mag_percent_by_bus.png", ["IA_MAG", "IB_MAG", "IC_MAG"], "Current Magnitude Error by Bus (%)"),
        ("comparison_frequency_percent_by_bus.png", ["Freq"], "Frequency Error by Bus (%)"),
        ("comparison_rocof_percent_by_bus.png", ["ROCOF"], "ROCOF Error by Bus (%)"),
    ]
    for filename, signals, title in families:
        fam_a = _family_bus_error_percent(metrics_a, signals)
        fam_b = _family_bus_error_percent(metrics_b, signals)
        fam_buses = sorted(set(fam_a.index.astype(str)).union(set(fam_b.index.astype(str))), key=lambda x: int(x))
        val_a = fam_a.reindex(fam_buses).fillna(0.0).to_numpy(dtype=float)
        val_b = fam_b.reindex(fam_buses).fillna(0.0).to_numpy(dtype=float)
        _plot_pair_bars(
            fam_buses,
            val_a,
            val_b,
            f"{title} - Run A vs Run B",
            "Relative RMSE (%)",
            out_dir / filename,
            "Run A",
            "Run B",
        )

    return {
        "comparison_dir": str(out_dir),
        "bus_rows": int(len(bus_cmp)),
        "signal_rows": int(len(sig_cmp)),
    }
