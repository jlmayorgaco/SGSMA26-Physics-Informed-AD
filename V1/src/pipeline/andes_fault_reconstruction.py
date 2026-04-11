"""ANDES IEEE-39 single-fault full-state reconstruction experiment.

This script runs one three-phase fault on a non-PMU IEEE-39 bus using ANDES,
keeps only the eight competition PMU buses as observations, reconstructs all
39 bus voltage states with the project Ybus/Kirchhoff estimator, and compares
the estimates against the simulated hidden-bus truth.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.estimator.topology_state import TopologyStateEstimator, _angle_diff_deg
from src.grid.load_case import load_case
from src.io.load_csv import PMU_BUSES

log = logging.getLogger(__name__)


@dataclass
class ReconstructionResult:
    out_dir: Path
    report_path: Path
    metrics: pd.DataFrame


def _andes_ieee39_case() -> str:
    """Return the installed ANDES IEEE-39 full dynamic case path."""
    import andes

    case = Path(andes.__file__).resolve().parent / "cases" / "ieee39" / "ieee39_full.xlsx"
    if not case.exists():
        raise FileNotFoundError(f"ANDES IEEE-39 case not found: {case}")
    return str(case)


def run_andes_fault(
    *,
    fault_bus: int = 7,
    t_final: float = 10.0,
    fps: float = 30.0,
    fault_duration: float = 5.0 / 60.0,
    xf: float = 1e-3,
    rf: float = 0.0,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Run ANDES IEEE-39 with a single mid-simulation bus fault.

    Returns:
        timestamps, vm_true_pu, va_true_deg as DataFrames indexed by time and
        with integer bus-number columns 1..39.
    """
    if fault_bus in PMU_BUSES:
        raise ValueError(f"fault_bus={fault_bus} is a PMU bus; choose a non-PMU bus")
    if fault_bus < 1 or fault_bus > 39:
        raise ValueError("fault_bus must be in the IEEE-39 bus range 1..39")

    import andes

    case = _andes_ieee39_case()
    tf = float(t_final) / 2.0
    tc = tf + float(fault_duration)

    log.info("Loading ANDES case %s", case)
    system = andes.load(
        case,
        setup=False,
        no_output=True,
    )
    system.add(
        "Fault",
        {
            "idx": f"Fault_Bus{fault_bus}",
            "u": 1,
            "name": f"Fault_Bus{fault_bus}",
            "bus": fault_bus,
            "tf": tf,
            "tc": tc,
            "xf": xf,
            "rf": rf,
        },
    )
    system.setup()

    if not system.PFlow.run():
        raise RuntimeError("ANDES power flow did not converge")

    system.TDS.config.tf = float(t_final)
    system.TDS.config.tstep = 1.0 / float(fps)
    system.TDS.config.fixt = 1
    system.TDS.config.no_tqdm = 1
    system.TDS.config.save_every = 1

    log.info(
        "Running ANDES TDS: fault bus=%s, tf=%.4fs, tc=%.4fs, t_final=%.2fs",
        fault_bus,
        tf,
        tc,
        t_final,
    )
    ok = system.TDS.run(no_summary=True)
    if not ok or not system.TDS.converged:
        raise RuntimeError("ANDES time-domain simulation did not converge")

    vm = system.TDS.get_timeseries(system.Bus.v).copy()
    va = system.TDS.get_timeseries(system.Bus.a).copy()
    vm.columns = [int(c) for c in vm.columns]
    va.columns = [int(c) for c in va.columns]
    vm = vm.reindex(sorted(vm.columns), axis=1)
    va = va.reindex(sorted(va.columns), axis=1)
    va_deg = np.rad2deg(va)

    timestamps = vm.index.to_numpy(dtype=float)
    vm.index.name = "timestamp"
    va_deg.index.name = "timestamp"
    return timestamps, vm, va_deg


def reconstruct_from_pmus(
    grid,
    vm_true: pd.DataFrame,
    va_true_deg: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate all-bus voltage state at every timestamp from PMU buses only."""
    estimator = TopologyStateEstimator(grid)
    obs_idx = np.array(
        [estimator.bus_to_idx[bus] for bus in PMU_BUSES],
        dtype=int,
    )

    bus_order = estimator.bus_order
    vm_est = np.empty((len(vm_true), len(bus_order)), dtype=float)
    va_est = np.empty((len(vm_true), len(bus_order)), dtype=float)

    for row, (_, vm_row) in enumerate(vm_true.iterrows()):
        va_row = va_true_deg.iloc[row]
        vm_obs = vm_row.loc[PMU_BUSES].to_numpy(dtype=float)
        va_obs = va_row.loc[PMU_BUSES].to_numpy(dtype=float)
        vm_est[row] = estimator._extend(obs_idx, vm_obs, estimator.base_vm)
        va_est[row] = estimator._extend(obs_idx, va_obs, estimator.base_va)

    idx = vm_true.index.copy()
    vm_est_df = pd.DataFrame(vm_est, index=idx, columns=bus_order)
    va_est_df = pd.DataFrame(va_est, index=idx, columns=bus_order)
    vm_est_df.index.name = "timestamp"
    va_est_df.index.name = "timestamp"
    return vm_est_df, va_est_df


def _error_metrics(
    vm_true: pd.DataFrame,
    va_true_deg: pd.DataFrame,
    vm_est: pd.DataFrame,
    va_est_deg: pd.DataFrame,
    hidden_buses: list[int],
    fault_bus: int,
    fault_time: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-bus metrics and pointwise hidden-bus errors."""
    rows = []
    err_cols = {"timestamp": vm_true.index.to_numpy(dtype=float)}
    post_mask = vm_true.index.to_numpy(dtype=float) >= fault_time

    for bus in hidden_buses:
        vm_err = vm_est[bus].to_numpy(dtype=float) - vm_true[bus].to_numpy(dtype=float)
        va_err = _angle_diff_deg(
            va_est_deg[bus].to_numpy(dtype=float),
            va_true_deg[bus].to_numpy(dtype=float),
        )
        err_cols[f"BUS{bus}_VM_ERR_PU"] = vm_err
        err_cols[f"BUS{bus}_VA_ERR_DEG"] = va_err

        def _rmse(x: np.ndarray) -> float:
            return float(np.sqrt(np.nanmean(x**2)))

        rows.append(
            {
                "bus": bus,
                "is_fault_bus": int(bus == fault_bus),
                "vm_rmse_pu": _rmse(vm_err),
                "vm_mae_pu": float(np.nanmean(np.abs(vm_err))),
                "va_rmse_deg": _rmse(va_err),
                "va_mae_deg": float(np.nanmean(np.abs(va_err))),
                "post_fault_vm_rmse_pu": _rmse(vm_err[post_mask]),
                "post_fault_va_rmse_deg": _rmse(va_err[post_mask]),
            }
        )

    metrics = pd.DataFrame(rows)
    errors = pd.DataFrame(err_cols)
    return metrics, errors


def _with_timestamp(df: pd.DataFrame, prefix: str, suffix: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [f"{prefix}{int(c)}_{suffix}" for c in out.columns]
    out.insert(0, "timestamp", df.index.to_numpy(dtype=float))
    return out


def _save_plots(
    out_dir: Path,
    fault_bus: int,
    fault_time: float,
    clear_time: float,
    vm_true: pd.DataFrame,
    va_true_deg: pd.DataFrame,
    vm_est: pd.DataFrame,
    va_est_deg: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict[str, Path]:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    t = vm_true.index.to_numpy(dtype=float)
    plot_paths: dict[str, Path] = {}

    def _mark_event(ax):
        ax.axvspan(fault_time, clear_time, color="#d9480f", alpha=0.18, label="fault on")
        ax.axvline(fault_time, color="#d9480f", linewidth=1.0)
        ax.axvline(clear_time, color="#d9480f", linewidth=1.0, linestyle="--")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, vm_true[fault_bus], label=f"BUS{fault_bus} truth", color="#0b7285", linewidth=2)
    ax.plot(t, vm_est[fault_bus], label=f"BUS{fault_bus} estimate", color="#f08c00", linewidth=2)
    _mark_event(ax)
    ax.set_title(f"Hidden Bus {fault_bus} Voltage Magnitude Reconstruction")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage magnitude [p.u.]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = fig_dir / f"bus{fault_bus}_vm_reconstruction.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths["fault_bus_vm"] = path

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, va_true_deg[fault_bus], label=f"BUS{fault_bus} truth", color="#1864ab", linewidth=2)
    ax.plot(t, va_est_deg[fault_bus], label=f"BUS{fault_bus} estimate", color="#e67700", linewidth=2)
    _mark_event(ax)
    ax.set_title(f"Hidden Bus {fault_bus} Voltage Angle Reconstruction")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage angle [deg]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = fig_dir / f"bus{fault_bus}_angle_reconstruction.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths["fault_bus_angle"] = path

    top = metrics.sort_values("post_fault_vm_rmse_pu", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    labels = [f"BUS{int(b)}" for b in top["bus"]]
    ax.bar(labels, top["post_fault_vm_rmse_pu"], color="#087f5b")
    ax.set_title("Largest Hidden-Bus Post-Fault Voltage-Magnitude RMSE")
    ax.set_xlabel("Hidden bus")
    ax.set_ylabel("RMSE [p.u.]")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = fig_dir / "hidden_bus_post_fault_vm_rmse.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths["post_fault_rmse"] = path

    return plot_paths


def _write_report(
    out_dir: Path,
    fault_bus: int,
    pmu_buses: list[int],
    hidden_buses: list[int],
    t_final: float,
    fault_time: float,
    clear_time: float,
    metrics: pd.DataFrame,
    plots: dict[str, Path],
) -> Path:
    top = metrics.sort_values("post_fault_vm_rmse_pu", ascending=False).head(8)
    fault_row = metrics.loc[metrics["bus"] == fault_bus].iloc[0]
    hidden_summary = {
        "hidden_vm_rmse_pu_mean": float(metrics["vm_rmse_pu"].mean()),
        "hidden_vm_rmse_pu_median": float(metrics["vm_rmse_pu"].median()),
        "hidden_va_rmse_deg_mean": float(metrics["va_rmse_deg"].mean()),
        "hidden_va_rmse_deg_median": float(metrics["va_rmse_deg"].median()),
        "post_fault_vm_rmse_pu_mean": float(metrics["post_fault_vm_rmse_pu"].mean()),
        "post_fault_va_rmse_deg_mean": float(metrics["post_fault_va_rmse_deg"].mean()),
    }

    lines = [
        "# ANDES IEEE-39 Fault Reconstruction Experiment",
        "",
        "## Scenario",
        "",
        f"- Simulator: ANDES IEEE-39 full dynamic case.",
        f"- Single event: three-phase fault at non-PMU Bus {fault_bus}.",
        f"- Time horizon: {t_final:.3f} s.",
        f"- Fault applied at t_final / 2 = {fault_time:.3f} s.",
        f"- Fault cleared at {clear_time:.3f} s.",
        f"- Observed PMU buses: {pmu_buses}.",
        f"- Hidden non-PMU buses compared against truth: {hidden_buses}.",
        "",
        "## Estimator",
        "",
        "The estimator receives only the 8 PMU voltage phasors. For each time step,",
        "it solves a Ybus-weighted harmonic extension problem over all 39 buses.",
        "This is a Kirchhoff/topology-constrained state proxy, not a full dynamic",
        "Kalman estimator. PMU buses are Dirichlet boundary conditions; non-PMU",
        "states are inferred from the IEEE-39 admittance graph.",
        "",
        "## Accuracy Summary",
        "",
        f"- Hidden-bus mean voltage RMSE: {hidden_summary['hidden_vm_rmse_pu_mean']:.6f} p.u.",
        f"- Hidden-bus median voltage RMSE: {hidden_summary['hidden_vm_rmse_pu_median']:.6f} p.u.",
        f"- Hidden-bus mean angle RMSE: {hidden_summary['hidden_va_rmse_deg_mean']:.4f} deg.",
        f"- Hidden-bus median angle RMSE: {hidden_summary['hidden_va_rmse_deg_median']:.4f} deg.",
        f"- Post-fault mean voltage RMSE: {hidden_summary['post_fault_vm_rmse_pu_mean']:.6f} p.u.",
        f"- Post-fault mean angle RMSE: {hidden_summary['post_fault_va_rmse_deg_mean']:.4f} deg.",
        f"- Fault Bus {fault_bus} post-fault voltage RMSE: {fault_row['post_fault_vm_rmse_pu']:.6f} p.u.",
        f"- Fault Bus {fault_bus} post-fault angle RMSE: {fault_row['post_fault_va_rmse_deg']:.4f} deg.",
        "",
        "## Worst Hidden Buses By Post-Fault Voltage RMSE",
        "",
        "| bus | post_fault_vm_rmse_pu | post_fault_va_rmse_deg |",
        "|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"| {int(row['bus'])} | {row['post_fault_vm_rmse_pu']:.6f} | "
            f"{row['post_fault_va_rmse_deg']:.4f} |"
        )

    lines += [
        "",
        "## Plots",
        "",
    ]
    for label, path in plots.items():
        rel = path.relative_to(out_dir).as_posix()
        lines.append(f"- {label}: [{rel}]({rel})")

    lines += [
        "",
        "## Files",
        "",
        "- `truth_all_buses.csv`: ANDES all-bus simulated voltage magnitude and angle truth.",
        "- `pmu_observed.csv`: only the 8 PMU buses used by the estimator.",
        "- `non_pmu_truth.csv`: hidden non-PMU truth used for scoring.",
        "- `full_state_estimate.csv`: reconstructed all-39-bus voltage state.",
        "- `non_pmu_errors.csv`: hidden-bus pointwise errors.",
        "- `metrics.csv`: per-hidden-bus RMSE/MAE metrics.",
        "- `metadata.json`: scenario configuration and bus partitions.",
        "",
        "## Interpretation",
        "",
        "The Ybus harmonic estimator is intentionally lightweight and defensible:",
        "it enforces network smoothness and Kirchhoff/topological coupling while",
        "using only the PMU phasors. It should capture the spatial footprint of",
        "the non-PMU fault, but it cannot perfectly reproduce the severe local",
        "voltage collapse at an unobserved faulted bus. That residual is useful",
        "for localization features and is the main motivation for keeping a later",
        "full dynamic Kalman/DAE estimator as an advanced branch.",
        "",
    ]

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_experiment(
    *,
    fault_bus: int = 7,
    t_final: float = 10.0,
    fps: float = 30.0,
    raw_path: Path | str = Path("data/metadata/IEEE 39 Bus Power System.raw"),
    out_dir: Path | str = Path("experiments/andes_fault_bus7"),
) -> ReconstructionResult:
    """Run the complete simulation, reconstruction, scoring, and reporting flow."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = load_case(raw_path)
    timestamps, vm_true, va_true_deg = run_andes_fault(
        fault_bus=fault_bus,
        t_final=t_final,
        fps=fps,
    )
    vm_est, va_est_deg = reconstruct_from_pmus(grid, vm_true, va_true_deg)

    pmu_buses = list(PMU_BUSES)
    hidden_buses = [bus for bus in range(1, 40) if bus not in pmu_buses]
    hidden_buses = [fault_bus] + [bus for bus in hidden_buses if bus != fault_bus]
    fault_time = t_final / 2.0
    clear_time = fault_time + 5.0 / 60.0

    metrics, errors = _error_metrics(
        vm_true,
        va_true_deg,
        vm_est,
        va_est_deg,
        hidden_buses,
        fault_bus,
        fault_time,
    )

    truth_all = pd.concat(
        [
            _with_timestamp(vm_true, "BUS", "VM_TRUE_PU").set_index("timestamp"),
            _with_timestamp(va_true_deg, "BUS", "VA_TRUE_DEG").set_index("timestamp"),
        ],
        axis=1,
    ).reset_index()
    pmu_observed = truth_all[
        ["timestamp"]
        + [f"BUS{bus}_VM_TRUE_PU" for bus in pmu_buses]
        + [f"BUS{bus}_VA_TRUE_DEG" for bus in pmu_buses]
    ]
    non_pmu_truth = truth_all[
        ["timestamp"]
        + [f"BUS{bus}_VM_TRUE_PU" for bus in hidden_buses]
        + [f"BUS{bus}_VA_TRUE_DEG" for bus in hidden_buses]
    ]
    estimate_all = pd.concat(
        [
            _with_timestamp(vm_est, "BUS", "VM_EST_PU").set_index("timestamp"),
            _with_timestamp(va_est_deg, "BUS", "VA_EST_DEG").set_index("timestamp"),
        ],
        axis=1,
    ).reset_index()

    truth_all.to_csv(out_dir / "truth_all_buses.csv", index=False)
    pmu_observed.to_csv(out_dir / "pmu_observed.csv", index=False)
    non_pmu_truth.to_csv(out_dir / "non_pmu_truth.csv", index=False)
    estimate_all.to_csv(out_dir / "full_state_estimate.csv", index=False)
    errors.to_csv(out_dir / "non_pmu_errors.csv", index=False)
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    metadata = {
        "simulator": "ANDES",
        "case": _andes_ieee39_case(),
        "fault_bus": fault_bus,
        "fault_bus_is_pmu": fault_bus in PMU_BUSES,
        "event": "single three-phase-to-ground fault",
        "t_final_sec": t_final,
        "fps": fps,
        "fault_time_sec": fault_time,
        "fault_clear_sec": clear_time,
        "pmu_buses": pmu_buses,
        "hidden_non_pmu_buses": hidden_buses,
        "estimator": "Ybus-weighted harmonic extension with PMU Dirichlet boundary conditions",
        "n_timestamps": int(len(timestamps)),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    plots = _save_plots(
        out_dir,
        fault_bus,
        fault_time,
        clear_time,
        vm_true,
        va_true_deg,
        vm_est,
        va_est_deg,
        metrics,
    )
    report_path = _write_report(
        out_dir,
        fault_bus,
        pmu_buses,
        hidden_buses,
        t_final,
        fault_time,
        clear_time,
        metrics,
        plots,
    )
    return ReconstructionResult(out_dir=out_dir, report_path=report_path, metrics=metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fault-bus", type=int, default=7, help="Non-PMU fault bus")
    parser.add_argument("--t-final", type=float, default=10.0, help="Simulation horizon in seconds")
    parser.add_argument("--fps", type=float, default=30.0, help="Simulation samples per second")
    parser.add_argument(
        "--raw",
        type=Path,
        default=Path("data/metadata/IEEE 39 Bus Power System.raw"),
        help="Project IEEE-39 RAW file used to build Ybus",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory; defaults to experiments/andes_fault_bus<F>",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    out_dir = args.out or Path(f"experiments/andes_fault_bus{args.fault_bus}")
    result = run_experiment(
        fault_bus=args.fault_bus,
        t_final=args.t_final,
        fps=args.fps,
        raw_path=args.raw,
        out_dir=out_dir,
    )
    summary = {
        "out_dir": str(result.out_dir),
        "report": str(result.report_path),
        "hidden_vm_rmse_mean": float(result.metrics["vm_rmse_pu"].mean()),
        "hidden_va_rmse_deg_mean": float(result.metrics["va_rmse_deg"].mean()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
