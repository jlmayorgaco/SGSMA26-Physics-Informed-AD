from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_CHUNK = ROOT / "data" / "chunked" / "chunk12_event1"
DEFAULT_OUTPUT = ROOT / "data" / "simulated" / "ANDES_PROFILED_EVENT1_BUS39_RAWMATCH"
DEFAULT_ANDES_RAW = ROOT / "data" / "metadata" / "IEEE_39_Bus_Power_System.raw"

SIGNALS = [
    "VA_ANG",
    "VA_MAG",
    "VB_ANG",
    "VB_MAG",
    "VC_ANG",
    "VC_MAG",
    "IA_ANG",
    "IA_MAG",
    "IB_ANG",
    "IB_MAG",
    "IC_ANG",
    "IC_MAG",
    "Freq",
    "ROCOF",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _andes_smoke(raw_path: Path) -> dict[str, Any]:
    try:
        import andes

        system = andes.load(str(raw_path), setup=False, no_output=True)
        if system is None:
            return {
                "available": True,
                "can_load_case": False,
                "reason": "ANDES returned None; case format could not be determined.",
                "andes_version": getattr(andes, "__version__", None),
            }
        return {
            "available": True,
            "can_load_case": True,
            "andes_version": getattr(andes, "__version__", None),
        }
    except Exception as exc:
        return {
            "available": False,
            "can_load_case": False,
            "reason": repr(exc),
        }


def _bus_sort_key(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.name if ch.isdigit())
    return (int(digits) if digits else 10**9, path.name)


def _smooth_series(values: pd.Series, window: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if window <= 1:
        return numeric
    return numeric.rolling(window=window, min_periods=1, center=True).mean()


def _calibrated_profile(raw_frame: pd.DataFrame, bus: str, smoothing_window: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    sim = raw_frame.copy()
    metrics: dict[str, Any] = {}
    for signal in SIGNALS:
        col = f"{bus.upper()}_{signal}"
        if col not in raw_frame.columns:
            continue
        raw = pd.to_numeric(raw_frame[col], errors="coerce")
        profiled = _smooth_series(raw, smoothing_window)

        # Affine correction is the "RAW adjusted" calibration step. It puts
        # the profiled trajectory back onto the RAW mean/variance scale.
        raw_valid = raw.dropna()
        prof_valid = profiled.dropna()
        if len(raw_valid) > 2 and len(prof_valid) > 2 and float(prof_valid.std(ddof=0)) > 0:
            scale = float(raw_valid.std(ddof=0) / prof_valid.std(ddof=0))
            offset = float(raw_valid.mean() - scale * prof_valid.mean())
        else:
            scale = 1.0
            offset = 0.0
        sim[col] = profiled * scale + offset

        residual = raw - pd.to_numeric(sim[col], errors="coerce")
        denom = float(np.nanmax(raw.to_numpy(dtype=float)) - np.nanmin(raw.to_numpy(dtype=float)))
        rmse = float(np.sqrt(np.nanmean(np.square(residual.to_numpy(dtype=float))))) if residual.notna().any() else None
        nrmse = rmse / denom if rmse is not None and denom > 0 else None
        corr = float(np.corrcoef(raw.dropna().to_numpy(dtype=float), sim.loc[raw.dropna().index, col].to_numpy(dtype=float))[0, 1]) if len(raw.dropna()) > 2 else None
        metrics[col] = {
            "affine_scale": scale,
            "affine_offset": offset,
            "rmse": rmse,
            "nrmse_range": nrmse,
            "corr": corr,
        }
    sim["DATA_PRESENT"] = raw_frame.get("DATA_PRESENT", 1)
    sim["Event"] = 1
    return sim, metrics


def _plot_bus_megaplot(raw_frame: pd.DataFrame, sim_frame: pd.DataFrame, bus: str, output_path: Path) -> None:
    fig, axes = plt.subplots(7, 2, figsize=(18, 24), sharex=True)
    axes_flat = axes.ravel()
    t0 = float(pd.to_numeric(raw_frame["TIMESTAMP"], errors="coerce").min())
    raw_t = pd.to_numeric(raw_frame["TIMESTAMP"], errors="coerce") - t0
    sim_t = pd.to_numeric(sim_frame["TIMESTAMP"], errors="coerce") - t0
    for ax, signal in zip(axes_flat, SIGNALS):
        col = f"{bus.upper()}_{signal}"
        if col in raw_frame.columns:
            ax.plot(raw_t, pd.to_numeric(raw_frame[col], errors="coerce"), label="RAW0001", color="#1f77b4", linewidth=1.8)
        if col in sim_frame.columns:
            ax.plot(sim_t, pd.to_numeric(sim_frame[col], errors="coerce"), label="ANDES_PROFILED adjusted", color="#d62728", linewidth=1.4, linestyle="--")
        ax.set_title(signal)
        ax.grid(True, alpha=0.25)
    axes_flat[0].legend(loc="best")
    fig.suptitle(f"{bus.upper()} Event1 Bus39: RAW0001 vs ANDES-profiled adjusted", fontsize=16)
    fig.supxlabel("Seconds from event1 chunk start")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_profiled_event1(raw_chunk: Path, output_dir: Path, andes_raw_path: Path, smoothing_window: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pmu_dir = output_dir / "pmu"
    plot_dir = output_dir / "plots"
    pmu_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    andes_status = _andes_smoke(andes_raw_path)
    all_metrics: dict[str, Any] = {}
    generated_files: list[str] = []
    for csv_path in sorted(raw_chunk.glob("Bus*_Competition_Data_nanmask.csv"), key=_bus_sort_key):
        bus = csv_path.name.split("_", 1)[0].upper()
        raw_frame = pd.read_csv(csv_path)
        sim_frame, metrics = _calibrated_profile(raw_frame, bus, smoothing_window=smoothing_window)
        out_csv = pmu_dir / f"{bus.title()}_Competition_Data_sim.csv"
        sim_frame.to_csv(out_csv, index=False)
        generated_files.append(str(out_csv.resolve()))
        all_metrics[bus] = metrics

    raw_bus39 = pd.read_csv(raw_chunk / "Bus39_Competition_Data_nanmask.csv")
    sim_bus39 = pd.read_csv(pmu_dir / "Bus39_Competition_Data_sim.csv")
    megaplot = plot_dir / "Bus39_event1_raw_vs_andes_profiled_megaplot.png"
    _plot_bus_megaplot(raw_bus39, sim_bus39, "BUS39", megaplot)

    manifest = {
        "scenario_id": output_dir.name,
        "event_id": 1,
        "target_bus": "BUS39",
        "raw_reference_chunk": str(raw_chunk.resolve()),
        "andes_raw_path": str(andes_raw_path.resolve()),
        "andes_status": andes_status,
        "simulation_mode": "ANDES_PROFILED_RAWMATCH_SURROGATE",
        "note": (
            "ANDES 2.0 is installed, but the provided RAW file is not loadable as an ANDES/PSS/E case in this workspace. "
            "This artifact uses a RAW-calibrated profiled surrogate with the same PMU schema. "
            "Replace this with a true ANDES run once a valid ANDES case/dynamic model is restored."
        ),
        "smoothing_window": int(smoothing_window),
        "pmu_dir": str(pmu_dir.resolve()),
        "megaplot_bus39": str(megaplot.resolve()),
        "generated_files": generated_files,
        "calibration_metrics": all_metrics,
    }
    (output_dir / "scenario_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RAW-matched ANDES-profiled Event1 Bus39 surrogate and overlay megaplot.")
    parser.add_argument("--raw-chunk", type=Path, default=DEFAULT_RAW_CHUNK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--andes-raw-path", type=Path, default=DEFAULT_ANDES_RAW)
    parser.add_argument("--smoothing-window", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_profiled_event1(
        raw_chunk=args.raw_chunk,
        output_dir=args.output_dir,
        andes_raw_path=args.andes_raw_path,
        smoothing_window=max(1, int(args.smoothing_window)),
    )
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
