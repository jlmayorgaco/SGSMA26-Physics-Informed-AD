from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config.constants import TIMESTAMP_COLUMN
from src.plotter.common import save_figure, style_axis, maybe_add_legend
from src.config.models import BusData


def plot_bus_overview(bus: BusData, out_dir: Path, dpi: int = 220) -> None:
    df = bus.df

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    axes[0].plot(df[TIMESTAMP_COLUMN], df["VA_mag"], label="VA_mag")
    axes[0].plot(df[TIMESTAMP_COLUMN], df["VB_mag"], label="VB_mag", alpha=0.8)
    axes[0].plot(df[TIMESTAMP_COLUMN], df["VC_mag"], label="VC_mag", alpha=0.8)
    style_axis(axes[0], title=f"{bus.bus_id} | Voltage magnitude", ylabel="Voltage mag")
    maybe_add_legend(axes[0], loc="upper right", ncol=3)

    axes[1].plot(df[TIMESTAMP_COLUMN], df["IA_mag"], label="IA_mag")
    axes[1].plot(df[TIMESTAMP_COLUMN], df["IB_mag"], label="IB_mag", alpha=0.8)
    axes[1].plot(df[TIMESTAMP_COLUMN], df["IC_mag"], label="IC_mag", alpha=0.8)
    style_axis(axes[1], title=f"{bus.bus_id} | Current magnitude", ylabel="Current mag")
    maybe_add_legend(axes[1], loc="upper right", ncol=3)

    axes[2].plot(df[TIMESTAMP_COLUMN], df["Frequency"], label="Frequency")
    axes[2].plot(df[TIMESTAMP_COLUMN], df["ROCOF"], label="ROCOF", alpha=0.85)
    style_axis(axes[2], title=f"{bus.bus_id} | Frequency and ROCOF", ylabel="Freq / ROCOF")
    maybe_add_legend(axes[2], loc="upper right", ncol=2)

    if "DATA_PRESENT" in df.columns:
        axes[3].step(df[TIMESTAMP_COLUMN], df["DATA_PRESENT"], where="post", label="DATA_PRESENT")
    if "Event" in df.columns:
        axes[3].step(df[TIMESTAMP_COLUMN], df["Event"], where="post", label="Event", alpha=0.85)
    style_axis(axes[3], title=f"{bus.bus_id} | Metadata", xlabel="Time [s]", ylabel="Meta")
    maybe_add_legend(axes[3], loc="upper right", ncol=2)

    fig.tight_layout()
    save_figure(fig, out_dir / f"{bus.bus_id}_overview.png", dpi=dpi)


def plot_bus_distribution_panels(bus: BusData, out_dir: Path, dpi: int = 220) -> None:
    df = bus.df
    chosen = ["VA_mag", "IA_mag", "Frequency", "ROCOF"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 14))

    for row, col in enumerate(chosen):
        values = df[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]

        axes[row, 0].hist(values, bins=100)
        style_axis(axes[row, 0], title=f"{bus.bus_id} | {col} histogram")

        if len(values) > 1:
            centered = values - np.mean(values)
            pxx = np.abs(np.fft.rfft(centered)) ** 2 / max(len(centered), 1)
            f = np.fft.rfftfreq(len(centered), d=1.0 / max(bus.sampling_rate_hz, 1e-12))
            top_idx = np.argsort(pxx[1:])[::-1][:5] + 1 if len(pxx) > 1 else np.array([0])
            axes[row, 1].stem(f[top_idx], pxx[top_idx], basefmt=" ")
        style_axis(axes[row, 1], title=f"{bus.bus_id} | {col} dominant spectral lines", xlabel="Frequency [Hz]")

    fig.tight_layout()
    save_figure(fig, out_dir / f"{bus.bus_id}_distribution_panels.png", dpi=dpi)


def plot_bus_signal_window(
    bus: BusData,
    column: str,
    out_path: Path,
    start_time: float | None = None,
    end_time: float | None = None,
    dpi: int = 220,
) -> None:
    df = bus.df

    if start_time is not None and end_time is not None:
        mask = (df[TIMESTAMP_COLUMN] >= start_time) & (df[TIMESTAMP_COLUMN] <= end_time)
        data = df.loc[mask]
    else:
        data = df

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(data[TIMESTAMP_COLUMN], data[column], label=column)
    style_axis(ax, title=f"{bus.bus_id} | {column}", xlabel="Time [s]", ylabel=column)
    maybe_add_legend(ax)

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)


def plot_bus_power(bus_df: pd.DataFrame, bus_id: str, out_dir: Path, dpi: int = 220) -> None:
    required = ["P_total", "Q_total"]
    if not all(col in bus_df.columns for col in required):
        return

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    axes[0].plot(bus_df[TIMESTAMP_COLUMN], bus_df["P_total"], label="P_total")
    style_axis(axes[0], title=f"{bus_id} | Active power", ylabel="P")
    maybe_add_legend(axes[0])

    axes[1].plot(bus_df[TIMESTAMP_COLUMN], bus_df["Q_total"], label="Q_total")
    style_axis(axes[1], title=f"{bus_id} | Reactive power", xlabel="Time [s]", ylabel="Q")
    maybe_add_legend(axes[1])

    fig.tight_layout()
    save_figure(fig, out_dir / f"{bus_id}_power_total.png", dpi=dpi)


def plot_bus_hilbert_features(
    time: np.ndarray,
    envelope: np.ndarray,
    phase_rad: np.ndarray,
    title_prefix: str,
    out_path: Path,
    dpi: int = 220,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    axes[0].plot(time, envelope, label="Envelope")
    style_axis(axes[0], title=f"{title_prefix} | Hilbert envelope", ylabel="Envelope")
    maybe_add_legend(axes[0])

    axes[1].plot(time, phase_rad, label="Instantaneous phase")
    style_axis(axes[1], title=f"{title_prefix} | Hilbert phase", xlabel="Time [s]", ylabel="Phase [rad]")
    maybe_add_legend(axes[1])

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)