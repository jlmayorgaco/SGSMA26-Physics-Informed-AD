from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.config.constants import TIMESTAMP_COLUMN
from src.config.models import BusData
from src.plotter.common import add_event_span, save_figure, style_axis, maybe_add_legend


def plot_event_overlay_frequency(
    buses: list[BusData],
    event_span: dict,
    out_path: Path,
    context_seconds: float = 1.0,
    dpi: int = 220,
) -> None:
    start_time = event_span["start_time"] - context_seconds
    end_time = event_span["end_time"] + context_seconds

    fig, ax = plt.subplots(figsize=(16, 6))

    for bus in buses:
        df = bus.df
        mask = (df[TIMESTAMP_COLUMN] >= start_time) & (df[TIMESTAMP_COLUMN] <= end_time)
        ax.plot(df.loc[mask, TIMESTAMP_COLUMN], df.loc[mask, "Frequency"], label=bus.bus_id)

    add_event_span(ax, event_span["start_time"], event_span["end_time"])
    style_axis(
        ax,
        title=f"Frequency overlay | Event {event_span['event_id']} ({event_span['label']})",
        xlabel="Time [s]",
        ylabel="Frequency",
    )
    maybe_add_legend(ax, loc="upper right", ncol=4)

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)


def plot_event_overlay_voltage(
    buses: list[BusData],
    event_span: dict,
    out_path: Path,
    context_seconds: float = 1.0,
    dpi: int = 220,
) -> None:
    start_time = event_span["start_time"] - context_seconds
    end_time = event_span["end_time"] + context_seconds

    fig, ax = plt.subplots(figsize=(16, 6))

    for bus in buses:
        df = bus.df
        mask = (df[TIMESTAMP_COLUMN] >= start_time) & (df[TIMESTAMP_COLUMN] <= end_time)
        ax.plot(df.loc[mask, TIMESTAMP_COLUMN], df.loc[mask, "VA_mag"], label=bus.bus_id)

    add_event_span(ax, event_span["start_time"], event_span["end_time"])
    style_axis(
        ax,
        title=f"VA_mag overlay | Event {event_span['event_id']} ({event_span['label']})",
        xlabel="Time [s]",
        ylabel="VA_mag",
    )
    maybe_add_legend(ax, loc="upper right", ncol=4)

    fig.tight_layout()
    save_figure(fig, out_path, dpi=dpi)


def plot_event_bus_window(
    bus: BusData,
    event_span: dict,
    out_dir: Path,
    context_seconds: float = 1.0,
    dpi: int = 220,
) -> None:
    df = bus.df
    start_time = event_span["start_time"] - context_seconds
    end_time = event_span["end_time"] + context_seconds
    mask = (df[TIMESTAMP_COLUMN] >= start_time) & (df[TIMESTAMP_COLUMN] <= end_time)
    data = df.loc[mask]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(data[TIMESTAMP_COLUMN], data["VA_mag"], label="VA_mag")
    axes[0].plot(data[TIMESTAMP_COLUMN], data["VB_mag"], label="VB_mag", alpha=0.8)
    axes[0].plot(data[TIMESTAMP_COLUMN], data["VC_mag"], label="VC_mag", alpha=0.8)
    add_event_span(axes[0], event_span["start_time"], event_span["end_time"])
    style_axis(axes[0], title=f"{bus.bus_id} | Voltage around event", ylabel="Voltage mag")
    maybe_add_legend(axes[0], loc="upper right", ncol=3)

    axes[1].plot(data[TIMESTAMP_COLUMN], data["IA_mag"], label="IA_mag")
    axes[1].plot(data[TIMESTAMP_COLUMN], data["IB_mag"], label="IB_mag", alpha=0.8)
    axes[1].plot(data[TIMESTAMP_COLUMN], data["IC_mag"], label="IC_mag", alpha=0.8)
    add_event_span(axes[1], event_span["start_time"], event_span["end_time"])
    style_axis(axes[1], title=f"{bus.bus_id} | Current around event", ylabel="Current mag")
    maybe_add_legend(axes[1], loc="upper right", ncol=3)

    axes[2].plot(data[TIMESTAMP_COLUMN], data["Frequency"], label="Frequency")
    axes[2].plot(data[TIMESTAMP_COLUMN], data["ROCOF"], label="ROCOF", alpha=0.85)
    add_event_span(axes[2], event_span["start_time"], event_span["end_time"])
    style_axis(axes[2], title=f"{bus.bus_id} | Frequency / ROCOF around event", xlabel="Time [s]", ylabel="Freq / ROCOF")
    maybe_add_legend(axes[2], loc="upper right", ncol=2)

    fig.tight_layout()

    safe_label = str(event_span["label"]).replace(" ", "_").replace("/", "_")
    save_figure(
        fig,
        out_dir / f"{bus.bus_id}_event_{event_span['event_id']}_{safe_label}.png",
        dpi=dpi,
    )