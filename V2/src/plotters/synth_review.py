"""Engineer-facing plots and markdown summaries for synthetic PMU scenarios.

The plotting layer is intentionally independent from the synthetic generator.
That keeps the simulator easy to test and lets a reviewer regenerate figures
from saved CSV/JSON artifacts without rerunning the random scenario sampler.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHASE_COLORS = {
    "A": "#c43b2f",
    "B": "#2f7ec4",
    "C": "#2f9f5b",
}


def _event_color(label: int) -> str:
    return {
        1: "#d62728",
        2: "#ff7f0e",
        3: "#9467bd",
        4: "#8c564b",
        5: "#7f7f7f",
        6: "#e377c2",
        7: "#bcbd22",
        8: "#17becf",
    }.get(label, "#d9d9d9")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _events_for_bus(events: Iterable[dict[str, Any]], bus: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for event in events:
        affected = set(event.get("affected_buses", event.get("affected_pmu_buses", [])))
        direct = set(event.get("nodes", []))
        dropout = event.get("pmu_bus")
        if bus in affected or bus in direct or bus == dropout:
            selected.append(event)
    return selected


def _shade_events(ax: plt.Axes, events: Iterable[dict[str, Any]], bus: int) -> None:
    ylim = ax.get_ylim()
    used: set[str] = set()
    for event in _events_for_bus(events, bus):
        start = float(event["start_sec"])
        end = float(event["end_sec"])
        label = int(event["label"])
        name = str(event["kind"]).replace("_", " ")
        legend_label = f"{label}: {name}"
        ax.axvspan(
            start,
            end,
            color=_event_color(label),
            alpha=0.14,
            linewidth=0,
            label=None if legend_label in used else legend_label,
        )
        used.add(legend_label)
    ax.set_ylim(*ylim)
    if used:
        ax.legend(loc="best", fontsize=8)


def _time_seconds(df: pd.DataFrame) -> np.ndarray:
    if "TIMESTAMP" not in df.columns:
        return np.arange(len(df), dtype=float)
    ts = pd.to_numeric(df["TIMESTAMP"], errors="coerce").to_numpy(dtype=float)
    if np.nanmax(ts) > 1e6:
        ts = ts - np.nanmin(ts)
    return ts


def plot_voltage_phases(
    df: pd.DataFrame,
    bus: int,
    events: Iterable[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot phase voltage magnitudes for one PMU bus."""

    _ensure_dir(output_dir)
    t = _time_seconds(df)
    prefix = f"BUS{bus}"
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    for phase in ("A", "B", "C"):
        col = f"{prefix}_V{phase}_MAG"
        if col in df.columns:
            ax.plot(t, df[col], lw=1.1, color=PHASE_COLORS[phase], label=f"V{phase} mag")
    ax.set_title(f"SIM voltage magnitude review: Bus {bus}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Line-neutral RMS voltage [V]")
    ax.grid(True, alpha=0.25)
    _shade_events(ax, events, bus)
    path = output_dir / f"node_{bus}_v_(a,b,c)_mag_vs_time.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_current_phases(
    df: pd.DataFrame,
    bus: int,
    events: Iterable[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot phase current magnitudes for one PMU bus."""

    _ensure_dir(output_dir)
    t = _time_seconds(df)
    prefix = f"BUS{bus}"
    fig, ax = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    for phase in ("A", "B", "C"):
        col = f"{prefix}_I{phase}_MAG"
        if col in df.columns:
            ax.plot(t, df[col], lw=1.1, color=PHASE_COLORS[phase], label=f"I{phase} mag")
    ax.set_title(f"SIM current magnitude review: Bus {bus}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current magnitude [A]")
    ax.grid(True, alpha=0.25)
    _shade_events(ax, events, bus)
    path = output_dir / f"node_{bus}_i_(a,b,c)_mag_vs_time.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_frequency_rocof(
    df: pd.DataFrame,
    bus: int,
    events: Iterable[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot frequency and ROCOF for one PMU bus."""

    _ensure_dir(output_dir)
    t = _time_seconds(df)
    prefix = f"BUS{bus}"
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.2), sharex=True, constrained_layout=True)
    freq_col = f"{prefix}_Freq"
    rocof_col = f"{prefix}_ROCOF"
    if freq_col in df.columns:
        axes[0].plot(t, df[freq_col], lw=1.1, color="#1f4e79")
    axes[0].set_ylabel("Frequency [Hz]")
    axes[0].grid(True, alpha=0.25)
    _shade_events(axes[0], events, bus)
    if rocof_col in df.columns:
        axes[1].plot(t, df[rocof_col], lw=1.1, color="#a64d00")
    axes[1].set_ylabel("ROCOF [Hz/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].grid(True, alpha=0.25)
    _shade_events(axes[1], events, bus)
    fig.suptitle(f"SIM frequency and ROCOF review: Bus {bus}")
    path = output_dir / f"node_{bus}_freq_rocof_vs_time.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _spectral_layout(buses: list[int], branches: list[tuple[int, int]]) -> dict[int, tuple[float, float]]:
    """Small dependency-free graph layout for the IEEE 39 one-line plot."""

    index = {bus: i for i, bus in enumerate(buses)}
    adjacency = np.zeros((len(buses), len(buses)), dtype=float)
    for left, right in branches:
        if left in index and right in index:
            i = index[left]
            j = index[right]
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency
    try:
        _, vectors = np.linalg.eigh(laplacian)
        xy = vectors[:, 1:3]
    except np.linalg.LinAlgError:
        angles = np.linspace(0, 2 * math.pi, len(buses), endpoint=False)
        xy = np.column_stack([np.cos(angles), np.sin(angles)])
    xy = xy - xy.mean(axis=0, keepdims=True)
    scale = np.maximum(np.abs(xy).max(axis=0), 1e-9)
    xy = xy / scale
    return {bus: (float(xy[index[bus], 0]), float(xy[index[bus], 1])) for bus in buses}

def write_engineering_report(
    scenario: dict[str, Any],
    csv_files: dict[int, str],
    plot_files: list[str],
    output_dir: Path,
) -> Path:
    """Write a compact markdown report for engineering review."""

    _ensure_dir(output_dir)
    events = scenario.get("events", [])
    lines = [
        f"# {scenario['scenario_id']} Synthetic PMU Scenario Review",
        "",
        "## Purpose",
        "This artifact documents a synthetic IEEE 39 bus PMU scenario generated for "
        "fault-detection, classification, and localization model training. The "
        "scenario uses IEEE 39 topology and metadata only; it does not sample "
        "hackathon raw measurements.",
        "",
        "## Electrical Assumptions",
        f"- Nominal frequency: {scenario.get('nominal_frequency_hz', 60.0)} Hz.",
        "- Voltage magnitudes are line-neutral RMS volts.",
        "- Phase angles are electrical degrees with balanced ABC separation plus event perturbations.",
        "- Current magnitudes are amperes from a topology/reactance-aware surrogate load-flow response.",
        "- Event propagation uses the IEEE 39 branch model and per-unit branch reactance from metadata when available.",
        "- DATA_PRESENT is set to 0 only for synthetic telemetry dropouts/missing-data intervals.",
        "",
        "## Events",
    ]
    for event in events:
        nodes = event.get("nodes") or []
        line = event.get("line")
        location = f"nodes {nodes}" if nodes else "system-wide response"
        if line:
            location = f"line {line[0]}-{line[1]}"
        if event.get("pmu_bus"):
            location = f"PMU bus {event['pmu_bus']}"
        lines.extend(
            [
                f"- Label {event['label']} `{event['kind']}` at {location}, "
                f"{event['start_sec']:.3f}-{event['end_sec']:.3f} s.",
                f"  Severity/amplitude: {event.get('severity', 'n/a')}; affected buses: {event.get('affected_buses', [])}.",
                f"  Engineering note: {event.get('engineering_note', 'n/a')}",
            ]
        )
    lines.extend(["", "## CSV Outputs"])
    for bus, file_name in sorted(csv_files.items()):
        lines.append(f"- Bus {bus}: `{file_name}`")
    lines.extend(["", "## Figures"])
    for plot in plot_files:
        lines.append(f"- `{plot}`")

    report_path = output_dir / "engineering_review.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
