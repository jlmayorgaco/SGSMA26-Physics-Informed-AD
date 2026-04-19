from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

PLOT_DPI = 150
FIG_WIDE = (12, 4.5)
FIG_EVENT = (13, 7)

EVENT_COLOR_MAP = {
    0: "#d9d9d9",
    1: "#fc8d62",
    2: "#fdb462",
    3: "#66c2a5",
    4: "#8da0cb",
    5: "#e78ac3",
    6: "#ffd92f",
    7: "#a6d854",
    8: "#e5c494",
}

EVENT_LABELS = {
    0: "Normal",
    1: "Fault",
    2: "Line outage",
    3: "Generation change",
    4: "Load change",
    5: "Missing data",
    6: "Missing + physical",
    7: "Bad data",
    8: "Unknown",
}


@dataclass(frozen=True)
class PMUCanonicalFrame:
    bus_name: str
    df: pd.DataFrame


def _norm(name: str) -> str:
    return str(name).strip().upper().replace(" ", "").replace("-", "_")


def _infer_bus_token(columns: Iterable[str]) -> str | None:
    for col in columns:
        match = re.match(r"^(BUS\d+)_", _norm(col))
        if match:
            return match.group(1)
    return None


def _find_column_case_insensitive(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def _canonical_rename_map(df: pd.DataFrame) -> tuple[dict[str, str], str | None]:
    rename_map: dict[str, str] = {}
    bus_token = _infer_bus_token(df.columns)

    aliases = {
        "FREQUENCY": "FREQ",
        "FREQ": "FREQ",
        "EVENT": "EVENT",
        "DATA_PRESENT": "DATA_PRESENT",
        "TIMESTAMP": "TIMESTAMP",
        "VA_ANG": "VA_ANG",
        "VA_MAG": "VA_MAG",
        "VB_ANG": "VB_ANG",
        "VB_MAG": "VB_MAG",
        "VC_ANG": "VC_ANG",
        "VC_MAG": "VC_MAG",
        "IA_ANG": "IA_ANG",
        "IA_MAG": "IA_MAG",
        "IB_ANG": "IB_ANG",
        "IB_MAG": "IB_MAG",
        "IC_ANG": "IC_ANG",
        "IC_MAG": "IC_MAG",
        "ROCOF": "ROCOF",
    }

    for col in df.columns:
        norm = _norm(col)

        if bus_token and norm.startswith(f"{bus_token}_"):
            norm = norm[len(bus_token) + 1 :]

        canonical = aliases.get(norm)
        if canonical is not None:
            rename_map[col] = canonical

    return rename_map, bus_token


def canonicalize_bus_frame(csv_path: Path) -> PMUCanonicalFrame:
    raw = pd.read_csv(csv_path)
    rename_map, bus_token = _canonical_rename_map(raw)
    df = raw.rename(columns=rename_map).copy()

    required = [
        "TIMESTAMP",
        "VA_ANG", "VA_MAG",
        "VB_ANG", "VB_MAG",
        "VC_ANG", "VC_MAG",
        "IA_ANG", "IA_MAG",
        "IB_ANG", "IB_MAG",
        "IC_ANG", "IC_MAG",
        "FREQ", "ROCOF",
        "DATA_PRESENT", "EVENT",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[required].copy()
    df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce")
    df = df.dropna(subset=["TIMESTAMP"]).sort_values("TIMESTAMP").reset_index(drop=True)

    numeric_cols = [c for c in df.columns if c != "TIMESTAMP"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bus_name = csv_path.stem
    if bus_token:
        bus_name = bus_token.replace("BUS", "Bus")

    return PMUCanonicalFrame(bus_name=bus_name, df=df)


def _estimate_dt(t: np.ndarray) -> float:
    if len(t) < 2:
        return 0.033
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 0.033
    return float(np.median(dt))


def _add_event_background(ax: plt.Axes, t: np.ndarray, event: np.ndarray) -> None:
    if len(t) == 0 or len(event) == 0:
        return

    dt = _estimate_dt(t)
    start = 0
    current = int(event[0])

    for i in range(1, len(event)):
        if int(event[i]) != current:
            x0 = float(t[start])
            x1 = float(t[i - 1] + dt)
            ax.axvspan(
                x0,
                x1,
                color=EVENT_COLOR_MAP.get(current, "#cccccc"),
                alpha=0.12,
                lw=0,
            )
            start = i
            current = int(event[i])

    ax.axvspan(
        float(t[start]),
        float(t[-1] + dt),
        color=EVENT_COLOR_MAP.get(current, "#cccccc"),
        alpha=0.12,
        lw=0,
    )


def _plot_group(
    df: pd.DataFrame,
    bus_name: str,
    cols: list[str],
    title: str,
    ylabel: str,
    out_path: Path,
) -> bool:
    existing = [c for c in cols if c in df.columns and df[c].notna().any()]
    if not existing:
        LOGGER.warning("No columns found for %s in %s", title, bus_name)
        return False

    t = df["TIMESTAMP"].to_numpy(dtype=float)
    ev = df["EVENT"].fillna(0).to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    _add_event_background(ax, t, ev)

    for col in existing:
        y = df[col].to_numpy(dtype=float)
        ax.plot(t, y, linewidth=1.1, label=col)

    ax.set_title(f"{bus_name} - {title}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=min(3, max(1, len(existing))))

    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    return True


def _plot_scalar(
    df: pd.DataFrame,
    bus_name: str,
    col: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> bool:
    if col not in df.columns or not df[col].notna().any():
        LOGGER.warning("No column found for %s in %s", title, bus_name)
        return False

    t = df["TIMESTAMP"].to_numpy(dtype=float)
    y = df[col].to_numpy(dtype=float)
    ev = df["EVENT"].fillna(0).to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=FIG_WIDE)
    _add_event_background(ax, t, ev)

    ax.plot(t, y, linewidth=1.2, label=col)
    ax.set_title(f"{bus_name} - {title}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    return True


def plot_bus_chunk(csv_path: Path) -> list[Path]:
    canonical = canonicalize_bus_frame(csv_path)
    bus_name = canonical.bus_name
    df = canonical.df

    created: list[Path] = []

    group_specs = [
        ("Current Mag", ["IA_MAG", "IB_MAG", "IC_MAG"], "Current magnitude", f"{bus_name}_Current_Mag.png"),
        ("Voltage Mag", ["VA_MAG", "VB_MAG", "VC_MAG"], "Voltage magnitude", f"{bus_name}_Voltage_Mag.png"),
        ("Voltage Phase", ["VA_ANG", "VB_ANG", "VC_ANG"], "Angle [deg]", f"{bus_name}_Voltage_Phase.png"),
        ("Current Phase", ["IA_ANG", "IB_ANG", "IC_ANG"], "Angle [deg]", f"{bus_name}_Current_Phase.png"),
    ]

    for title, cols, ylabel, filename in group_specs:
        out_path = csv_path.with_name(filename)
        if _plot_group(df=df, bus_name=bus_name, cols=cols, title=title, ylabel=ylabel, out_path=out_path):
            created.append(out_path)

    scalar_specs = [
        ("FREQ", "Frequency", "Frequency [Hz]", f"{bus_name}_Frequency.png"),
        ("ROCOF", "ROCOF", "ROCOF [Hz/s]", f"{bus_name}_ROCOF.png"),
    ]

    for col, title, ylabel, filename in scalar_specs:
        out_path = csv_path.with_name(filename)
        if _plot_scalar(df=df, bus_name=bus_name, col=col, title=title, ylabel=ylabel, out_path=out_path):
            created.append(out_path)

    return created


def plot_chunk_events_by_bus(chunk_dir: Path, bus_csvs: list[Path]) -> Path | None:
    rows: list[tuple[str, np.ndarray, np.ndarray]] = []

    for csv_path in sorted(bus_csvs):
        canonical = canonicalize_bus_frame(csv_path)
        df = canonical.df
        t = df["TIMESTAMP"].to_numpy(dtype=float)
        ev = df["EVENT"].fillna(0).to_numpy(dtype=int)
        rows.append((canonical.bus_name, t, ev))

    if not rows:
        return None

    fig, ax = plt.subplots(figsize=FIG_EVENT)
    y_positions = np.arange(len(rows))
    ax.set_yticks(y_positions)
    ax.set_yticklabels([row[0] for row in rows])

    shown_labels: set[int] = set()

    for y_idx, (bus_name, t, ev) in enumerate(rows):
        if len(t) == 0:
            continue

        dt = _estimate_dt(t)
        start = 0
        current = int(ev[0])

        for i in range(1, len(ev)):
            if int(ev[i]) != current:
                x0 = float(t[start])
                width = float((t[i - 1] + dt) - t[start])
                label = EVENT_LABELS.get(current, f"Event {current}") if current not in shown_labels else None
                ax.broken_barh(
                    [(x0, width)],
                    (y_idx - 0.35, 0.7),
                    facecolors=EVENT_COLOR_MAP.get(current, "#cccccc"),
                    edgecolors="black",
                    linewidth=0.35,
                    label=label,
                )
                shown_labels.add(current)
                start = i
                current = int(ev[i])

        x0 = float(t[start])
        width = float((t[-1] + dt) - t[start])
        label = EVENT_LABELS.get(current, f"Event {current}") if current not in shown_labels else None
        ax.broken_barh(
            [(x0, width)],
            (y_idx - 0.35, 0.7),
            facecolors=EVENT_COLOR_MAP.get(current, "#cccccc"),
            edgecolors="black",
            linewidth=0.35,
            label=label,
        )
        shown_labels.add(current)

        abnormal_idx = np.where(ev != 0)[0]
        for idx in abnormal_idx:
            ax.text(
                float(t[idx]),
                y_idx,
                str(int(ev[idx])),
                fontsize=7,
                va="center",
                ha="center",
                color="black",
            )

    ax.set_title(f"{chunk_dir.name} - Events by bus and time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bus")
    ax.grid(axis="x", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8, ncol=3, loc="upper right")

    out_path = chunk_dir / "chunk_events_by_bus.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI)
    plt.close(fig)
    return out_path


def plot_chunk_directory(chunk_dir: Path) -> dict[str, list[Path] | Path | None]:
    bus_csvs = sorted(chunk_dir.glob("Bus*.csv"))
    if not bus_csvs:
        LOGGER.warning("No Bus*.csv files found in %s", chunk_dir)
        return {"bus_plots": [], "event_plot": None}

    created_bus_plots: list[Path] = []
    for csv_path in bus_csvs:
        created_bus_plots.extend(plot_bus_chunk(csv_path))

    event_plot = plot_chunk_events_by_bus(chunk_dir, bus_csvs)
    return {"bus_plots": created_bus_plots, "event_plot": event_plot}