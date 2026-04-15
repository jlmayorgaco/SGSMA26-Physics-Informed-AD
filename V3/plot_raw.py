from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


BUS_FILES = [
    "Bus2_Competition_Data_nanmask.csv",
    "Bus5_Competition_Data_nanmask.csv",
    "Bus6_Competition_Data_nanmask.csv",
    "Bus10_Competition_Data_nanmask.csv",
    "Bus19_Competition_Data_nanmask.csv",
    "Bus22_Competition_Data_nanmask.csv",
    "Bus29_Competition_Data_nanmask.csv",
    "Bus39_Competition_Data_nanmask.csv",
]

EVENT_LABELS = {
    0: "Normal",
    1: "Fault",
    2: "Line outage",
    3: "Generation change/outage",
    4: "Load change/drop",
    5: "Missing data",
    6: "Missing data + physical event",
    7: "Bad data",
    8: "Unknown event",
}

EVENT_COLORS = {
    1: "#ef4444",
    2: "#f97316",
    3: "#eab308",
    4: "#22c55e",
    5: "#3b82f6",
    6: "#8b5cf6",
    7: "#ec4899",
    8: "#6b7280",
}

PHASE_COLORS = {
    "A": "#2563eb",
    "B": "#16a34a",
    "C": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-node PMU plots with event shading and metadata annotations."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Directory containing the 8 raw PMU CSV files. If omitted, the script auto-discovers it.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory where per-node folders will be created. Defaults to <raw_dir_parent>/plots_raw",
    )
    parser.add_argument(
        "--timeline-xlsx",
        type=Path,
        default=None,
        help="Path to Event Timeline & Location.xlsx. If omitted, the script auto-discovers it.",
    )
    parser.add_argument(
        "--pmu-meta-txt",
        type=Path,
        default=None,
        help="Path to PMUbus_Location.txt / PMUbus_ Location.txt. If omitted, the script auto-discovers it.",
    )
    parser.add_argument(
        "--bus",
        type=str,
        default=None,
        help="Optional single bus/file to plot, e.g. Bus2 or Bus2_Competition_Data_nanmask.csv",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Plot every Nth sample. Use >1 only if rendering becomes too heavy.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="DPI for normal PNGs.",
    )
    parser.add_argument(
        "--hires-dpi",
        type=int,
        default=350,
        help="DPI for the very large Va high-resolution PNG.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving them.",
    )
    return parser.parse_args()


def normalize_bus_argument(bus: str | None) -> str | None:
    if bus is None:
        return None

    bus = bus.strip()
    if bus.endswith(".csv"):
        return bus

    if bus.lower().startswith("bus"):
        return f"{bus}_Competition_Data_nanmask.csv"

    return bus


def looks_like_raw_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and all((path / name).exists() for name in BUS_FILES)


def discover_raw_dir(script_dir: Path) -> Path:
    candidates = [
        script_dir / "data" / "raw",
        script_dir / "raw",
        script_dir.parent / "data" / "raw",
        script_dir.parent / "raw",
        script_dir.parent / "V3" / "data" / "raw",
        script_dir.parent / "V3" / "raw",
        script_dir.parent.parent / "V3" / "data" / "raw",
        script_dir.parent.parent / "V3" / "raw",
    ]
    for candidate in dedupe_paths(candidates):
        if looks_like_raw_dir(candidate):
            return candidate.resolve()

    searched = "\n".join(f"  - {p.resolve()}" for p in dedupe_paths(candidates))
    raise FileNotFoundError(
        "Could not auto-discover the raw data directory.\n"
        f"Searched:\n{searched}\n"
        "Pass it explicitly with --raw-dir .\\V3\\data\\raw"
    )


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def discover_support_file(script_dir: Path, candidate_names: list[str]) -> Path | None:
    candidate_dirs = dedupe_paths([
        script_dir,
        script_dir / "data",
        script_dir / "data" / "raw",
        script_dir.parent,
        script_dir.parent / "data",
        script_dir.parent / "data" / "raw",
        script_dir.parent / "V3",
        script_dir.parent / "V3" / "data",
        script_dir.parent / "V3" / "data" / "raw",
        script_dir.parent.parent,
        script_dir.parent.parent / "V3",
        script_dir.parent.parent / "V3" / "data",
        script_dir.parent.parent / "V3" / "data" / "raw",
    ])

    for directory in candidate_dirs:
        for name in candidate_names:
            path = directory / name
            if path.exists():
                return path.resolve()
    return None


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None, Path | None]:
    script_dir = Path(__file__).resolve().parent

    raw_dir = args.raw_dir.resolve() if args.raw_dir is not None else discover_raw_dir(script_dir)
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (raw_dir.parent / "plots_raw")

    timeline_xlsx = (
        args.timeline_xlsx.resolve()
        if args.timeline_xlsx is not None
        else discover_support_file(
            script_dir,
            [
                "Event Timeline & Location.xlsx",
                "Event_Timeline_&_Location.xlsx",
                "Event Timeline and Location.xlsx",
            ],
        )
    )

    pmu_meta_txt = (
        args.pmu_meta_txt.resolve()
        if args.pmu_meta_txt is not None
        else discover_support_file(
            script_dir,
            [
                "PMUbus_ Location.txt",
                "PMUbus_Location.txt",
                "PMUbus Location.txt",
            ],
        )
    )

    return raw_dir, out_dir, timeline_xlsx, pmu_meta_txt


def extract_bus_id(filename: str) -> str:
    return Path(filename).stem.split("_")[0]


def extract_bus_number(bus_id: str) -> str:
    return re.sub(r"[^0-9]", "", bus_id)


def contiguous_spans_from_values(t: np.ndarray, values: np.ndarray) -> list[tuple[int, float, float]]:
    if len(values) == 0:
        return []

    changes = np.flatnonzero(np.diff(values) != 0) + 1
    bounds = np.r_[0, changes, len(values)]

    spans: list[tuple[int, float, float]] = []
    for i0, i1 in zip(bounds[:-1], bounds[1:]):
        spans.append((int(values[i0]), float(t[i0]), float(t[i1 - 1])))
    return spans


def canonical_column_map_for_bus(bus_id: str) -> dict[str, str]:
    bus_num = extract_bus_number(bus_id)
    prefix = f"BUS{bus_num}_"
    return {
        f"{prefix}VA_MAG": "VA_mag",
        f"{prefix}VA_ANG": "VA_ang",
        f"{prefix}VB_MAG": "VB_mag",
        f"{prefix}VB_ANG": "VB_ang",
        f"{prefix}VC_MAG": "VC_mag",
        f"{prefix}VC_ANG": "VC_ang",
        f"{prefix}IA_MAG": "IA_mag",
        f"{prefix}IA_ANG": "IA_ang",
        f"{prefix}IB_MAG": "IB_mag",
        f"{prefix}IB_ANG": "IB_ang",
        f"{prefix}IC_MAG": "IC_mag",
        f"{prefix}IC_ANG": "IC_ang",
        f"{prefix}Freq": "Frequency",
        f"{prefix}FREQ": "Frequency",
        f"{prefix}ROCOF": "ROCOF",
    }


def normalize_dataframe_columns(df: pd.DataFrame, bus_id: str) -> pd.DataFrame:
    rename_map = canonical_column_map_for_bus(bus_id)
    applicable = {src: dst for src, dst in rename_map.items() if src in df.columns}
    if applicable:
        df = df.rename(columns=applicable)
    return df


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    required = [
        "TIMESTAMP",
        "VA_mag", "VB_mag", "VC_mag",
        "VA_ang", "VB_ang", "VC_ang",
        "IA_mag", "IB_mag", "IC_mag",
        "IA_ang", "IB_ang", "IC_ang",
        "Frequency",
        "ROCOF",
        "DATA_PRESENT",
        "Event",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path.name} is missing required columns after normalization: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def load_csv(csv_path: Path, downsample: int = 1) -> pd.DataFrame:
    bus_id = extract_bus_id(csv_path.name)
    df = pd.read_csv(csv_path)
    df = normalize_dataframe_columns(df, bus_id)
    validate_columns(df, csv_path)

    if downsample > 1:
        df = df.iloc[::downsample].reset_index(drop=True)

    return df.sort_values("TIMESTAMP").reset_index(drop=True)


def load_pmu_metadata(txt_path: Path | None) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    if txt_path is None or not txt_path.exists():
        return metadata

    pattern = re.compile(
        r"'BUS(?P<num>\d+)(?:x1)?'\s*,\s*"
        r"(?P<kv>[-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"(?P<type>\d+)\s*,\s*"
        r"(?P<pu>[-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"(?P<theta>[-+]?\d+(?:\.\d+)?)\s*,\s*"
        r"(?P<pmu>.*)$"
    )

    for line in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line.strip())
        if not match:
            continue

        bus_num = match.group("num")
        bus_id = f"Bus{bus_num}"
        pmu_raw = match.group("pmu").strip()
        pmu_clean = pmu_raw if pmu_raw and pmu_raw != "0" else None

        metadata[bus_id] = {
            "kv": float(match.group("kv")),
            "bus_type": int(match.group("type")),
            "v_pu": float(match.group("pu")),
            "theta_deg": float(match.group("theta")),
            "pmu_location": pmu_clean,
        }

    return metadata


def parse_approx_minute(value: Any) -> float | None:
    if pd.isna(value):
        return None
    text = str(value)
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(match.group(1)) if match else None


def short_timeline_label(record: dict[str, Any]) -> str:
    minute = record.get("approx_minute")
    minute_txt = f"{int(minute)} min" if minute is not None and float(minute).is_integer() else f"{minute:.1f} min"

    event_type = str(record.get("event_type", "")).strip()
    impact = str(record.get("impact", "")).strip()
    location = str(record.get("location", "")).strip()

    if "3lg" in impact.lower() or "line to ground fault" in impact.lower() or "fault" in impact.lower():
        kind = "Fault"
    elif "line outage" in impact.lower():
        kind = "Line outage"
    elif "generation" in impact.lower():
        kind = "Generation change"
    elif "load change" in impact.lower():
        kind = "Load change"
    elif "data drop" in impact.lower():
        kind = "Data drop"
    elif event_type:
        kind = event_type
    else:
        kind = "Event"

    label = f"{minute_txt} | {kind}"
    if location:
        label += f" | {location}"
    return label


def load_timeline_metadata(xlsx_path: Path | None) -> list[dict[str, Any]]:
    if xlsx_path is None or not xlsx_path.exists():
        return []

    df = pd.read_excel(xlsx_path)
    columns = {str(col).strip(): col for col in df.columns}

    event_number_col = next((columns[c] for c in columns if c.lower() == "event number"), None)
    event_type_col = next((columns[c] for c in columns if c.lower() == "event type"), None)
    impact_col = next((columns[c] for c in columns if c.lower().startswith("event imp")), None)
    approx_col = next((columns[c] for c in columns if c.lower().startswith("approximated time")), None)
    location_col = next((columns[c] for c in columns if c.lower() == "event location"), None)

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        raw_number = row[event_number_col] if event_number_col is not None else None
        minute = parse_approx_minute(row[approx_col] if approx_col is not None else None)

        if pd.isna(raw_number) or minute is None:
            continue

        try:
            event_number = int(raw_number)
        except Exception:
            continue

        record = {
            "event_number": event_number,
            "event_type": "" if event_type_col is None or pd.isna(row[event_type_col]) else str(row[event_type_col]).strip(),
            "impact": "" if impact_col is None or pd.isna(row[impact_col]) else str(row[impact_col]).strip(),
            "approx_minute": minute,
            "approx_seconds": minute * 60.0,
            "location": "" if location_col is None or pd.isna(row[location_col]) else str(row[location_col]).strip(),
        }
        record["short_label"] = short_timeline_label(record)
        records.append(record)

    records.sort(key=lambda item: (item["approx_seconds"], item["event_number"]))
    return records


def build_bus_descriptor(bus_id: str, pmu_meta: dict[str, dict[str, Any]]) -> str:
    meta = pmu_meta.get(bus_id)
    if not meta:
        return bus_id

    parts = [bus_id]
    if meta.get("pmu_location"):
        parts.append(str(meta["pmu_location"]))
    parts.append(f'{meta["kv"]:.1f} kV')
    parts.append(f'Vbase={meta["v_pu"]:.4f} p.u.')
    parts.append(f'θ={meta["theta_deg"]:+.2f}°')
    return " | ".join(parts)


def build_output_dir_for_bus(base_out_dir: Path, bus_id: str) -> Path:
    bus_dir = base_out_dir / bus_id
    bus_dir.mkdir(parents=True, exist_ok=True)
    return bus_dir


def add_event_regions(ax: plt.Axes, t: np.ndarray, event_series: pd.Series) -> None:
    values = event_series.fillna(0).astype(int).to_numpy()
    spans = [span for span in contiguous_spans_from_values(t, values) if span[0] > 0]
    if not spans:
        return

    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min if y_max > y_min else 1.0

    for idx, (event_id, t0, t1) in enumerate(spans):
        color = EVENT_COLORS.get(event_id, "#9ca3af")
        ax.axvspan(t0, t1, color=color, alpha=0.12, lw=0)

        mid = 0.5 * (t0 + t1)
        y_text = y_max - (0.05 + 0.08 * (idx % 2)) * y_span
        ax.text(
            mid,
            y_text,
            EVENT_LABELS.get(event_id, f"Event {event_id}"),
            ha="center",
            va="top",
            fontsize=8,
            color="black",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": color, "alpha": 0.16, "edgecolor": "none"},
        )


def add_missing_data_regions(ax: plt.Axes, t: np.ndarray, data_present: pd.Series) -> None:
    present = data_present.fillna(1).astype(int).to_numpy()
    missing = (present == 0).astype(int)
    for flag, t0, t1 in contiguous_spans_from_values(t, missing):
        if flag == 1:
            ax.axvspan(t0, t1, color="black", alpha=0.06, lw=0)


def location_matches_bus(location: str, bus_id: str) -> bool:
    return extract_bus_number(bus_id) in re.findall(r"\d+", location)


def add_timeline_markers(ax: plt.Axes, timeline: list[dict[str, Any]], bus_id: str) -> None:
    if not timeline:
        return

    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min if y_max > y_min else 1.0

    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for record in timeline:
        grouped[float(record["approx_seconds"])].append(record)

    for idx, sec in enumerate(sorted(grouped)):
        records = grouped[sec]
        local = any(location_matches_bus(rec.get("location", ""), bus_id) for rec in records)

        line_color = "#991b1b" if local else "#111827"
        ax.axvline(sec, color=line_color, linestyle="--", linewidth=1.2 if local else 0.8, alpha=0.55 if local else 0.28)

        labels = []
        for rec in records:
            label = rec["short_label"]
            if local:
                label = f"LOCAL | {label}"
            labels.append(label)

        text = "\n".join(labels)
        y_text = y_max - (0.16 + 0.12 * (idx % 2)) * y_span
        ax.text(
            sec,
            y_text,
            text,
            rotation=90,
            ha="left",
            va="top",
            fontsize=7.5,
            color=line_color,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.65, "edgecolor": line_color, "linewidth": 0.6},
        )


def style_time_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.25)


def plot_single_trace(
    t: np.ndarray,
    y: pd.Series,
    title: str,
    ylabel: str,
    out_path: Path,
    event_series: pd.Series,
    data_present: pd.Series,
    timeline: list[dict[str, Any]],
    bus_id: str,
    color: str,
    figsize: tuple[float, float],
    dpi: int,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, y.to_numpy(), linewidth=0.9, color=color)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel)
    style_time_axis(ax)
    add_missing_data_regions(ax, t, data_present)
    add_event_regions(ax, t, event_series)
    add_timeline_markers(ax, timeline, bus_id)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_phase_trace(
    t: np.ndarray,
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    event_series: pd.Series,
    data_present: pd.Series,
    timeline: list[dict[str, Any]],
    bus_id: str,
    figsize: tuple[float, float],
    dpi: int,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    for phase in ("A", "B", "C"):
        ax.plot(t, df[f"V{phase}_ang"].to_numpy(), linewidth=0.8, label=f"V{phase} angle", color=PHASE_COLORS[phase])

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Voltage angle [deg]")
    style_time_axis(ax)
    ax.legend(loc="upper right", ncol=3, fontsize=9)
    add_missing_data_regions(ax, t, data_present)
    add_event_regions(ax, t, event_series)
    add_timeline_markers(ax, timeline, bus_id)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_frequency_rocof(
    t: np.ndarray,
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    event_series: pd.Series,
    data_present: pd.Series,
    timeline: list[dict[str, Any]],
    bus_id: str,
    figsize: tuple[float, float],
    dpi: int,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, constrained_layout=True)

    axes[0].plot(t, df["Frequency"].to_numpy(), linewidth=0.9, color="#7c3aed")
    axes[0].set_title(title, fontsize=14)
    axes[0].set_ylabel("Frequency [Hz]")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, df["ROCOF"].to_numpy(), linewidth=0.9, color="#ea580c")
    axes[1].set_ylabel("ROCOF [Hz/s]")
    axes[1].grid(True, alpha=0.25)
    axes[1].set_xlabel("Time [s]")

    for ax in axes:
        add_missing_data_regions(ax, t, data_present)
        add_event_regions(ax, t, event_series)
        add_timeline_markers(ax, timeline, bus_id)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def build_event_legend() -> list[Patch]:
    handles = [
        Patch(facecolor=EVENT_COLORS[event_id], edgecolor="none", alpha=0.20, label=f"{event_id}: {label}")
        for event_id, label in EVENT_LABELS.items()
        if event_id != 0
    ]
    handles.append(Patch(facecolor="black", edgecolor="none", alpha=0.06, label="DATA_PRESENT = 0"))
    return handles


def write_readme(
    bus_dir: Path,
    bus_id: str,
    descriptor: str,
    timeline: list[dict[str, Any]],
) -> None:
    lines = [descriptor, "", "Approximate timeline markers used:"]
    if timeline:
        for record in timeline:
            lines.append(f"- {record['short_label']}")
    else:
        lines.append("- No timeline xlsx found; only per-sample Event shading was used.")

    lines += [
        "",
        "Generated files:",
        "- va_vs_t.png",
        "- va_vs_t_hires.png",
        "- ia_vs_t.png",
        "- phase_vs_t.png",
        "- frequency_rocof_vs_t.png",
    ]

    (bus_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_bus(
    csv_path: Path,
    base_out_dir: Path,
    timeline: list[dict[str, Any]],
    pmu_meta: dict[str, dict[str, Any]],
    downsample: int,
    dpi: int,
    hires_dpi: int,
    show: bool = False,
) -> list[Path]:
    df = load_csv(csv_path, downsample=downsample)
    bus_id = extract_bus_id(csv_path.name)
    descriptor = build_bus_descriptor(bus_id, pmu_meta)

    bus_dir = build_output_dir_for_bus(base_out_dir, bus_id)
    t = df["TIMESTAMP"].to_numpy()

    outputs: list[Path] = []

    va_path = bus_dir / "va_vs_t.png"
    plot_single_trace(
        t=t,
        y=df["VA_mag"],
        title=f"{descriptor} | Va vs t",
        ylabel="Va magnitude [V]",
        out_path=va_path,
        event_series=df["Event"],
        data_present=df["DATA_PRESENT"],
        timeline=timeline,
        bus_id=bus_id,
        color=PHASE_COLORS["A"],
        figsize=(24, 8),
        dpi=dpi,
        show=show,
    )
    outputs.append(va_path)

    va_hires_path = bus_dir / "va_vs_t_hires.png"
    plot_single_trace(
        t=t,
        y=df["VA_mag"],
        title=f"{descriptor} | Va vs t | HIGH RESOLUTION",
        ylabel="Va magnitude [V]",
        out_path=va_hires_path,
        event_series=df["Event"],
        data_present=df["DATA_PRESENT"],
        timeline=timeline,
        bus_id=bus_id,
        color=PHASE_COLORS["A"],
        figsize=(42, 14),
        dpi=hires_dpi,
        show=False,
    )
    outputs.append(va_hires_path)

    ia_path = bus_dir / "ia_vs_t.png"
    plot_single_trace(
        t=t,
        y=df["IA_mag"],
        title=f"{descriptor} | Ia vs t",
        ylabel="Ia magnitude [A]",
        out_path=ia_path,
        event_series=df["Event"],
        data_present=df["DATA_PRESENT"],
        timeline=timeline,
        bus_id=bus_id,
        color="#0f766e",
        figsize=(24, 8),
        dpi=dpi,
        show=show,
    )
    outputs.append(ia_path)

    phase_path = bus_dir / "phase_vs_t.png"
    plot_phase_trace(
        t=t,
        df=df,
        title=f"{descriptor} | Voltage phase angles vs t",
        out_path=phase_path,
        event_series=df["Event"],
        data_present=df["DATA_PRESENT"],
        timeline=timeline,
        bus_id=bus_id,
        figsize=(24, 8),
        dpi=dpi,
        show=show,
    )
    outputs.append(phase_path)

    fr_path = bus_dir / "frequency_rocof_vs_t.png"
    plot_frequency_rocof(
        t=t,
        df=df,
        title=f"{descriptor} | Frequency + ROCOF vs t",
        out_path=fr_path,
        event_series=df["Event"],
        data_present=df["DATA_PRESENT"],
        timeline=timeline,
        bus_id=bus_id,
        figsize=(24, 10),
        dpi=dpi,
        show=show,
    )
    outputs.append(fr_path)

    write_readme(bus_dir, bus_id, descriptor, timeline)

    return outputs


def iter_target_files(raw_dir: Path, selected_bus: str | None) -> Iterable[Path]:
    if selected_bus:
        candidate = raw_dir / selected_bus
        if not candidate.exists():
            raise FileNotFoundError(f"Requested file not found: {candidate}")
        yield candidate
        return

    for filename in BUS_FILES:
        csv_path = raw_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected CSV not found: {csv_path}")
        yield csv_path


def main() -> None:
    args = parse_args()
    raw_dir, out_dir, timeline_xlsx, pmu_meta_txt = resolve_paths(args)
    selected_bus = normalize_bus_argument(args.bus)

    timeline = load_timeline_metadata(timeline_xlsx)
    pmu_meta = load_pmu_metadata(pmu_meta_txt)

    print(f"[plot_raw] raw_dir        = {raw_dir}")
    print(f"[plot_raw] out_dir        = {out_dir}")
    print(f"[plot_raw] timeline_xlsx  = {timeline_xlsx}")
    print(f"[plot_raw] pmu_meta_txt   = {pmu_meta_txt}")

    generated: list[Path] = []
    for csv_path in iter_target_files(raw_dir, selected_bus):
        print(f"[plot_raw] plotting {csv_path.name} ...")
        generated.extend(
            plot_bus(
                csv_path=csv_path,
                base_out_dir=out_dir,
                timeline=timeline,
                pmu_meta=pmu_meta,
                downsample=max(1, args.downsample),
                dpi=args.dpi,
                hires_dpi=args.hires_dpi,
                show=args.show,
            )
        )

    print(f"[plot_raw] done. generated {len(generated)} PNG files.")


if __name__ == "__main__":
    main()