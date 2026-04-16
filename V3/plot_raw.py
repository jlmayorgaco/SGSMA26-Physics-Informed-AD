from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import signal, stats


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

PHASE_COLORS = {"A": "#2563eb", "B": "#16a34a", "C": "#dc2626"}

ALL_MEASUREMENT_COLUMNS = [
    "VA_mag", "VA_ang", "VB_mag", "VB_ang", "VC_mag", "VC_ang",
    "IA_mag", "IA_ang", "IB_mag", "IB_ang", "IC_mag", "IC_ang",
    "Frequency", "ROCOF",
]

CHANNEL_GROUPS: dict[str, list[str]] = {
    "voltage_magnitude": ["VA_mag", "VB_mag", "VC_mag"],
    "voltage_angle": ["VA_ang", "VB_ang", "VC_ang"],
    "current_magnitude": ["IA_mag", "IB_mag", "IC_mag"],
    "current_angle": ["IA_ang", "IB_ang", "IC_ang"],
    "frequency_rocof": ["Frequency", "ROCOF"],
}

KEY_CHANNELS = ["VA_mag", "IA_mag", "VA_ang", "Frequency", "ROCOF"]


@dataclass
class EventSpan:
    event_id: int
    label: str
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    duration_s: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Advanced PMU signal analysis for SGSMA/IEEE-39 CSV files: overview plots, "
            "fault/event zooms, FFT/PSD, Hilbert/envelope, filtering, missing-data analysis, "
            "noise statistics, correlation analysis, JSON summaries, and a journal-style markdown report."
        )
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--timeline-xlsx", type=Path, default=None)
    parser.add_argument("--pmu-meta-txt", type=Path, default=None)
    parser.add_argument("--bus", type=str, default=None)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--hires-dpi", type=int, default=450)
    parser.add_argument("--event-pre-seconds", type=float, default=1.0)
    parser.add_argument("--event-post-seconds", type=float, default=1.0)
    parser.add_argument("--skip-combined", action="store_true")
    parser.add_argument("--show", action="store_true")
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


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


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
    out_dir = args.out_dir.resolve() if args.out_dir is not None else (raw_dir.parent / "results" / "sgsma_signal_analysis")

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


def contiguous_spans_from_values(t: np.ndarray, values: np.ndarray) -> list[tuple[int, float, float, int, int]]:
    if len(values) == 0:
        return []
    changes = np.flatnonzero(np.diff(values) != 0) + 1
    bounds = np.r_[0, changes, len(values)]
    spans: list[tuple[int, float, float, int, int]] = []
    for i0, i1 in zip(bounds[:-1], bounds[1:]):
        spans.append((int(values[i0]), float(t[i0]), float(t[i1 - 1]), int(i0), int(i1 - 1)))
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
    return df.rename(columns=applicable) if applicable else df


def validate_columns(df: pd.DataFrame, csv_path: Path) -> None:
    required = ["TIMESTAMP", *ALL_MEASUREMENT_COLUMNS, "DATA_PRESENT", "Event"]
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
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
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
    (bus_dir / "plots").mkdir(parents=True, exist_ok=True)
    (bus_dir / "stats").mkdir(parents=True, exist_ok=True)
    (bus_dir / "event_zoom").mkdir(parents=True, exist_ok=True)
    return bus_dir


def style_time_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Time [s]")
    ax.grid(True, alpha=0.25)


def add_event_regions(ax: plt.Axes, t: np.ndarray, event_series: pd.Series) -> None:
    values = event_series.fillna(0).astype(int).to_numpy()
    spans = [span for span in contiguous_spans_from_values(t, values) if span[0] > 0]
    if not spans:
        return

    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min if y_max > y_min else 1.0
    for idx, (event_id, t0, t1, _, _) in enumerate(spans):
        color = EVENT_COLORS.get(event_id, "#9ca3af")
        ax.axvspan(t0, t1, color=color, alpha=0.10, lw=0)
        mid = 0.5 * (t0 + t1)
        y_text = y_max - (0.05 + 0.08 * (idx % 2)) * y_span
        ax.text(
            mid,
            y_text,
            EVENT_LABELS.get(event_id, f"Event {event_id}"),
            ha="center",
            va="top",
            fontsize=7.5,
            color="black",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": color, "alpha": 0.14, "edgecolor": "none"},
        )


def add_missing_data_regions(ax: plt.Axes, t: np.ndarray, data_present: pd.Series) -> None:
    present = data_present.fillna(1).astype(int).to_numpy()
    missing = (present == 0).astype(int)
    for flag, t0, t1, _, _ in contiguous_spans_from_values(t, missing):
        if flag == 1:
            ax.axvspan(t0, t1, color="black", alpha=0.05, lw=0)


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
        ax.axvline(sec, color=line_color, linestyle="--", linewidth=1.0 if local else 0.7, alpha=0.45 if local else 0.22)
        text = "\n".join((f"LOCAL | {rec['short_label']}" if local else rec["short_label"]) for rec in records)
        y_text = y_max - (0.16 + 0.12 * (idx % 2)) * y_span
        ax.text(
            sec,
            y_text,
            text,
            rotation=90,
            ha="left",
            va="top",
            fontsize=6.5,
            color=line_color,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.60, "edgecolor": line_color, "linewidth": 0.5},
        )


def make_safe_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


def estimate_sampling_rate(t: np.ndarray) -> float:
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def interpolate_series(series: pd.Series) -> pd.Series:
    return series.astype(float).interpolate(method="linear", limit_direction="both")


def unwrap_angle_degrees(x: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(x)))


def smooth_signal(x: np.ndarray, fs: float) -> np.ndarray:
    n = len(x)
    if n < 7:
        return x.copy()
    target = int(max(7, round(fs * 2.0)))
    window = min(n - (1 - n % 2), target)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return x.copy()
    polyorder = 3 if window >= 7 else 2
    return signal.savgol_filter(x, window_length=window, polyorder=polyorder, mode="interp")


def band_energy(freqs: np.ndarray, psd: np.ndarray, f_lo: float, f_hi: float) -> float:
    mask = (freqs >= f_lo) & (freqs < f_hi)
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
            return None
        if isinstance(value, (int, np.integer, float, np.floating)):
            return float(value)
        return float(value)
    except Exception:
        return None


def series_basic_stats(series: pd.Series, fs: float) -> dict[str, Any]:
    s = series.astype(float)
    valid = s.dropna()
    n_total = int(len(s))
    n_valid = int(valid.size)
    n_missing = int(n_total - n_valid)
    out: dict[str, Any] = {
        "n_total": n_total,
        "n_valid": n_valid,
        "n_missing": n_missing,
        "missing_ratio": float(n_missing / n_total) if n_total else None,
    }
    if n_valid == 0:
        return out

    x = valid.to_numpy(dtype=float)
    out.update({
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if n_valid > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "peak_to_peak": float(np.ptp(x)),
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "energy": float(np.sum(np.square(x)) / fs),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "skew": float(stats.skew(x, bias=False)) if n_valid > 2 else None,
        "kurtosis_excess": float(stats.kurtosis(x, fisher=True, bias=False)) if n_valid > 3 else None,
    })

    if n_valid > 2:
        dx = np.diff(x) * fs
        out.update({
            "derivative_mean": float(np.mean(dx)),
            "derivative_std": float(np.std(dx, ddof=1)) if len(dx) > 1 else 0.0,
            "derivative_abs_max": float(np.max(np.abs(dx))),
        })
    return out


def noise_analysis(series: pd.Series, fs: float, is_angle: bool) -> dict[str, Any]:
    s = interpolate_series(series)
    x = s.to_numpy(dtype=float)
    if is_angle:
        x = unwrap_angle_degrees(x)
    trend = smooth_signal(x, fs)
    residual = x - trend
    residual_std = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    trend_std = float(np.std(trend, ddof=1)) if len(trend) > 1 else 0.0
    snr_db = None
    if residual_std > 0 and trend_std > 0:
        snr_db = float(20.0 * np.log10(trend_std / residual_std))

    normaltest_p = None
    if len(residual) >= 8:
        try:
            normaltest_p = float(stats.normaltest(residual).pvalue)
        except Exception:
            normaltest_p = None

    return {
        "trend_std": trend_std,
        "residual_std": residual_std,
        "residual_mad": float(stats.median_abs_deviation(residual, scale="normal")),
        "snr_db": snr_db,
        "residual_skew": float(stats.skew(residual, bias=False)) if len(residual) > 2 else None,
        "residual_kurtosis_excess": float(stats.kurtosis(residual, fisher=True, bias=False)) if len(residual) > 3 else None,
        "normaltest_pvalue": normaltest_p,
    }


def spectral_analysis(series: pd.Series, fs: float, is_angle: bool) -> dict[str, Any]:
    s = interpolate_series(series)
    x = s.to_numpy(dtype=float)
    if is_angle:
        x = unwrap_angle_degrees(x)
    x = signal.detrend(x)
    if len(x) < 8:
        return {
            "dominant_frequencies_hz": [],
            "dominant_psd": [],
            "band_energy": {},
            "harmonic_like_components": [],
        }

    nperseg = min(4096, len(x))
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, scaling="density")
    if len(freqs) <= 1:
        return {
            "dominant_frequencies_hz": [],
            "dominant_psd": [],
            "band_energy": {},
            "harmonic_like_components": [],
        }

    nonzero = np.where(freqs > 1e-9)[0]
    peak_indices = nonzero[np.argsort(psd[nonzero])[-5:]][::-1] if len(nonzero) else np.array([], dtype=int)
    dominant_freqs = [float(freqs[i]) for i in peak_indices]
    dominant_psd = [float(psd[i]) for i in peak_indices]

    bands = {
        "0_to_0p1_hz": band_energy(freqs, psd, 0.0, 0.1),
        "0p1_to_1_hz": band_energy(freqs, psd, 0.1, 1.0),
        "1_to_5_hz": band_energy(freqs, psd, 1.0, 5.0),
        "5_to_nyquist_hz": band_energy(freqs, psd, 5.0, fs / 2.0),
    }

    harmonic_like_components: list[dict[str, Any]] = []
    if dominant_freqs:
        f0 = dominant_freqs[0]
        if f0 > 0.02:
            for k in range(1, 6):
                fk = k * f0
                if fk >= fs / 2.0:
                    break
                idx = int(np.argmin(np.abs(freqs - fk)))
                harmonic_like_components.append({
                    "multiple": k,
                    "target_frequency_hz": float(fk),
                    "closest_bin_frequency_hz": float(freqs[idx]),
                    "psd": float(psd[idx]),
                })

    return {
        "dominant_frequencies_hz": dominant_freqs,
        "dominant_psd": dominant_psd,
        "band_energy": bands,
        "harmonic_like_components": harmonic_like_components,
    }


def hilbert_analysis(series: pd.Series, fs: float, is_angle: bool) -> dict[str, Any]:
    s = interpolate_series(series)
    x = s.to_numpy(dtype=float)
    if is_angle:
        x = unwrap_angle_degrees(x)
    x = signal.detrend(x)
    if len(x) < 8:
        return {}

    analytic = signal.hilbert(x)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * fs / (2 * np.pi)
    inst_freq = inst_freq[np.isfinite(inst_freq)]

    return {
        "envelope_mean": float(np.mean(envelope)),
        "envelope_std": float(np.std(envelope, ddof=1)) if len(envelope) > 1 else 0.0,
        "envelope_max": float(np.max(envelope)),
        "instantaneous_frequency_mean_hz": float(np.mean(inst_freq)) if len(inst_freq) else None,
        "instantaneous_frequency_std_hz": float(np.std(inst_freq, ddof=1)) if len(inst_freq) > 1 else None,
        "instantaneous_frequency_absmax_hz": float(np.max(np.abs(inst_freq))) if len(inst_freq) else None,
    }


def get_event_spans(df: pd.DataFrame) -> list[EventSpan]:
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    values = df["Event"].fillna(0).astype(int).to_numpy()
    spans: list[EventSpan] = []
    for event_id, t0, t1, i0, i1 in contiguous_spans_from_values(t, values):
        if event_id <= 0:
            continue
        spans.append(EventSpan(
            event_id=event_id,
            label=EVENT_LABELS.get(event_id, f"Event {event_id}"),
            start_idx=i0,
            end_idx=i1,
            start_time=t0,
            end_time=t1,
            duration_s=float(t1 - t0),
        ))
    return spans


def missing_data_analysis(df: pd.DataFrame) -> dict[str, Any]:
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    present = df["DATA_PRESENT"].fillna(1).astype(int).to_numpy()
    missing = (present == 0).astype(int)
    spans = [span for span in contiguous_spans_from_values(t, missing) if span[0] == 1]
    durations = [float(t1 - t0) for _, t0, t1, _, _ in spans]
    return {
        "missing_frame_count": int(np.sum(missing)),
        "missing_ratio": float(np.mean(missing)) if len(missing) else 0.0,
        "gap_count": int(len(spans)),
        "max_gap_seconds": float(max(durations)) if durations else 0.0,
        "median_gap_seconds": float(np.median(durations)) if durations else 0.0,
        "gaps": [
            {"start_s": float(t0), "end_s": float(t1), "duration_s": float(t1 - t0)}
            for _, t0, t1, _, _ in spans
        ],
    }


def dataset_event_summary(df: pd.DataFrame) -> dict[str, Any]:
    spans = get_event_spans(df)
    by_id: dict[int, dict[str, Any]] = defaultdict(lambda: {"count": 0, "durations_s": []})
    for span in spans:
        entry = by_id[span.event_id]
        entry["count"] += 1
        entry["durations_s"].append(span.duration_s)
    out: dict[str, Any] = {}
    for event_id, entry in sorted(by_id.items()):
        durations = entry["durations_s"]
        out[str(event_id)] = {
            "label": EVENT_LABELS.get(event_id, f"Event {event_id}"),
            "count": int(entry["count"]),
            "durations_s": [float(d) for d in durations],
            "mean_duration_s": float(np.mean(durations)) if durations else 0.0,
        }
    return out


def compute_channel_analysis(df: pd.DataFrame, fs: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for col in ALL_MEASUREMENT_COLUMNS:
        is_angle = col.endswith("_ang")
        out[col] = {
            "basic": series_basic_stats(df[col], fs),
            "noise": noise_analysis(df[col], fs, is_angle=is_angle),
            "spectral": spectral_analysis(df[col], fs, is_angle=is_angle),
            "hilbert": hilbert_analysis(df[col], fs, is_angle=is_angle),
        }
    return out


def save_json(path: Path, payload: Any) -> None:
    def normalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): normalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [normalize(v) for v in obj]
        if isinstance(obj, tuple):
            return [normalize(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            value = float(obj)
            return None if not np.isfinite(value) else value
        if isinstance(obj, float):
            return None if not math.isfinite(obj) else obj
        return obj

    path.write_text(json.dumps(normalize(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def savefig(fig: plt.Figure, path: Path, dpi: int, show: bool) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_overview_magnitude_frequency(
    df: pd.DataFrame,
    bus_id: str,
    descriptor: str,
    timeline: list[dict[str, Any]],
    out_path: Path,
    dpi: int,
    show: bool,
) -> None:
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    fig, axes = plt.subplots(4, 1, figsize=(28, 16), sharex=True, constrained_layout=True)

    for phase in ("A", "B", "C"):
        axes[0].plot(t, df[f"V{phase}_mag"].to_numpy(), label=f"V{phase}", linewidth=0.8, color=PHASE_COLORS[phase])
        axes[1].plot(t, df[f"I{phase}_mag"].to_numpy(), label=f"I{phase}", linewidth=0.8, color=PHASE_COLORS[phase])

    axes[0].set_title(f"{descriptor} | Voltage magnitudes", fontsize=14)
    axes[1].set_title(f"{descriptor} | Current magnitudes", fontsize=14)
    axes[2].plot(t, df["Frequency"].to_numpy(), linewidth=0.9, color="#7c3aed")
    axes[2].set_title(f"{descriptor} | Frequency", fontsize=14)
    axes[3].plot(t, df["ROCOF"].to_numpy(), linewidth=0.9, color="#ea580c")
    axes[3].set_title(f"{descriptor} | ROCOF", fontsize=14)

    axes[0].set_ylabel("Voltage")
    axes[1].set_ylabel("Current")
    axes[2].set_ylabel("Hz")
    axes[3].set_ylabel("Hz/s")
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    axes[1].legend(loc="upper right", ncol=3, fontsize=9)

    for ax in axes:
        ax.grid(True, alpha=0.25)
        add_missing_data_regions(ax, t, df["DATA_PRESENT"])
        add_event_regions(ax, t, df["Event"])
        add_timeline_markers(ax, timeline, bus_id)
    axes[-1].set_xlabel("Time [s]")
    savefig(fig, out_path, dpi=dpi, show=show)


def plot_overview_angles(
    df: pd.DataFrame,
    bus_id: str,
    descriptor: str,
    timeline: list[dict[str, Any]],
    out_path: Path,
    dpi: int,
    show: bool,
) -> None:
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(28, 10), sharex=True, constrained_layout=True)
    for phase in ("A", "B", "C"):
        axes[0].plot(t, df[f"V{phase}_ang"].to_numpy(), label=f"V{phase}", linewidth=0.8, color=PHASE_COLORS[phase])
        axes[1].plot(t, df[f"I{phase}_ang"].to_numpy(), label=f"I{phase}", linewidth=0.8, color=PHASE_COLORS[phase])
    axes[0].set_title(f"{descriptor} | Voltage angles", fontsize=14)
    axes[1].set_title(f"{descriptor} | Current angles", fontsize=14)
    axes[0].set_ylabel("deg")
    axes[1].set_ylabel("deg")
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    axes[1].legend(loc="upper right", ncol=3, fontsize=9)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        add_missing_data_regions(ax, t, df["DATA_PRESENT"])
        add_event_regions(ax, t, df["Event"])
        add_timeline_markers(ax, timeline, bus_id)
    axes[-1].set_xlabel("Time [s]")
    savefig(fig, out_path, dpi=dpi, show=show)


def plot_histograms(df: pd.DataFrame, group_name: str, columns: list[str], out_path: Path, dpi: int, show: bool) -> None:
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, columns):
        s = df[col].dropna().astype(float)
        if s.empty:
            ax.text(0.5, 0.5, f"{col}: no valid samples", ha="center", va="center")
            ax.axis("off")
            continue
        ax.hist(s.to_numpy(), bins=80, alpha=0.8)
        ax.set_title(f"{group_name} | {col} histogram")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.25)
    savefig(fig, out_path, dpi=dpi, show=show)


def plot_spectra(df: pd.DataFrame, group_name: str, columns: list[str], fs: float, out_path: Path, dpi: int, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    for col in columns:
        s = interpolate_series(df[col]).astype(float)
        x = s.to_numpy()
        if col.endswith("_ang"):
            x = unwrap_angle_degrees(x)
        x = signal.detrend(x)
        if len(x) < 8:
            continue
        freqs, psd = signal.welch(x, fs=fs, nperseg=min(4096, len(x)), scaling="density")
        ax.semilogy(freqs, psd, linewidth=1.0, label=col)
    ax.set_title(f"{group_name} | Welch PSD")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    savefig(fig, out_path, dpi=dpi, show=show)


def plot_noise_diagnostic(df: pd.DataFrame, col: str, fs: float, out_path: Path, dpi: int, show: bool) -> None:
    s = interpolate_series(df[col]).astype(float)
    x = s.to_numpy()
    if col.endswith("_ang"):
        x = unwrap_angle_degrees(x)
    trend = smooth_signal(x, fs)
    residual = x - trend
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    axes[0].plot(t, x, linewidth=0.9, label="signal")
    axes[0].plot(t, trend, linewidth=1.0, label="trend")
    axes[0].set_title(f"{col} | signal vs trend")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t, residual, linewidth=0.8)
    axes[1].set_title(f"{col} | residual / estimated noise")
    axes[1].grid(True, alpha=0.25)

    axes[2].hist(residual, bins=80, alpha=0.8, density=True)
    if len(residual) > 1:
        mu = float(np.mean(residual))
        sigma = float(np.std(residual, ddof=1))
        if sigma > 0:
            xgrid = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
            axes[2].plot(xgrid, stats.norm.pdf(xgrid, loc=mu, scale=sigma), linewidth=1.2)
    axes[2].set_title(f"{col} | residual histogram with Gaussian fit")
    axes[2].grid(True, alpha=0.25)
    savefig(fig, out_path, dpi=dpi, show=show)


def plot_hilbert_diagnostic(df: pd.DataFrame, col: str, fs: float, out_path: Path, dpi: int, show: bool) -> None:
    s = interpolate_series(df[col]).astype(float)
    x = s.to_numpy()
    if col.endswith("_ang"):
        x = unwrap_angle_degrees(x)
    x = signal.detrend(x)
    if len(x) < 8:
        return
    analytic = signal.hilbert(x)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * fs / (2 * np.pi)
    t = df["TIMESTAMP"].to_numpy(dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    axes[0].plot(t, x, linewidth=0.9, label="detrended")
    axes[0].plot(t, envelope, linewidth=1.0, label="envelope")
    axes[0].set_title(f"{col} | Hilbert envelope")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(t[:-1], inst_freq, linewidth=0.8)
    axes[1].set_title(f"{col} | instantaneous frequency from analytic phase")
    axes[1].grid(True, alpha=0.25)

    axes[2].hist(envelope, bins=80, alpha=0.8)
    axes[2].set_title(f"{col} | envelope histogram")
    axes[2].grid(True, alpha=0.25)
    savefig(fig, out_path, dpi=dpi, show=show)


def butter_filter(x: np.ndarray, fs: float, kind: str, cutoff: float | tuple[float, float], order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    if kind == "band":
        low, high = cutoff
        wn = [max(low / nyq, 1e-5), min(high / nyq, 0.999)]
        b, a = signal.butter(order, wn, btype="bandpass")
    elif kind == "low":
        wn = min(float(cutoff) / nyq, 0.999)
        b, a = signal.butter(order, wn, btype="lowpass")
    elif kind == "high":
        wn = max(float(cutoff) / nyq, 1e-5)
        b, a = signal.butter(order, wn, btype="highpass")
    else:
        raise ValueError(kind)
    return signal.filtfilt(b, a, x)


def plot_filter_bank(df: pd.DataFrame, col: str, fs: float, out_path: Path, dpi: int, show: bool) -> None:
    s = interpolate_series(df[col]).astype(float)
    x = s.to_numpy()
    if col.endswith("_ang"):
        x = unwrap_angle_degrees(x)
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    low = butter_filter(x, fs, "low", min(1.0, fs / 3.0))
    high = butter_filter(x, fs, "high", min(0.5, fs / 6.0))
    band_hi = min(5.0, fs / 2.0 - 1e-3)
    band_lo = min(0.1, band_hi / 2.0)
    if band_hi <= band_lo:
        band_hi = min(fs / 2.0 - 1e-3, band_lo + 0.1)
    band = butter_filter(x, fs, "band", (band_lo, band_hi))

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, constrained_layout=True)
    axes[0].plot(t, x, linewidth=0.8)
    axes[0].set_title(f"{col} | raw/interpolated")
    axes[1].plot(t, low, linewidth=0.8)
    axes[1].set_title(f"{col} | low-pass")
    axes[2].plot(t, high, linewidth=0.8)
    axes[2].set_title(f"{col} | high-pass")
    axes[3].plot(t, band, linewidth=0.8)
    axes[3].set_title(f"{col} | band-pass ({band_lo:.2f}–{band_hi:.2f} Hz)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time [s]")
    savefig(fig, out_path, dpi=dpi, show=show)


def plot_event_zoom_panels(
    df: pd.DataFrame,
    bus_id: str,
    descriptor: str,
    span: EventSpan,
    pre_s: float,
    post_s: float,
    out_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    start = span.start_time - pre_s
    end = span.end_time + post_s
    mask = (t >= start) & (t <= end)
    if not np.any(mask):
        return
    win = df.loc[mask].reset_index(drop=True)
    tw = win["TIMESTAMP"].to_numpy(dtype=float)
    event_name = make_safe_filename(f"event_{span.event_id}_{span.label}_{span.start_time:.3f}s")

    fig, axes = plt.subplots(4, 1, figsize=(22, 16), sharex=True, constrained_layout=True)
    for phase in ("A", "B", "C"):
        axes[0].plot(tw, win[f"V{phase}_mag"].to_numpy(), linewidth=1.0, label=f"V{phase}", color=PHASE_COLORS[phase])
        axes[1].plot(tw, win[f"I{phase}_mag"].to_numpy(), linewidth=1.0, label=f"I{phase}", color=PHASE_COLORS[phase])
        axes[2].plot(tw, win[f"V{phase}_ang"].to_numpy(), linewidth=1.0, label=f"V{phase} ang", color=PHASE_COLORS[phase])
    axes[3].plot(tw, win["Frequency"].to_numpy(), linewidth=1.0, label="Frequency", color="#7c3aed")
    axes[3].plot(tw, win["ROCOF"].to_numpy(), linewidth=1.0, label="ROCOF", color="#ea580c")

    axes[0].set_title(f"{descriptor} | {span.label} | event-centric zoom")
    axes[0].set_ylabel("Voltage")
    axes[1].set_ylabel("Current")
    axes[2].set_ylabel("Angle [deg]")
    axes[3].set_ylabel("Hz / Hz/s")

    for ax in axes:
        ax.axvspan(span.start_time, span.end_time, color=EVENT_COLORS.get(span.event_id, "#999999"), alpha=0.18)
        ax.axvline(span.start_time, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(span.end_time, color="black", linestyle="--", linewidth=1.0)
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)
    axes[2].legend(loc="upper right", ncol=3, fontsize=8)
    axes[3].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    savefig(fig, out_dir / f"{event_name}_zoom.png", dpi=dpi, show=show)


def build_event_legend() -> list[Patch]:
    handles = [
        Patch(facecolor=EVENT_COLORS[event_id], edgecolor="none", alpha=0.20, label=f"{event_id}: {label}")
        for event_id, label in EVENT_LABELS.items() if event_id != 0
    ]
    handles.append(Patch(facecolor="black", edgecolor="none", alpha=0.06, label="DATA_PRESENT = 0"))
    return handles


def write_bus_readme(bus_dir: Path, bus_id: str, descriptor: str, fs: float, spans: list[EventSpan]) -> None:
    lines = [
        f"{descriptor}",
        "",
        f"Estimated sampling rate: {fs:.4f} fps",
        "",
        "Event spans detected from the per-sample Event column:",
    ]
    if spans:
        for span in spans:
            lines.append(
                f"- Event {span.event_id} ({span.label}): start={span.start_time:.3f}s, "
                f"end={span.end_time:.3f}s, duration={span.duration_s:.3f}s"
            )
    else:
        lines.append("- No abnormal spans detected.")
    lines.extend([
        "",
        "Generated content:",
        "- plots/*.png: overview, histograms, spectra, filters, Hilbert, noise diagnostics",
        "- event_zoom/*.png: high-resolution event-centric transient zooms",
        "- stats/channel_analysis.json: per-signal statistics, spectral content, Hilbert metrics, noise analysis",
        "- stats/summary.json: bus-level summary",
    ])
    (bus_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_bus(
    csv_path: Path,
    base_out_dir: Path,
    timeline: list[dict[str, Any]],
    pmu_meta: dict[str, dict[str, Any]],
    downsample: int,
    dpi: int,
    hires_dpi: int,
    pre_s: float,
    post_s: float,
    show: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = load_csv(csv_path, downsample=downsample)
    bus_id = extract_bus_id(csv_path.name)
    descriptor = build_bus_descriptor(bus_id, pmu_meta)
    bus_dir = build_output_dir_for_bus(base_out_dir, bus_id)
    plots_dir = bus_dir / "plots"
    stats_dir = bus_dir / "stats"
    fs = estimate_sampling_rate(df["TIMESTAMP"].to_numpy(dtype=float))
    spans = get_event_spans(df)

    plot_overview_magnitude_frequency(df, bus_id, descriptor, timeline, plots_dir / "overview_magnitude_frequency.png", dpi, show)
    plot_overview_angles(df, bus_id, descriptor, timeline, plots_dir / "overview_angles.png", dpi, show)

    for group_name, columns in CHANNEL_GROUPS.items():
        plot_histograms(df, group_name, columns, plots_dir / f"hist_{group_name}.png", dpi, show)
        plot_spectra(df, group_name, columns, fs, plots_dir / f"spectrum_{group_name}.png", dpi, show)

    for col in KEY_CHANNELS:
        plot_noise_diagnostic(df, col, fs, plots_dir / f"noise_{col}.png", dpi, show)
        plot_hilbert_diagnostic(df, col, fs, plots_dir / f"hilbert_{col}.png", dpi, show)
        plot_filter_bank(df, col, fs, plots_dir / f"filters_{col}.png", dpi, show)

    for span in spans:
        plot_event_zoom_panels(df, bus_id, descriptor, span, pre_s, post_s, bus_dir / "event_zoom", hires_dpi, show)

    channel_analysis = compute_channel_analysis(df, fs)
    missing_summary = missing_data_analysis(df)
    event_summary = dataset_event_summary(df)

    summary = {
        "bus_id": bus_id,
        "descriptor": descriptor,
        "csv_path": str(csv_path),
        "sampling_rate_hz": fs,
        "row_count": int(len(df)),
        "event_spans": [span.__dict__ for span in spans],
        "event_summary": event_summary,
        "missing_data": missing_summary,
        "notes": {
            "harmonics_note": (
                "The CSVs contain PMU phasor features sampled at ~30 fps, not the raw 60 Hz waveform. "
                "Therefore, the spectral and 'harmonic-like' analysis here refers to low-frequency dynamics, "
                "oscillations, envelopes, and inter-area/electromechanical content observable in the PMU features."
            )
        },
    }

    save_json(stats_dir / "summary.json", summary)
    save_json(stats_dir / "channel_analysis.json", channel_analysis)
    write_bus_readme(bus_dir, bus_id, descriptor, fs, spans)
    return df, {"summary": summary, "channel_analysis": channel_analysis}


def merge_buses(bus_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for bus_id, df in bus_frames.items():
        suffix_map = {col: f"{bus_id}_{col}" for col in ALL_MEASUREMENT_COLUMNS + ["DATA_PRESENT", "Event"]}
        bus_df = df[["TIMESTAMP", *suffix_map.keys()]].rename(columns=suffix_map)
        merged = bus_df if merged is None else merged.merge(bus_df, on="TIMESTAMP", how="inner")
    if merged is None:
        raise RuntimeError("No bus frames to merge.")
    return merged.sort_values("TIMESTAMP").reset_index(drop=True)


def plot_missing_matrix(merged: pd.DataFrame, bus_ids: list[str], out_path: Path, dpi: int, show: bool) -> None:
    matrix = np.vstack([merged[f"{bus_id}_DATA_PRESENT"].fillna(0).astype(float).to_numpy() for bus_id in bus_ids])
    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(bus_ids)))
    ax.set_yticklabels(bus_ids)
    ax.set_xlabel("Time index")
    ax.set_title("DATA_PRESENT matrix across PMU buses (1=present, 0=missing)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    savefig(fig, out_path, dpi=dpi, show=show)


def correlation_heatmap(df: pd.DataFrame, columns: list[str], title: str, out_path: Path, dpi: int, show: bool) -> None:
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.to_numpy(), interpolation="nearest")
    ax.set_xticks(np.arange(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(columns)))
    ax.set_yticklabels(columns)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    savefig(fig, out_path, dpi=dpi, show=show)


def get_reference_event_spans(bus_frames: dict[str, pd.DataFrame]) -> list[EventSpan]:
    if "Bus39" in bus_frames:
        return get_event_spans(bus_frames["Bus39"])
    first_bus = sorted(bus_frames)[0]
    return get_event_spans(bus_frames[first_bus])


def plot_combined_event_overlay(
    merged: pd.DataFrame,
    bus_ids: list[str],
    ref_spans: list[EventSpan],
    value_suffix: str,
    pre_s: float,
    post_s: float,
    out_dir: Path,
    dpi: int,
    show: bool,
) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    t = merged["TIMESTAMP"].to_numpy(dtype=float)
    for span in ref_spans:
        mask = (t >= span.start_time - pre_s) & (t <= span.end_time + post_s)
        if not np.any(mask):
            continue
        win = merged.loc[mask].reset_index(drop=True)
        tw = win["TIMESTAMP"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(16, 6))
        event_rank: list[tuple[str, float]] = []
        for bus_id in bus_ids:
            col = f"{bus_id}_{value_suffix}"
            series = interpolate_series(win[col]).astype(float)
            baseline_mask = tw < span.start_time
            baseline = series.loc[baseline_mask].to_numpy() if np.any(baseline_mask) else series.to_numpy()
            mu = float(np.mean(baseline)) if len(baseline) else 0.0
            sigma = float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 1.0
            sigma = sigma if sigma > 1e-12 else 1.0
            z = np.abs((series.to_numpy() - mu) / sigma)
            event_rank.append((bus_id, float(np.max(z))))
            ax.plot(tw, series.to_numpy(), linewidth=1.0, label=bus_id)

        ax.axvspan(span.start_time, span.end_time, color=EVENT_COLORS.get(span.event_id, "#999999"), alpha=0.18)
        ax.axvline(span.start_time, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(span.end_time, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(f"{value_suffix} | multi-bus overlay | event {span.event_id} ({span.label})")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(value_suffix)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", ncol=4, fontsize=8)
        filename = make_safe_filename(f"overlay_{value_suffix}_event_{span.event_id}_{span.label}_{span.start_time:.3f}s.png")
        savefig(fig, out_dir / filename, dpi=dpi, show=show)

        event_rank.sort(key=lambda item: item[1], reverse=True)
        rankings.append({
            "event_id": span.event_id,
            "label": span.label,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "value_suffix": value_suffix,
            "ranked_buses_by_peak_abs_zscore": [{"bus": bus, "score": score} for bus, score in event_rank],
        })
    return rankings


def combined_analysis(
    base_out_dir: Path,
    bus_frames: dict[str, pd.DataFrame],
    per_bus_results: dict[str, dict[str, Any]],
    pre_s: float,
    post_s: float,
    dpi: int,
    show: bool,
) -> dict[str, Any]:
    combined_dir = base_out_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    bus_ids = sorted(bus_frames)
    merged = merge_buses(bus_frames)
    fs = estimate_sampling_rate(merged["TIMESTAMP"].to_numpy(dtype=float))

    plot_missing_matrix(merged, bus_ids, combined_dir / "missing_matrix.png", dpi, show)
    correlation_heatmap(
        merged,
        [f"{bus}_Frequency" for bus in bus_ids],
        "Cross-bus correlation | Frequency",
        combined_dir / "corr_frequency.png",
        dpi,
        show,
    )
    correlation_heatmap(
        merged,
        [f"{bus}_ROCOF" for bus in bus_ids],
        "Cross-bus correlation | ROCOF",
        combined_dir / "corr_rocof.png",
        dpi,
        show,
    )

    ref_spans = get_reference_event_spans(bus_frames)
    rankings_freq = plot_combined_event_overlay(merged, bus_ids, ref_spans, "Frequency", pre_s, post_s, combined_dir, dpi, show)
    rankings_rocof = plot_combined_event_overlay(merged, bus_ids, ref_spans, "ROCOF", pre_s, post_s, combined_dir, dpi, show)
    rankings_va = plot_combined_event_overlay(merged, bus_ids, ref_spans, "VA_mag", pre_s, post_s, combined_dir, dpi, show)

    summary = {
        "sampling_rate_hz": fs,
        "row_count": int(len(merged)),
        "bus_ids": bus_ids,
        "rankings_frequency": rankings_freq,
        "rankings_rocof": rankings_rocof,
        "rankings_va_mag": rankings_va,
        "per_bus_missing_ratio": {
            bus: per_bus_results[bus]["summary"]["missing_data"]["missing_ratio"] for bus in bus_ids
        },
    }
    save_json(combined_dir / "summary.json", summary)
    return summary


def generate_markdown_report(
    out_dir: Path,
    raw_dir: Path,
    pmu_meta_txt: Path | None,
    timeline_xlsx: Path | None,
    per_bus_results: dict[str, dict[str, Any]],
    combined_summary: dict[str, Any] | None,
) -> Path:
    report_path = out_dir / "dataset_signal_analysis_report.md"
    bus_ids = sorted(per_bus_results)

    lines: list[str] = []
    lines.append("# Advanced PMU Signal Analysis Report")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("")
    lines.append("This report was generated automatically from the raw SGSMA/IEEE-39 PMU CSV files. It extends the original plotting-only workflow with advanced exploratory signal analysis, event-centric transient inspection, frequency-domain analysis, Hilbert-envelope diagnostics, filtering views, missing-data quantification, and cross-bus correlation/overlay analysis.")
    lines.append("")
    lines.append("## 2. Inputs")
    lines.append("")
    lines.append(f"- Raw directory: `{raw_dir}`")
    lines.append(f"- PMU metadata file: `{pmu_meta_txt}`")
    lines.append(f"- Timeline file: `{timeline_xlsx}`")
    lines.append(f"- Buses analyzed: {', '.join(bus_ids)}")
    lines.append("")
    lines.append("## 3. Important methodological note")
    lines.append("")
    lines.append("The available signals are PMU phasor features sampled at roughly 30 fps, not high-frequency instantaneous voltage/current waveforms. Therefore, the FFT, spectral peaks, Hilbert transform, and 'harmonic-like' outputs in this report characterize low-frequency PMU dynamics, envelopes, oscillatory modes, transient signatures, and inter-area/electromechanical behaviour visible in the phasor-domain time series. They do **not** represent classical waveform harmonic estimation of the 60 Hz instantaneous signal.")
    lines.append("")
    lines.append("## 4. Output structure")
    lines.append("")
    lines.append("- `Bus*/plots/`: overviews, histograms, spectra, noise diagnostics, Hilbert plots, filter-bank plots")
    lines.append("- `Bus*/event_zoom/`: high-resolution event-centered transient zooms")
    lines.append("- `Bus*/stats/`: JSON summaries for all channels")
    lines.append("- `combined/`: cross-bus overlays, correlations, missing-data matrix, combined summary")
    lines.append("")

    if combined_summary is not None:
        lines.append("## 5. Combined multi-PMU findings")
        lines.append("")
        missing_ratios = combined_summary.get("per_bus_missing_ratio", {})
        if missing_ratios:
            worst_bus = max(missing_ratios, key=lambda k: missing_ratios[k] or 0.0)
            lines.append(f"- Highest missing-data ratio: **{worst_bus}** with ratio `{missing_ratios[worst_bus]:.6f}`.")
        lines.append("- Cross-bus overlay rankings for `Frequency`, `ROCOF`, and `VA_mag` were generated to help identify which PMU exhibits the strongest normalized excursion during each event.")
        lines.append("- See `combined/corr_frequency.png`, `combined/corr_rocof.png`, and `combined/missing_matrix.png` for the principal joint diagnostics.")
        lines.append("")

    lines.append("## 6. Per-bus summary")
    lines.append("")
    for bus_id in bus_ids:
        summary = per_bus_results[bus_id]["summary"]
        channel_analysis = per_bus_results[bus_id]["channel_analysis"]
        missing = summary["missing_data"]
        freq_dom = channel_analysis["Frequency"]["spectral"].get("dominant_frequencies_hz", [])
        rocof_dom = channel_analysis["ROCOF"]["spectral"].get("dominant_frequencies_hz", [])
        freq_noise = channel_analysis["Frequency"]["noise"]
        rocof_noise = channel_analysis["ROCOF"]["noise"]
        va_stats = channel_analysis["VA_mag"]["basic"]
        ia_stats = channel_analysis["IA_mag"]["basic"]

        lines.append(f"### {bus_id}")
        lines.append("")
        lines.append(f"- Descriptor: `{summary['descriptor']}`")
        lines.append(f"- Sampling rate: `{summary['sampling_rate_hz']:.4f}` fps")
        lines.append(f"- Rows: `{summary['row_count']}`")
        lines.append(f"- Missing ratio: `{missing['missing_ratio']:.6f}`; gap count: `{missing['gap_count']}`; max gap: `{missing['max_gap_seconds']:.6f} s`")
        lines.append(f"- `VA_mag` range: `{_float_or_none(va_stats.get('min'))}` to `{_float_or_none(va_stats.get('max'))}`; peak-to-peak: `{_float_or_none(va_stats.get('peak_to_peak'))}`")
        lines.append(f"- `IA_mag` range: `{_float_or_none(ia_stats.get('min'))}` to `{_float_or_none(ia_stats.get('max'))}`; peak-to-peak: `{_float_or_none(ia_stats.get('peak_to_peak'))}`")
        lines.append(f"- Dominant `Frequency` spectral peaks [Hz]: `{freq_dom}`")
        lines.append(f"- Dominant `ROCOF` spectral peaks [Hz]: `{rocof_dom}`")
        lines.append(f"- Estimated `Frequency` SNR [dB]: `{_float_or_none(freq_noise.get('snr_db'))}`; normality p-value of residual: `{_float_or_none(freq_noise.get('normaltest_pvalue'))}`")
        lines.append(f"- Estimated `ROCOF` SNR [dB]: `{_float_or_none(rocof_noise.get('snr_db'))}`; normality p-value of residual: `{_float_or_none(rocof_noise.get('normaltest_pvalue'))}`")
        if summary["event_spans"]:
            lines.append("- Detected abnormal spans:")
            for span in summary["event_spans"]:
                lines.append(
                    f"  - Event `{span['event_id']}` ({span['label']}): `{span['start_time']:.3f}s` to `{span['end_time']:.3f}s`, duration `{span['duration_s']:.3f}s`"
                )
        else:
            lines.append("- Detected abnormal spans: none")
        lines.append("")

    lines.append("## 7. Suggested next research extensions")
    lines.append("")
    lines.append("1. Build event-onset detectors from the derivative/Hilbert/spectral features already saved in the JSON files.")
    lines.append("2. Add topology-aware analysis using the RAW file to relate event magnitude with electrical distance.")
    lines.append("3. Extend the combined overlays to estimate propagation delay or bus ranking confidence intervals.")
    lines.append("4. Use the generated event windows as a deterministic feature-extraction stage for classifiers or sequence models.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


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
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline = load_timeline_metadata(timeline_xlsx)
    pmu_meta = load_pmu_metadata(pmu_meta_txt)

    print(f"[pmu_advanced] raw_dir       = {raw_dir}")
    print(f"[pmu_advanced] out_dir       = {out_dir}")
    print(f"[pmu_advanced] timeline_xlsx = {timeline_xlsx}")
    print(f"[pmu_advanced] pmu_meta_txt  = {pmu_meta_txt}")

    bus_frames: dict[str, pd.DataFrame] = {}
    per_bus_results: dict[str, dict[str, Any]] = {}

    for csv_path in iter_target_files(raw_dir, selected_bus):
        bus_id = extract_bus_id(csv_path.name)
        print(f"[pmu_advanced] analyzing {csv_path.name} ...")
        df, result = plot_bus(
            csv_path=csv_path,
            base_out_dir=out_dir,
            timeline=timeline,
            pmu_meta=pmu_meta,
            downsample=max(1, args.downsample),
            dpi=args.dpi,
            hires_dpi=args.hires_dpi,
            pre_s=args.event_pre_seconds,
            post_s=args.event_post_seconds,
            show=args.show,
        )
        bus_frames[bus_id] = df
        per_bus_results[bus_id] = result

    combined_summary = None
    if not args.skip_combined and len(bus_frames) >= 2:
        print("[pmu_advanced] running combined multi-PMU analysis ...")
        combined_summary = combined_analysis(
            base_out_dir=out_dir,
            bus_frames=bus_frames,
            per_bus_results=per_bus_results,
            pre_s=args.event_pre_seconds,
            post_s=args.event_post_seconds,
            dpi=args.dpi,
            show=args.show,
        )

    manifest = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "timeline_xlsx": str(timeline_xlsx) if timeline_xlsx else None,
        "pmu_meta_txt": str(pmu_meta_txt) if pmu_meta_txt else None,
        "analyzed_buses": sorted(per_bus_results),
        "timeline_records": timeline,
        "legend": {str(k): v for k, v in EVENT_LABELS.items()},
    }
    save_json(out_dir / "manifest.json", manifest)
    report_path = generate_markdown_report(
        out_dir=out_dir,
        raw_dir=raw_dir,
        pmu_meta_txt=pmu_meta_txt,
        timeline_xlsx=timeline_xlsx,
        per_bus_results=per_bus_results,
        combined_summary=combined_summary,
    )
    print(f"[pmu_advanced] report saved to {report_path}")
    print("[pmu_advanced] done.")


if __name__ == "__main__":
    main()
