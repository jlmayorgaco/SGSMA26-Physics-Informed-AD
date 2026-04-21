import argparse
import glob
import json
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
INPUT_DIR = "data/RAW0001/"
OUTPUT_DIR = "output/SCENARIO_RAW0001_NORMALIZED/"
CHUNKS_DIR = os.path.join(OUTPUT_DIR, "chunks")
BASELINES_CSV = os.path.join(OUTPUT_DIR, "normalization_baselines.csv")
CHUNK_INDEX_JSON = os.path.join(OUTPUT_DIR, "normalized_chunk_index.json")

EVENT_MAP = {
    0: "Normal operation",
    1: "Fault",
    2: "Line outage",
    3: "Generation change/outage",
    4: "Load change/drop",
    5: "Missing data",
    6: "Missing data + physical event",
    7: "Bad data",
    8: "Unknown event",
}

CATEGORY_BACKGROUNDS = {
    "Cyber": {"events": [5, 7], "color": "#E6D8FF", "alpha": 0.5},
    "Physical": {"events": [1, 2, 3, 4], "color": "#DDEBFF", "alpha": 0.5},
    "Cyber Physical": {"events": [6], "color": "#FFF5CC", "alpha": 0.5},
    "Unknown": {"events": [8], "color": "#EEEEEE", "alpha": 0.5},
}


def load_and_synchronize_data(input_dir: str):
    all_files = glob.glob(os.path.join(input_dir, "*_nanmask.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    bus_data = {}
    for file_path in all_files:
        bus_name = os.path.basename(file_path).split("_")[0]
        df = pd.read_csv(file_path)
        df["TIMESTAMP"] = pd.to_numeric(df["TIMESTAMP"], errors="coerce").round(3)
        df = df.dropna(subset=["TIMESTAMP"])
        df = df.drop_duplicates(subset=["TIMESTAMP"], keep="first")
        df = df.sort_values("TIMESTAMP")
        df.set_index("TIMESTAMP", inplace=True)
        bus_data[bus_name] = df

    event_df = pd.DataFrame()
    for bus_name, df in bus_data.items():
        event_df[bus_name] = pd.to_numeric(df["Event"], errors="coerce")
    event_df = event_df.sort_index().ffill().fillna(0).astype(int)
    return bus_data, event_df


def calculate_angular_speed(angle_deg_series: pd.Series, time_index: pd.Index) -> np.ndarray:
    rad_angles = np.deg2rad(pd.to_numeric(angle_deg_series, errors="coerce").to_numpy(dtype=float))
    time_vals = np.asarray(time_index, dtype=float)

    if len(rad_angles) == 0:
        return np.array([], dtype=float)

    # Fill occasional NaN gaps before unwrap/gradient.
    if np.isnan(rad_angles).any():
        valid = np.where(np.isfinite(rad_angles))[0]
        if len(valid) == 0:
            return np.zeros_like(rad_angles)
        rad_angles = np.interp(np.arange(len(rad_angles)), valid, rad_angles[valid])

    unwrapped_rad = np.unwrap(rad_angles)
    if len(unwrapped_rad) < 2:
        return np.zeros_like(unwrapped_rad)
    return np.gradient(unwrapped_rad, time_vals)


def wrap_degrees(values: np.ndarray) -> np.ndarray:
    return ((values + 180.0) % 360.0) - 180.0


def circular_mean_degrees(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return 0.0
    rad = np.deg2rad(clean)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))


def robust_trimmed_mean(values: pd.Series, trim_frac: float = 0.1) -> tuple[float, str]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return 1.0, "fallback_constant"
    return float(np.mean(clean)), "mean"


def robust_center(values: pd.Series) -> tuple[float, str]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return 0.0, "fallback_zero"
    if len(clean) >= 50:
        return float(np.median(clean)), "median"
    return float(np.mean(clean)), "mean"


def detect_signal_family(column_name: str) -> str:
    col = column_name.upper()
    if col.endswith("_ANG"):
        if any(tag in col for tag in ["_IA_ANG", "_IB_ANG", "_IC_ANG"]):
            return "current_angle"
        return "voltage_angle"
    if col.endswith("_MAG"):
        if any(tag in col for tag in ["_IA_MAG", "_IB_MAG", "_IC_MAG"]):
            return "current_mag"
        return "voltage_mag"
    if col.endswith("_FREQ"):
        return "frequency"
    if col.endswith("_ROCOF"):
        return "rocof"
    return "other"


def normalize_bus_data(bus_data: dict, event_df: pd.DataFrame):
    """
    Build normalized representation used by m3 calibration.

    Final representation:
    - ANG signals: convert to ANG_SPEED_RAD_S (rad/s)
    - Magnitudes/Frequency: divide by normal-state mean baseline
    - ROCOF: subtract robust center
    """
    print("Computing robust normal baselines and normalized data...")
    normalized_data = {}
    baseline_rows = []

    global_state = event_df.max(axis=1)
    normal_timestamps = global_state[global_state == 0].index

    for bus, df in bus_data.items():
        normal_df = df.loc[df.index.intersection(normal_timestamps)]
        if normal_df.empty:
            normal_df = df
            normal_source = "full_series_fallback"
        else:
            normal_source = "global_normal_event0"

        norm_df = pd.DataFrame(index=df.index)
        norm_df["DATA_PRESENT"] = pd.to_numeric(df["DATA_PRESENT"], errors="coerce")
        norm_df["Event"] = pd.to_numeric(df["Event"], errors="coerce").fillna(0).astype(int)

        for col in df.columns:
            if col in ["DATA_PRESENT", "Event"]:
                continue

            family = detect_signal_family(col)
            series = pd.to_numeric(df[col], errors="coerce")
            normal_series = pd.to_numeric(normal_df[col], errors="coerce") if col in normal_df.columns else series

            if family in ["voltage_angle", "current_angle"]:
                out_col = col.replace("ANG", "ANG_SPEED_RAD_S")
                norm_df[out_col] = calculate_angular_speed(series, df.index)
                baseline_rows.append(
                    {
                        "bus_id": bus,
                        "raw_signal": col,
                        "output_signal": out_col,
                        "signal_family": family,
                        "transform": "angle_to_angular_speed_rad_s",
                        "baseline_method": "none",
                        "baseline_value": np.nan,
                        "n_normal_samples": int(normal_series.dropna().shape[0]),
                        "baseline_source": normal_source,
                        "notes": "ANG signals are represented as angular speed in rad/s.",
                    }
                )

            elif family in ["voltage_mag", "current_mag", "frequency"]:
                baseline_value, baseline_method = robust_trimmed_mean(normal_series)
                if abs(baseline_value) < 1e-9:
                    baseline_value = 1.0
                    baseline_method = "fallback_constant"
                norm_df[col] = series / baseline_value
                baseline_rows.append(
                    {
                        "bus_id": bus,
                        "raw_signal": col,
                        "output_signal": col,
                        "signal_family": family,
                        "transform": "divide_by_baseline",
                        "baseline_method": baseline_method,
                        "baseline_value": float(baseline_value),
                        "n_normal_samples": int(normal_series.dropna().shape[0]),
                        "baseline_source": normal_source,
                        "notes": "PU normalization with robust normal-state baseline.",
                    }
                )

            elif family == "rocof":
                baseline_value, baseline_method = robust_center(normal_series)
                norm_df[col] = series - baseline_value
                baseline_rows.append(
                    {
                        "bus_id": bus,
                        "raw_signal": col,
                        "output_signal": col,
                        "signal_family": family,
                        "transform": "subtract_center",
                        "baseline_method": baseline_method,
                        "baseline_value": float(baseline_value),
                        "n_normal_samples": int(normal_series.dropna().shape[0]),
                        "baseline_source": normal_source,
                        "notes": "ROCOF centered around robust normal-state center.",
                    }
                )

            else:
                norm_df[col] = series
                baseline_rows.append(
                    {
                        "bus_id": bus,
                        "raw_signal": col,
                        "output_signal": col,
                        "signal_family": family,
                        "transform": "pass_through",
                        "baseline_method": "none",
                        "baseline_value": np.nan,
                        "n_normal_samples": int(normal_series.dropna().shape[0]),
                        "baseline_source": normal_source,
                        "notes": "Unrecognized signal family, left unchanged.",
                    }
                )

        normalized_data[bus] = norm_df

    baseline_df = pd.DataFrame(baseline_rows)
    return normalized_data, baseline_df


def find_chunk_boundaries(event_df: pd.DataFrame):
    global_state = event_df.max(axis=1)
    state_changes = global_state != global_state.shift(1)

    boundaries = global_state.index[state_changes].tolist()
    if event_df.index[0] not in boundaries:
        boundaries.insert(0, event_df.index[0])
    boundaries.append(event_df.index[-1])

    return boundaries, global_state


def _chunk_mask(index: pd.Index, start_t: float, end_t: float, include_end: bool) -> np.ndarray:
    if include_end:
        return (index >= start_t) & (index <= end_t)
    return (index >= start_t) & (index < end_t)


def plot_event_timeline(chunk_meta_list, max_time_s, output_path, bus_name=None):
    fig, ax1 = plt.subplots(figsize=(18, 3), dpi=200)
    ax1.set_xlim(0, max_time_s)
    ax1.set_ylim(-0.5, len(EVENT_MAP) - 0.5)

    ax1.set_yticks(list(EVENT_MAP.keys()))
    ax1.set_yticklabels(list(EVENT_MAP.values()), fontsize=6)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Event label")

    ax2 = ax1.twiny()
    ax2.set_xlim(0, max_time_s / 60.0)
    ax2.set_xlabel("Time (min)")
    ax1.axhline(y=0, color="black", linewidth=0.5)

    legend_patches = []
    for cat, props in CATEGORY_BACKGROUNDS.items():
        patch = mpatches.Patch(color=props["color"], label=cat, alpha=props["alpha"])
        legend_patches.append(patch)

    for chunk in chunk_meta_list:
        start = chunk["start_time_s"]
        duration = chunk["duration_s"]
        category = chunk.get("category")

        if category:
            props = CATEGORY_BACKGROUNDS[category]
            ax1.axvspan(
                start,
                start + duration,
                facecolor=props["color"],
                alpha=props["alpha"],
                edgecolor="black",
                linestyle="--",
                linewidth=0.5,
            )

        events_to_plot = (
            [chunk["per_bus_labels"].get(bus_name, 0)]
            if bus_name
            else [event_id for event_id in chunk["labels_present"] if event_id != 0]
        )

        for event_id in events_to_plot:
            if event_id == 0:
                continue
            color = (
                "#8da0cb"
                if event_id in [5, 7]
                else "#66c2a5"
                if event_id == 3
                else "#fc8d62"
                if event_id in [1, 2]
                else "gray"
            )
            ax1.broken_barh(
                [(start, duration)],
                (event_id - 0.4, 0.8),
                facecolors=color,
                edgecolors="black",
                linewidth=0.5,
            )

    title = f"{bus_name} Normalized - Event timeline" if bus_name else "All PMU buses Normalized - Event timeline"
    plt.title(title, pad=20)
    ax1.legend(handles=legend_patches, loc="upper right", fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def process_normalized_pipeline(generate_plots: bool = True):
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    print("Loading raw data...")
    raw_bus_data, event_df = load_and_synchronize_data(INPUT_DIR)

    bus_data, baselines_df = normalize_bus_data(raw_bus_data, event_df)
    baselines_df.to_csv(BASELINES_CSV, index=False)
    print(f"Normalization baseline report saved to: {BASELINES_CSV}")

    print("Computing exact chunk boundaries...")
    boundaries, global_state = find_chunk_boundaries(event_df)
    max_time_s = boundaries[-1]

    chunks_meta_list = []
    chunk_index_entries = []

    for i in range(len(boundaries) - 1):
        start_t = boundaries[i]
        end_t = boundaries[i + 1]
        include_end = i == len(boundaries) - 2

        dominant_label = int(global_state.loc[start_t])

        mask = _chunk_mask(event_df.index, start_t, end_t, include_end)
        event_slice = event_df.loc[mask]
        if event_slice.empty:
            continue

        labels_present = set(event_slice.values.flatten())
        per_bus_labels = {bus: int(event_slice[bus].max()) for bus in event_slice.columns}
        affected = [bus for bus, evt in per_bus_labels.items() if evt != 0]

        category = None
        for cat, props in CATEGORY_BACKGROUNDS.items():
            if dominant_label in props["events"]:
                category = cat
                break

        chunk_meta = {
            "chunk_order": i + 1,
            "label": int(dominant_label),
            "label_name": EVENT_MAP.get(dominant_label, "Unknown"),
            "category": category,
            "start_time_s": float(start_t),
            "end_time_s": float(end_t),
            "duration_s": float(end_t - start_t),
            "affected_buses": affected,
            "labels_present": sorted([int(label) for label in labels_present]),
            "per_bus_labels": per_bus_labels,
        }
        chunks_meta_list.append(chunk_meta)

        label_name_safe = EVENT_MAP[dominant_label].replace(" ", "_").replace("/", "_").lower()
        chunk_dir_name = f"chunk{i + 1:02d}_event_{dominant_label}_{label_name_safe}"
        chunk_path = os.path.join(CHUNKS_DIR, chunk_dir_name)
        os.makedirs(chunk_path, exist_ok=True)
        chunk_index_entry = {
            "chunk_id": chunk_dir_name,
            "chunk_path": chunk_path,
            "label": int(dominant_label),
            "label_name": EVENT_MAP.get(dominant_label, "Unknown"),
            "duration_s": float(end_t - start_t),
            "start_time_s": float(start_t),
            "end_time_s": float(end_t),
            "buses_available": [],
            "bus_columns": {},
        }

        for bus, df in bus_data.items():
            bus_mask = _chunk_mask(df.index, start_t, end_t, include_end)
            bus_slice = df.loc[bus_mask].copy()
            csv_out = os.path.join(chunk_path, f"{bus}_normalized.csv")
            bus_slice.to_csv(csv_out)
            if len(bus_slice) > 0:
                chunk_index_entry["buses_available"].append(bus)
                chunk_index_entry["bus_columns"][bus] = [c for c in bus_slice.columns]

            if not generate_plots:
                continue

            signals_to_plot = [col for col in bus_slice.columns if col not in ["DATA_PRESENT", "Event"]]
            if bus_slice.empty or len(signals_to_plot) == 0:
                continue

            fig, axes = plt.subplots(
                len(signals_to_plot),
                1,
                figsize=(12, 2 * len(signals_to_plot)),
                sharex=True,
            )
            if len(signals_to_plot) == 1:
                axes = [axes]

            for ax, signal in zip(axes, signals_to_plot):
                ax.plot(bus_slice.index, bus_slice[signal], linewidth=1.0, color="#1f77b4")

                signal_upper = signal.upper()
                if "MAG" in signal_upper or signal_upper.endswith("_FREQ"):
                    ax.axhline(1.0, color="red", linestyle="--", alpha=0.6, linewidth=1.0)
                elif "ROCOF" in signal_upper or "SPEED" in signal_upper or "DELTA" in signal_upper:
                    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.6, linewidth=1.0)

                ax.set_ylabel(signal, fontsize=8)
                ax.tick_params(axis="both", which="major", labelsize=8)
                ax.grid(True, alpha=0.3)

            plt.xlabel("Time (s)")
            plt.suptitle(
                f"{bus} Normalized Signals - Chunk {i + 1} ({chunk_meta['label_name']})",
                fontsize=14,
            )
            plt.tight_layout()
            plt.savefig(os.path.join(chunk_path, f"{bus}_normalized_signals_plot.png"), dpi=150)
            plt.close()

        chunk_index_entries.append(chunk_index_entry)

    with open(os.path.join(OUTPUT_DIR, "chunks_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump({"chunks": chunks_meta_list}, fh, indent=2)

    event0_chunks = [entry["chunk_id"] for entry in chunk_index_entries if int(entry.get("label", -1)) == 0]
    with open(CHUNK_INDEX_JSON, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "focus_event_label": 0,
                "event0_chunk_ids": event0_chunks,
                "chunks": chunk_index_entries,
            },
            fh,
            indent=2,
        )
    print(f"Normalized chunk index saved to: {CHUNK_INDEX_JSON}")

    if generate_plots:
        print("Generating normalized timeline plots...")
        plot_event_timeline(
            chunks_meta_list,
            max_time_s,
            os.path.join(OUTPUT_DIR, "event_plot_all_buses_timeline.png"),
        )
        for bus in raw_bus_data.keys():
            plot_event_timeline(
                chunks_meta_list,
                max_time_s,
                os.path.join(OUTPUT_DIR, f"event_plot_{bus}.png"),
                bus_name=bus,
            )

    print(f"m1 normalization pipeline complete. Output path: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize PMU chunks with robust current handling.")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip normalized signal plots and timelines",
    )
    args = parser.parse_args()
    process_normalized_pipeline(generate_plots=not args.skip_plots)
