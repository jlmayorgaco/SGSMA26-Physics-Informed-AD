"""Export helpers for M9 scenario artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math

import numpy as np
import pandas as pd

from src.domain.topology import bus_sort_key, canonical_bus_name
from src.simulation.m9.calibration import channel_stat, write_json
from src.simulation.m9.constants import PMU_BUSES_OFFICIAL, PMU_MEASUREMENT_SUFFIXES
from src.simulation.m9.cyber import build_cyber_frame_metadata, combine_event_label
from src.simulation.m9.physical import PhysicalTruth, compute_injection_currents_pu, truth_to_dataframe


def ensure_scenario_dirs(scenario_dir: str | Path) -> dict[str, Path]:
    root = Path(scenario_dir)
    paths = {
        "root": root,
        "config": root / "config",
        "pmu": root / "pmu",
        "all_buses": root / "all_buses",
        "labels": root / "labels",
        "metadata": root / "metadata",
        "plots": root / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _token(bus: str) -> str:
    return str(int("".join(ch for ch in canonical_bus_name(bus) if ch.isdigit())))


def _wrap_deg(x: np.ndarray) -> np.ndarray:
    return ((x + 180.0) % 360.0) - 180.0


def _safe_std(reference_stats: dict[str, Any], bus: str, suffix: str, default: float, max_fraction_of_mean: float = 0.02) -> float:
    mean = channel_stat(reference_stats, bus, suffix, "mean", default)
    std = channel_stat(reference_stats, bus, suffix, "increment_std", channel_stat(reference_stats, bus, suffix, "std", default))
    if suffix.endswith("MAG"):
        std = min(abs(std), max(abs(mean) * max_fraction_of_mean, 1e-6))
    if suffix == "Freq":
        std = min(abs(std), 0.025)
    if suffix == "ROCOF":
        std = min(abs(std), 0.15)
    if suffix.endswith("ANG"):
        std = min(abs(std), 1.50)
    return float(max(std, 1e-9))


def build_observed_pmu_frames(
    truth: PhysicalTruth,
    raw_path: str | Path,
    reference_stats: dict[str, Any],
    pmu_buses: list[str] | None = None,
    seed: int = 12345,
) -> dict[str, pd.DataFrame]:
    pmu_buses = pmu_buses or PMU_BUSES_OFFICIAL
    bus_pos = {canonical_bus_name(b): i for i, b in enumerate(truth.bus_order)}
    currents_pu = compute_injection_currents_pu(raw_path, truth.bus_order, truth.voltage_complex_pu)
    rng = np.random.default_rng(seed)
    frames: dict[str, pd.DataFrame] = {}
    for bus in pmu_buses:
        bus = canonical_bus_name(bus)
        if bus not in bus_pos:
            continue
        j = bus_pos[bus]
        z = truth.voltage_complex_pu[:, j]
        i_pu = currents_pu[:, j]
        mag_pu = np.abs(z)
        ang_deg = np.rad2deg(np.angle(z))
        base_mag = max(float(mag_pu[0]), 1e-6)
        base_ang = float(ang_deg[0])
        angle_delta = _wrap_deg(ang_deg - base_ang)
        current_mag_shape = 1.0 + 0.30 * (mag_pu / base_mag - 1.0) + 0.20 * np.abs(truth.frequency_hz - 60.0)
        current_ang_delta = angle_delta + _wrap_deg(np.rad2deg(np.angle(i_pu)) - float(np.rad2deg(np.angle(i_pu[0])) if len(i_pu) else 0.0)) * 0.15
        data: dict[str, Any] = {"TIMESTAMP": truth.timestamps.astype(float)}
        for phase, shift in [("VA", 0.0), ("VB", 120.0), ("VC", -120.0)]:
            suffix_ang = f"{phase}_ANG"
            suffix_mag = f"{phase}_MAG"
            mean_ang = channel_stat(reference_stats, bus, suffix_ang, "mean", float(ang_deg[0] + shift - 60.0))
            mean_mag = channel_stat(reference_stats, bus, suffix_mag, "mean", 200000.0)
            noise_ang = rng.normal(0.0, _safe_std(reference_stats, bus, suffix_ang, 0.08), size=len(truth.timestamps))
            noise_mag = rng.normal(0.0, _safe_std(reference_stats, bus, suffix_mag, 500.0), size=len(truth.timestamps))
            data[f"{bus}_{suffix_ang}"] = _wrap_deg(mean_ang + angle_delta + noise_ang)
            data[f"{bus}_{suffix_mag}"] = np.maximum(0.0, mean_mag * (mag_pu / base_mag) + noise_mag)
        for phase, shift in [("IA", -10.0), ("IB", 110.0), ("IC", -130.0)]:
            suffix_ang = f"{phase}_ANG"
            suffix_mag = f"{phase}_MAG"
            mean_ang = channel_stat(reference_stats, bus, suffix_ang, "mean", float(ang_deg[0] + shift - 70.0))
            mean_mag = channel_stat(reference_stats, bus, suffix_mag, "mean", 250.0)
            noise_ang = rng.normal(0.0, _safe_std(reference_stats, bus, suffix_ang, 0.10), size=len(truth.timestamps))
            noise_mag = rng.normal(0.0, _safe_std(reference_stats, bus, suffix_mag, 3.0, max_fraction_of_mean=0.05), size=len(truth.timestamps))
            data[f"{bus}_{suffix_ang}"] = _wrap_deg(mean_ang + current_ang_delta + noise_ang)
            data[f"{bus}_{suffix_mag}"] = np.maximum(0.0, mean_mag * current_mag_shape + noise_mag)
        freq_mean = channel_stat(reference_stats, bus, "Freq", "mean", 60.0)
        rocof_mean = channel_stat(reference_stats, bus, "ROCOF", "mean", 0.0)
        data[f"{bus}_Freq"] = freq_mean + (truth.frequency_hz - 60.0) + rng.normal(0.0, _safe_std(reference_stats, bus, "Freq", 0.005), size=len(truth.timestamps))
        data[f"{bus}_ROCOF"] = rocof_mean + truth.rocof_hz_per_s + rng.normal(0.0, _safe_std(reference_stats, bus, "ROCOF", 0.03), size=len(truth.timestamps))
        data["DATA_PRESENT"] = np.ones(len(truth.timestamps), dtype=int)
        data["Event"] = truth.event_by_frame.astype(int)
        cols = ["TIMESTAMP"] + [f"{bus}_{suffix}" for suffix in PMU_MEASUREMENT_SUFFIXES] + ["DATA_PRESENT", "Event"]
        frames[bus] = pd.DataFrame(data)[cols]
    return frames


def build_all_bus_measurements(truth: PhysicalTruth, raw_path: str | Path, pmu_buses: set[str]) -> pd.DataFrame:
    currents = compute_injection_currents_pu(raw_path, truth.bus_order, truth.voltage_complex_pu)
    rows: list[dict[str, Any]] = []
    for i, ts in enumerate(truth.timestamps):
        for j, bus in enumerate(truth.bus_order):
            v = truth.voltage_complex_pu[i, j]
            cur = currents[i, j]
            vang = float(np.rad2deg(np.angle(v)))
            imag = float(abs(cur) * 250.0)
            rows.append(
                {
                    "TIMESTAMP": float(ts),
                    "BUS": bus,
                    "VA_ANG": vang,
                    "VA_MAG": float(abs(v)),
                    "VB_ANG": float(_wrap_deg(np.asarray([vang + 120.0]))[0]),
                    "VB_MAG": float(abs(v)),
                    "VC_ANG": float(_wrap_deg(np.asarray([vang - 120.0]))[0]),
                    "VC_MAG": float(abs(v)),
                    "IA_ANG": float(_wrap_deg(np.asarray([np.rad2deg(np.angle(cur))]))[0]),
                    "IA_MAG": imag,
                    "IB_ANG": float(_wrap_deg(np.asarray([np.rad2deg(np.angle(cur)) + 120.0]))[0]),
                    "IB_MAG": imag,
                    "IC_ANG": float(_wrap_deg(np.asarray([np.rad2deg(np.angle(cur)) - 120.0]))[0]),
                    "IC_MAG": imag,
                    "Freq": float(truth.frequency_hz[i]),
                    "ROCOF": float(truth.rocof_hz_per_s[i]),
                    "IS_PMU_BUS": bool(bus in pmu_buses),
                    "OBSERVABLE_IN_INPUT": bool(bus in pmu_buses),
                    "EVENT": int(truth.event_by_frame[i]),
                }
            )
    return pd.DataFrame(rows)


def build_full_state_target(truth: PhysicalTruth, pmu_buses: set[str]) -> pd.DataFrame:
    df = truth_to_dataframe(truth, pmu_buses)
    return df[
        [
            "TIMESTAMP",
            "BUS",
            "IS_PMU_BUS",
            "IS_NON_PMU_BUS",
            "V_TRUE_REAL_PU",
            "V_TRUE_IMAG_PU",
            "V_TRUE_MAG_PU",
            "V_TRUE_ANG_DEG",
            "EVENT",
            "WINDOW_TYPE",
        ]
    ]


def write_parquet_if_available(df: pd.DataFrame, path: Path) -> bool:
    try:
        df.to_parquet(path, index=False)
        return True
    except Exception:
        return False


def write_pmu_csvs(pmu_frames: dict[str, pd.DataFrame], pmu_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for bus, df in sorted(pmu_frames.items(), key=lambda item: bus_sort_key(item[0])):
        path = pmu_dir / f"Bus{_token(bus)}_Competition_Data_sim.csv"
        df.to_csv(path, index=False)
        out[bus] = str(path)
    return out


def write_labels(
    paths: dict[str, Path],
    truth: PhysicalTruth,
    physical_events: list[dict[str, Any]],
    cyber_events: list[dict[str, Any]],
) -> dict[str, str]:
    cyber_labels, cyber_types, cyber_subtypes, cyber_targets = build_cyber_frame_metadata(truth.timestamps, cyber_events)
    rows: list[dict[str, Any]] = []
    for i, ts in enumerate(truth.timestamps):
        event = combine_event_label(int(truth.event_by_frame[i]), int(cyber_labels[i]), cyber_types[i])
        rows.append(
            {
                "TIMESTAMP": float(ts),
                "EVENT": int(event),
                "IS_ABNORMAL": bool(event != 0),
                "IS_PHYSICAL_EVENT": bool(int(truth.event_by_frame[i]) != 0),
                "IS_CYBER_EVENT": bool(int(cyber_labels[i]) != 0),
                "IS_CONCURRENT_EVENT": bool(int(truth.event_by_frame[i]) != 0 and int(cyber_labels[i]) != 0),
                "PHYSICAL_EVENT_TYPE": truth.physical_type_by_frame[i],
                "CYBER_EVENT_TYPE": cyber_types[i],
                "CYBER_SUBTYPE": cyber_subtypes[i],
                "TARGET_PMU": cyber_targets[i],
                "TARGET_BUS": truth.origin_bus_by_frame[i],
                "TARGET_LINE": truth.origin_line_by_frame[i],
            }
        )
    frame_labels = pd.DataFrame(rows)
    frame_path = paths["labels"] / "event_frame_labels.csv"
    frame_labels.to_csv(frame_path, index=False)

    intervals: list[dict[str, Any]] = []
    interval_columns = [
        "EVENT_ID",
        "EVENT",
        "EVENT_FAMILY",
        "EVENT_TYPE",
        "SUBTYPE",
        "START_TIME_S",
        "END_TIME_S",
        "TARGET_BUS",
        "TARGET_LINE",
        "TARGET_PMU",
    ]
    for event in physical_events:
        intervals.append({"EVENT_ID": event.get("event_id"), "EVENT": int(event.get("event_label", 8)), "EVENT_FAMILY": "physical", "EVENT_TYPE": event.get("event_type"), "SUBTYPE": event.get("subtype"), "START_TIME_S": event.get("start_time_s"), "END_TIME_S": event.get("end_time_s"), "TARGET_BUS": event.get("target_bus", ""), "TARGET_LINE": event.get("target_line", ""), "TARGET_PMU": ""})
    for event in cyber_events:
        intervals.append({"EVENT_ID": event.get("event_id"), "EVENT": int(event.get("event_label", 7)), "EVENT_FAMILY": "cyber", "EVENT_TYPE": event.get("event_type"), "SUBTYPE": event.get("subtype"), "START_TIME_S": event.get("start_time_s"), "END_TIME_S": event.get("end_time_s"), "TARGET_BUS": "", "TARGET_LINE": "", "TARGET_PMU": ";".join(event.get("target_pmus", []))})
    intervals_path = paths["labels"] / "event_intervals.csv"
    pd.DataFrame(intervals, columns=interval_columns).to_csv(intervals_path, index=False)

    loc_rows: list[dict[str, Any]] = []
    loc_columns = ["EVENT_ID", "EVENT", "LOCALIZATION_TYPE", "TARGET_BUS", "TARGET_LINE", "START_TIME_S", "END_TIME_S"]
    for event in physical_events:
        loc_rows.append({"EVENT_ID": event.get("event_id"), "EVENT": int(event.get("event_label", 8)), "LOCALIZATION_TYPE": "bus" if event.get("target_bus") else "line", "TARGET_BUS": event.get("target_bus", ""), "TARGET_LINE": event.get("target_line", ""), "START_TIME_S": event.get("start_time_s"), "END_TIME_S": event.get("end_time_s")})
    loc_path = paths["labels"] / "localization_targets.csv"
    pd.DataFrame(loc_rows, columns=loc_columns).to_csv(loc_path, index=False)
    return {"event_frame_labels": str(frame_path), "event_intervals": str(intervals_path), "localization_targets": str(loc_path)}


def write_metadata(paths: dict[str, Path], pmu_buses: list[str], truth: PhysicalTruth, reference_stats: dict[str, Any], raw_vs_sim: dict[str, Any] | None = None) -> dict[str, str]:
    mapping = {f"PMU{i + 1}": bus for i, bus in enumerate(pmu_buses)}
    pmu_map_path = paths["metadata"] / "pmu_bus_mapping.json"
    bus_index_path = paths["metadata"] / "bus_index_map.json"
    network_path = paths["metadata"] / "network_summary.json"
    stats_path = paths["metadata"] / "statistics_reference.json"
    comp_path = paths["metadata"] / "raw_vs_sim_comparison.json"
    write_json(pmu_map_path, {"pmu_buses": pmu_buses, "pmu_map": mapping})
    write_json(bus_index_path, {bus: i for i, bus in enumerate(truth.bus_order)})
    write_json(network_path, {"bus_count": len(truth.bus_order), "buses": truth.bus_order, **truth.metadata})
    write_json(stats_path, reference_stats)
    if raw_vs_sim is not None:
        write_json(comp_path, raw_vs_sim)
    return {"pmu_bus_mapping": str(pmu_map_path), "bus_index_map": str(bus_index_path), "network_summary": str(network_path), "statistics_reference": str(stats_path), "raw_vs_sim_comparison": str(comp_path)}


def write_summary_markdown(path: Path, manifest: dict[str, Any], validation: dict[str, Any] | None = None) -> None:
    lines = [
        f"# Scenario {manifest['scenario_id']}",
        "",
        f"Template: `{manifest.get('scenario_template')}`",
        f"Seed: `{manifest.get('seed')}`",
        f"Duration: `{manifest.get('duration_s')}` s at `{manifest.get('fps')}` fps",
        f"PMU buses: {', '.join(manifest.get('pmu_buses', []))}",
        "",
        "## Physical Events",
    ]
    if manifest.get("physical_events"):
        for event in manifest["physical_events"]:
            lines.append(f"- `{event.get('event_id')}` label {event.get('event_label')}: {event.get('event_type')} / {event.get('subtype')} from {event.get('start_time_s')} to {event.get('end_time_s')} target bus={event.get('target_bus')} line={event.get('target_line')}")
    else:
        lines.append("- None")
    lines.extend(["", "## Cyber Events"])
    if manifest.get("cyber_events"):
        for event in manifest["cyber_events"]:
            lines.append(f"- `{event.get('event_id')}` label {event.get('event_label')}: {event.get('event_type')} / {event.get('subtype')} from {event.get('start_time_s')} to {event.get('end_time_s')} target PMUs={event.get('target_pmus')}")
    else:
        lines.append("- None")
    if validation:
        lines.extend(["", "## Validation", f"- Scenario valid: `{validation.get('scenario_valid')}`", f"- Downstream ready: `{validation.get('downstream_readiness', {}).get('overall_ready')}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(paths: dict[str, Path], truth: PhysicalTruth, pmu_frames: dict[str, pd.DataFrame], raw_vs_sim: dict[str, Any] | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        import matplotlib.pyplot as plt

        t = truth.timestamps
        labels = truth.event_by_frame
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, labels, drawstyle="steps-post")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Physical Event")
        ax.set_title("Scenario timeline")
        path = paths["plots"] / "scenario_timeline.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        out["scenario_timeline"] = str(path)

        fig, ax = plt.subplots(figsize=(10, 4))
        for bus, df in list(pmu_frames.items())[:8]:
            col = f"{bus}_VA_MAG"
            if col in df:
                ax.plot(df["TIMESTAMP"], df[col], label=bus, alpha=0.85)
        ax.set_title("PMU voltage magnitude overview")
        ax.set_xlabel("Time (s)")
        ax.legend(ncol=4, fontsize=8)
        path = paths["plots"] / "pmu_overview.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        out["pmu_overview"] = str(path)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, np.abs(truth.voltage_complex_pu).mean(axis=1), label="All-bus mean |V|")
        ax.set_title("All-bus truth overview")
        ax.set_xlabel("Time (s)")
        ax.legend()
        path = paths["plots"] / "all_bus_overview.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        out["all_bus_overview"] = str(path)

        fig, ax = plt.subplots(figsize=(10, 3))
        for bus, df in pmu_frames.items():
            ax.plot(df["TIMESTAMP"], df["DATA_PRESENT"] + 0.03 * len(out), label=bus, alpha=0.6)
        ax.set_ylim(-0.2, 1.4)
        ax.set_title("DATA_PRESENT overview")
        ax.set_xlabel("Time (s)")
        path = paths["plots"] / "data_present_overview.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        out["data_present_overview"] = str(path)

        fig, ax = plt.subplots(figsize=(8, 4))
        if raw_vs_sim:
            summary = raw_vs_sim.get("distance_summary", {})
            vals = [summary.get("mean_normalized_mean_delta", 0.0), summary.get("mean_normalized_std_delta", 0.0)]
            ax.bar(["mean delta", "std delta"], vals)
        ax.set_title("RAW vs SIM statistical distance")
        path = paths["plots"] / "raw_vs_sim_stats.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        out["raw_vs_sim_stats"] = str(path)

        fig, ax = plt.subplots(figsize=(10, 3))
        for bus, df in pmu_frames.items():
            bad = (df["Event"].astype(int).isin([5, 6, 7, 8])).astype(int)
            ax.plot(df["TIMESTAMP"], bad, alpha=0.5, label=bus)
        ax.set_title("Cyber/data-quality event overview")
        path = paths["plots"] / "cyber_event_overview.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        out["cyber_event_overview"] = str(path)
    except Exception:
        return out
    return out
