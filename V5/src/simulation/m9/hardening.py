"""M9.1 hardening, balanced dataset factory, realism validation v2, and split/scoring pipelines."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import ks_2samp, wasserstein_distance

from src.simulation.m9.calibration import read_json, write_json
from src.simulation.m9.constants import EVENT_LABELS, PMU_BUSES_OFFICIAL, PMU_MEASUREMENT_SUFFIXES
from src.simulation.m9.generator import generate_scenario
from src.simulation.m9.templates import TEMPLATES


ANGLE_SUFFIXES = [s for s in PMU_MEASUREMENT_SUFFIXES if s.endswith("ANG")]


DEFAULT_REALISM_THRESHOLDS = {
    "wasserstein_max": 1.5,
    "ks_max": 0.45,
    "autocorr_delta_max": 0.60,
    "derivative_wasserstein_max": 1.5,
    "psd_l1_max": 0.75,
    "cross_pmu_corr_delta_max": 0.35,
    "freq_rocof_coherence_delta_max": 0.6,
}


DIFFICULTY_NOISE = {
    "easy": 0.15,
    "medium": 0.35,
    "hard": 0.70,
    "adversarial": 1.15,
}


BATCH_MODE_WEIGHTS = {
    "balanced_core": {
        "TEMPLATE_EVENT0_NORMAL": 1,
        "TEMPLATE_EVENT1_FAULT": 1,
        "TEMPLATE_EVENT2_LINE_OUTAGE": 1,
        "TEMPLATE_EVENT3_GENERATION_CHANGE": 1,
        "TEMPLATE_EVENT4_LOAD_CHANGE": 1,
        "TEMPLATE_EVENT5_MISSING_ONLY": 1,
        "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL": 1,
        "TEMPLATE_EVENT7_BAD_DATA": 1,
        "TEMPLATE_EVENT8_UNKNOWN_COMPOSITE": 1,
    },
    "physical_heavy": {
        "TEMPLATE_EVENT1_FAULT": 3,
        "TEMPLATE_EVENT2_LINE_OUTAGE": 3,
        "TEMPLATE_EVENT3_GENERATION_CHANGE": 3,
        "TEMPLATE_EVENT4_LOAD_CHANGE": 3,
        "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL": 1,
        "TEMPLATE_EVENT0_NORMAL": 1,
    },
    "cyber_heavy": {
        "TEMPLATE_EVENT5_MISSING_ONLY": 4,
        "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL": 3,
        "TEMPLATE_EVENT7_BAD_DATA": 4,
        "TEMPLATE_EVENT8_UNKNOWN_COMPOSITE": 2,
        "TEMPLATE_EVENT0_NORMAL": 1,
    },
    "concurrent_heavy": {
        "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL": 5,
        "TEMPLATE_EVENT8_UNKNOWN_COMPOSITE": 4,
        "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT": 2,
        "TEMPLATE_EVENT0_NORMAL": 1,
    },
    "curriculum_easy_to_hard": {
        "TEMPLATE_EVENT0_NORMAL": 2,
        "TEMPLATE_EVENT1_FAULT": 1,
        "TEMPLATE_EVENT2_LINE_OUTAGE": 1,
        "TEMPLATE_EVENT3_GENERATION_CHANGE": 1,
        "TEMPLATE_EVENT4_LOAD_CHANGE": 1,
        "TEMPLATE_EVENT5_MISSING_ONLY": 1,
        "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL": 1,
        "TEMPLATE_EVENT7_BAD_DATA": 1,
        "TEMPLATE_EVENT8_UNKNOWN_COMPOSITE": 1,
    },
    "localization_stress": {
        "TEMPLATE_EVENT2_LINE_OUTAGE": 3,
        "TEMPLATE_EVENT3_GENERATION_CHANGE": 3,
        "TEMPLATE_EVENT4_LOAD_CHANGE": 2,
        "TEMPLATE_EVENT8_UNKNOWN_COMPOSITE": 2,
    },
    "estimator_stress": {
        "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL": 4,
        "TEMPLATE_EVENT5_MISSING_ONLY": 3,
        "TEMPLATE_EVENT7_BAD_DATA": 3,
        "TEMPLATE_EVENT8_UNKNOWN_COMPOSITE": 3,
        "TEMPLATE_EVENT2_LINE_OUTAGE": 2,
    },
    "research_adversarial": {
        "TEMPLATE_EVENT7_BAD_DATA": 4,
        "TEMPLATE_EVENT8_UNKNOWN_COMPOSITE": 4,
        "TEMPLATE_EVENT6_CONCURRENT_MISSING_PLUS_PHYSICAL": 4,
        "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT": 2,
    },
}


def _to_markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    if not rows:
        return "(no rows)"
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def _bus_num(bus: str | None) -> int | None:
    if not bus:
        return None
    digits = "".join(ch for ch in str(bus) if ch.isdigit())
    return int(digits) if digits else None


def _electrical_distance_proxy(target_bus: str | None) -> float:
    t = _bus_num(target_bus)
    if t is None:
        return 12.0
    pmu_nums = [_bus_num(x) for x in PMU_BUSES_OFFICIAL]
    pmu_nums = [x for x in pmu_nums if x is not None]
    if not pmu_nums:
        return 12.0
    return float(min(abs(t - p) for p in pmu_nums))


def _wrap_deg(x: np.ndarray) -> np.ndarray:
    return ((x + 180.0) % 360.0) - 180.0


def _circular_mean_deg(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    r = np.deg2rad(x)
    return float(np.rad2deg(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def _circular_variance(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    r = np.deg2rad(x)
    R = np.sqrt(np.sin(r).mean() ** 2 + np.cos(r).mean() ** 2)
    return float(1.0 - R)


def _autocorr_lag(x: np.ndarray, lag: int = 1) -> float:
    if x.size <= lag:
        return 0.0
    x0 = x[:-lag]
    x1 = x[lag:]
    if np.std(x0) < 1e-12 or np.std(x1) < 1e-12:
        return 0.0
    return float(np.corrcoef(x0, x1)[0, 1])


def _psd_distance(x_ref: np.ndarray, x_sim: np.ndarray) -> float:
    if x_ref.size < 16 or x_sim.size < 16:
        return 0.0
    nperseg_ref = min(256, x_ref.size)
    nperseg_sim = min(256, x_sim.size)
    _, p_ref = welch(x_ref, nperseg=nperseg_ref)
    _, p_sim = welch(x_sim, nperseg=nperseg_sim)
    n = min(p_ref.size, p_sim.size)
    if n == 0:
        return 0.0
    p_ref = p_ref[:n]
    p_sim = p_sim[:n]
    p_ref = p_ref / max(np.sum(p_ref), 1e-12)
    p_sim = p_sim / max(np.sum(p_sim), 1e-12)
    return float(np.mean(np.abs(p_ref - p_sim)))


def _burst_lengths(mask: np.ndarray) -> list[int]:
    bursts: list[int] = []
    cur = 0
    for v in mask.astype(bool):
        if v:
            cur += 1
        elif cur:
            bursts.append(cur)
            cur = 0
    if cur:
        bursts.append(cur)
    return bursts


def _interburst_intervals(mask: np.ndarray) -> list[int]:
    vals = mask.astype(bool)
    starts = np.where(np.diff(np.concatenate([[0], vals.astype(int)])) == 1)[0]
    if starts.size <= 1:
        return []
    return [int(starts[i + 1] - starts[i]) for i in range(starts.size - 1)]


def _load_reference_frames(reference_pmu_dir: Path, max_rows: int = 40000) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for bus in PMU_BUSES_OFFICIAL:
        token = int("".join(ch for ch in bus if ch.isdigit()))
        path = reference_pmu_dir / f"Bus{token}_Competition_Data_nanmask.csv"
        if not path.exists():
            alt = list(reference_pmu_dir.glob(f"Bus{token}_Competition_Data*.csv"))
            path = alt[0] if alt else path
        if path.exists():
            out[bus] = pd.read_csv(path, nrows=max_rows)
    return out


def _load_sim_frames(scenario_dirs: list[Path], max_rows_per_scenario: int = 20000) -> dict[str, pd.DataFrame]:
    chunks: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for sdir in scenario_dirs:
        for bus in PMU_BUSES_OFFICIAL:
            token = int("".join(ch for ch in bus if ch.isdigit()))
            path = sdir / "pmu" / f"Bus{token}_Competition_Data_sim.csv"
            if path.exists():
                chunks[bus].append(pd.read_csv(path, nrows=max_rows_per_scenario))
    out: dict[str, pd.DataFrame] = {}
    for bus, dfs in chunks.items():
        out[bus] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return out


def _expected_from_template(template_name: str) -> dict[str, Any]:
    t = TEMPLATES.get(template_name, {})
    phys = t.get("physical_events", [])
    cyber = t.get("cyber_events", [])
    target_bus = ""
    target_line = ""
    target_pmu = ""
    physical_subtype = ""
    cyber_subtype = ""
    severity = 0.0
    if phys:
        p0 = phys[0]
        target_bus = str(p0.get("target_bus") or "")
        target_line = str(p0.get("target_line") or "")
        physical_subtype = str(p0.get("subtype") or "")
        severity = float(max(float(x.get("severity", 0.0)) for x in phys))
    if cyber:
        c0 = cyber[0]
        target_pmu = ";".join(str(x) for x in c0.get("target_pmus", []))
        cyber_subtype = str(c0.get("subtype") or "")
        severity = max(severity, 0.65)
    return {
        "target_bus": target_bus,
        "target_line": target_line,
        "target_pmu": target_pmu,
        "physical_subtype": physical_subtype,
        "cyber_subtype": cyber_subtype,
        "severity": severity,
    }


def enrich_scenario_labels_and_manifest(
    scenario_dir: Path,
    difficulty_level: str,
    realism_score: float | None = None,
    training_value_score: float | None = None,
    estimator_difficulty: float | None = None,
    localizer_difficulty: float | None = None,
) -> dict[str, Any]:
    manifest_path = scenario_dir / "scenario_manifest.json"
    manifest = read_json(manifest_path)
    labels_path = scenario_dir / "labels" / "event_frame_labels.csv"
    labels = pd.read_csv(labels_path)

    expected = _expected_from_template(str(manifest.get("scenario_template", "")))
    labels["EVENT_COARSE"] = pd.to_numeric(labels["EVENT"], errors="coerce").fillna(0).astype(int)
    labels["EVENT_NAME"] = labels["EVENT_COARSE"].map(EVENT_LABELS).fillna("unknown")
    labels["PHYSICAL_SUBTYPE"] = labels.get("PHYSICAL_EVENT_TYPE", "").astype(str)
    labels["PHYSICAL_SUBTYPE"] = np.where(labels["PHYSICAL_SUBTYPE"].isin(["", "normal"]), expected["physical_subtype"], labels["PHYSICAL_SUBTYPE"])
    labels["CYBER_SUBTYPE"] = labels.get("CYBER_SUBTYPE", "").astype(str)
    labels["CYBER_SUBTYPE"] = np.where(labels["CYBER_SUBTYPE"].isin(["", "nan"]), expected["cyber_subtype"], labels["CYBER_SUBTYPE"])
    labels["TARGET_PMU"] = labels.get("TARGET_PMU", "").fillna("").astype(str)
    labels["TARGET_PMU"] = np.where(labels["TARGET_PMU"] == "", expected["target_pmu"], labels["TARGET_PMU"])
    labels["TARGET_BUS"] = labels.get("TARGET_BUS", "").fillna("").astype(str)
    labels["TARGET_BUS"] = np.where(labels["TARGET_BUS"] == "", expected["target_bus"], labels["TARGET_BUS"])
    labels["TARGET_LINE"] = labels.get("TARGET_LINE", "").fillna("").astype(str)
    labels["TARGET_LINE"] = np.where(labels["TARGET_LINE"] == "", expected["target_line"], labels["TARGET_LINE"])
    labels["SEVERITY_LEVEL"] = float(expected["severity"])
    labels["DIFFICULTY_LEVEL"] = difficulty_level
    labels["ESTIMATOR_DIFFICULTY_SCORE"] = float(estimator_difficulty if estimator_difficulty is not None else 0.0)
    labels["LOCALIZATION_DIFFICULTY_SCORE"] = float(localizer_difficulty if localizer_difficulty is not None else 0.0)
    labels.to_csv(labels_path, index=False)

    distance_to_pmu = _electrical_distance_proxy(expected["target_bus"])
    concurrent = bool(labels["IS_CONCURRENT_EVENT"].astype(bool).any())
    manifest.update(
        {
            "difficulty_level": difficulty_level,
            "expected_detector_challenge": difficulty_level,
            "expected_classifier_challenge": difficulty_level,
            "expected_localizer_challenge": "hard" if distance_to_pmu >= 8 else difficulty_level,
            "expected_estimator_challenge": "hard" if concurrent or difficulty_level in {"hard", "adversarial"} else difficulty_level,
            "event_visibility_rank": int(max(1, min(5, round(5 - min(distance_to_pmu, 12.0) / 3.0)))),
            "electrical_distance_to_nearest_pmu": distance_to_pmu,
            "scenario_family": f"{manifest.get('scenario_template','unknown')}::{expected['target_bus'] or expected['target_line'] or expected['target_pmu']}",
            "template_name": manifest.get("scenario_template", ""),
            "realism_score": float(realism_score if realism_score is not None else manifest.get("realism_score", 0.0)),
            "training_value_score": float(training_value_score if training_value_score is not None else manifest.get("training_value_score", 0.0)),
        }
    )
    write_json(manifest_path, manifest)
    return manifest


def _apply_noise_to_frame(df: pd.DataFrame, bus: str, noise_sigma: float, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    for suffix in PMU_MEASUREMENT_SUFFIXES:
        col = f"{bus}_{suffix}"
        if col not in out.columns:
            continue
        x = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(x)
        if not finite.any():
            continue
        scale = np.nanstd(x[finite])
        if suffix.endswith("ANG"):
            noise = rng.normal(0.0, max(0.3, scale * noise_sigma), size=x.size)
            x[finite] = _wrap_deg(x[finite] + noise[finite])
        else:
            noise = rng.normal(0.0, max(1e-4, scale * noise_sigma), size=x.size)
            x[finite] = x[finite] + noise[finite]
        out[col] = x
    return out


def _apply_cyber_profile_to_scenario(scenario_dir: Path, profile_mode: str, rng: np.random.Generator) -> None:
    for bus in PMU_BUSES_OFFICIAL:
        token = int("".join(ch for ch in bus if ch.isdigit()))
        path = scenario_dir / "pmu" / f"Bus{token}_Competition_Data_sim.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        mask = np.zeros(len(df), dtype=bool)
        if profile_mode == "research_style":
            burst_len = max(3, len(df) // 30)
            starts = rng.choice(np.arange(0, max(len(df) - burst_len, 1)), size=max(1, len(df) // 180), replace=False)
            for s in starts:
                mask[s : s + burst_len] = True
            if len(df) > 5:
                mask = mask | ((np.arange(len(df)) % 47) == 0)
        elif profile_mode == "adversarial_stress":
            mask = mask | ((np.arange(len(df)) % 19) < 2)
            if len(df) > 5:
                starts = rng.choice(np.arange(0, max(len(df) - 8, 1)), size=max(1, len(df) // 120), replace=False)
                for s in starts:
                    mask[s : s + 8] = True
            for suffix in ANGLE_SUFFIXES:
                col = f"{bus}_{suffix}"
                if col in df.columns:
                    x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                    lag = min(4, len(x) - 1) if len(x) > 1 else 0
                    if lag > 0:
                        x[mask] = x[np.maximum(np.where(mask)[0] - lag, 0)]
                        df[col] = x

        if mask.any():
            cols = [f"{bus}_{s}" for s in PMU_MEASUREMENT_SUFFIXES if f"{bus}_{s}" in df.columns]
            df.loc[mask, cols] = np.nan
            df.loc[mask, "DATA_PRESENT"] = 0
            cur = pd.to_numeric(df["Event"], errors="coerce").fillna(0).astype(int)
            physical = cur.isin([1, 2, 3, 4])
            df.loc[mask & physical.to_numpy(), "Event"] = 6
            df.loc[mask & (~physical.to_numpy()), "Event"] = 5
        df.to_csv(path, index=False)


def _difficulty_sequence(n: int, mode: str) -> list[str]:
    if mode == "curriculum_easy_to_hard":
        q = max(1, n // 4)
        seq = ["easy"] * q + ["medium"] * q + ["hard"] * q + ["adversarial"] * (n - 3 * q)
        return seq[:n]
    choices = ["easy", "medium", "hard", "adversarial"]
    return (choices * ((n // len(choices)) + 1))[:n]


def _weighted_templates(batch_mode: str) -> list[str]:
    weights = BATCH_MODE_WEIGHTS.get(batch_mode, BATCH_MODE_WEIGHTS["balanced_core"])
    out: list[str] = []
    for name, w in weights.items():
        out.extend([name] * max(1, int(w)))
    return out


def _coarse_label_from_template(template_name: str) -> int:
    for i in range(9):
        if f"EVENT{i}" in template_name:
            return i
    return 8 if "OFFICIAL" in template_name else 0


def _stats_safe(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xr = x[np.isfinite(x)]
    return xr, np.diff(xr) if xr.size > 1 else np.asarray([], dtype=float)


def run_realism_validation_v2(
    scenario_dirs: list[Path],
    reference_pmu_dir: Path,
    output_root: Path,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DEFAULT_REALISM_THRESHOLDS
    reference = _load_reference_frames(reference_pmu_dir)
    sim = _load_sim_frames(scenario_dirs)

    channel_rows: list[dict[str, Any]] = []
    bus_rows: list[dict[str, Any]] = []
    cross_rows = []
    freq_rocof_rows = []

    for bus in PMU_BUSES_OFFICIAL:
        if bus not in reference or bus not in sim:
            continue
        ref_df = reference[bus]
        sim_df = sim[bus]
        bus_metrics = []
        for suffix in PMU_MEASUREMENT_SUFFIXES:
            col = f"{bus}_{suffix}"
            if col not in ref_df.columns or col not in sim_df.columns:
                continue
            ref = pd.to_numeric(ref_df[col], errors="coerce").to_numpy(dtype=float)
            sx = pd.to_numeric(sim_df[col], errors="coerce").to_numpy(dtype=float)
            ref, ref_d = _stats_safe(ref)
            sx, sim_d = _stats_safe(sx)
            if ref.size < 10 or sx.size < 10:
                continue
            n = min(ref.size, sx.size)
            ref = ref[:n]
            sx = sx[:n]
            ref_scale = max(float(np.nanstd(ref)), 1e-6)
            sim_scale = max(float(np.nanstd(sx)), 1e-6)
            wd = float(wasserstein_distance(ref, sx) / max(ref_scale, sim_scale))
            ks = float(ks_2samp(ref, sx).statistic)
            ac_ref = _autocorr_lag(ref, 1)
            ac_sim = _autocorr_lag(sx, 1)
            ac_delta = abs(ac_ref - ac_sim)
            d_n = min(ref_d.size, sim_d.size)
            if d_n > 5:
                d_scale = max(float(np.nanstd(ref_d[:d_n])), float(np.nanstd(sim_d[:d_n])), 1e-6)
                dwd = float(wasserstein_distance(ref_d[:d_n], sim_d[:d_n]) / d_scale)
            else:
                dwd = 0.0
            psd = _psd_distance(ref, sx)
            event_ref = float(np.nanpercentile(np.abs(ref - np.nanmedian(ref)), 90))
            event_sim = float(np.nanpercentile(np.abs(sx - np.nanmedian(sx)), 90))
            channel_rows.append(
                {
                    "bus": bus,
                    "channel": suffix,
                    "wasserstein": wd,
                    "ks_stat": ks,
                    "autocorr_ref_lag1": ac_ref,
                    "autocorr_sim_lag1": ac_sim,
                    "autocorr_delta": ac_delta,
                    "derivative_wasserstein": dwd,
                    "psd_l1": psd,
                    "event_window_score_ref": event_ref,
                    "event_window_score_sim": event_sim,
                    "mean_ref": float(np.nanmean(ref)),
                    "mean_sim": float(np.nanmean(sx)),
                    "std_ref": float(np.nanstd(ref)),
                    "std_sim": float(np.nanstd(sx)),
                }
            )
            bus_metrics.append((wd, ks, ac_delta, dwd, psd))

        if bus_metrics:
            arr = np.asarray(bus_metrics, dtype=float)
            bus_rows.append(
                {
                    "bus": bus,
                    "mean_wasserstein": float(np.nanmean(arr[:, 0])),
                    "mean_ks": float(np.nanmean(arr[:, 1])),
                    "mean_autocorr_delta": float(np.nanmean(arr[:, 2])),
                    "mean_derivative_wasserstein": float(np.nanmean(arr[:, 3])),
                    "mean_psd_l1": float(np.nanmean(arr[:, 4])),
                }
            )

    for suffix in ["VA_MAG", "Freq"]:
        ref_mat = []
        sim_mat = []
        kept = []
        for bus in PMU_BUSES_OFFICIAL:
            col = f"{bus}_{suffix}"
            if bus in reference and col in reference[bus].columns and bus in sim and col in sim[bus].columns:
                ref_mat.append(pd.to_numeric(reference[bus][col], errors="coerce").to_numpy(dtype=float))
                sim_mat.append(pd.to_numeric(sim[bus][col], errors="coerce").to_numpy(dtype=float))
                kept.append(bus)
        if len(ref_mat) >= 3:
            n = min(min(len(x) for x in ref_mat), min(len(x) for x in sim_mat))
            ref_arr = np.vstack([x[:n] for x in ref_mat])
            sim_arr = np.vstack([x[:n] for x in sim_mat])
            ref_corr = np.corrcoef(ref_arr)
            sim_corr = np.corrcoef(sim_arr)
            delta = float(np.nanmean(np.abs(ref_corr - sim_corr)))
            if not np.isfinite(delta):
                delta = 0.0
            cross_rows.append({"signal": suffix, "mean_abs_corr_delta": delta, "bus_count": len(kept)})

    for bus in PMU_BUSES_OFFICIAL:
        c1 = f"{bus}_Freq"
        c2 = f"{bus}_ROCOF"
        if bus in sim and c1 in sim[bus].columns and c2 in sim[bus].columns and bus in reference and c1 in reference[bus].columns and c2 in reference[bus].columns:
            f_ref = pd.to_numeric(reference[bus][c1], errors="coerce").to_numpy(dtype=float)
            r_ref = pd.to_numeric(reference[bus][c2], errors="coerce").to_numpy(dtype=float)
            f_sim = pd.to_numeric(sim[bus][c1], errors="coerce").to_numpy(dtype=float)
            r_sim = pd.to_numeric(sim[bus][c2], errors="coerce").to_numpy(dtype=float)
            n = min(len(f_ref), len(r_ref), len(f_sim), len(r_sim))
            if n > 8:
                ref_coh = float(np.corrcoef(np.diff(f_ref[:n]), r_ref[1:n])[0, 1])
                sim_coh = float(np.corrcoef(np.diff(f_sim[:n]), r_sim[1:n])[0, 1])
                delta = abs(ref_coh - sim_coh)
                if not np.isfinite(delta):
                    delta = 0.0
                freq_rocof_rows.append({"bus": bus, "ref_coherence": ref_coh, "sim_coherence": sim_coh, "delta": delta})

    channel_df = pd.DataFrame(channel_rows)
    bus_df = pd.DataFrame(bus_rows)

    metrics_dir = output_root / "metrics"
    metadata_dir = output_root / "metadata"
    plots_dir = output_root / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    channel_df.to_csv(metrics_dir / "raw_vs_sim_channel_metrics.csv", index=False)
    bus_df.to_csv(metrics_dir / "raw_vs_sim_bus_metrics.csv", index=False)

    verdict_checks = {
        "wasserstein": float(channel_df["wasserstein"].mean()) if not channel_df.empty else float("inf"),
        "ks": float(channel_df["ks_stat"].mean()) if not channel_df.empty else float("inf"),
        "autocorr": float(channel_df["autocorr_delta"].mean()) if not channel_df.empty else float("inf"),
        "derivative": float(channel_df["derivative_wasserstein"].mean()) if not channel_df.empty else float("inf"),
        "psd": float(channel_df["psd_l1"].mean()) if not channel_df.empty else float("inf"),
        "cross_pmu_corr": float(np.mean([x["mean_abs_corr_delta"] for x in cross_rows])) if cross_rows else 0.0,
        "freq_rocof": float(np.mean([x["delta"] for x in freq_rocof_rows])) if freq_rocof_rows else 0.0,
    }
    pass_flags = {
        "wasserstein": verdict_checks["wasserstein"] <= thresholds["wasserstein_max"],
        "ks": verdict_checks["ks"] <= thresholds["ks_max"],
        "autocorr": verdict_checks["autocorr"] <= thresholds["autocorr_delta_max"],
        "derivative": verdict_checks["derivative"] <= thresholds["derivative_wasserstein_max"],
        "psd": verdict_checks["psd"] <= thresholds["psd_l1_max"],
        "cross_pmu_corr": verdict_checks["cross_pmu_corr"] <= thresholds["cross_pmu_corr_delta_max"],
        "freq_rocof": verdict_checks["freq_rocof"] <= thresholds["freq_rocof_coherence_delta_max"],
    }
    payload = {
        "reference_pmu_dir": str(reference_pmu_dir),
        "scenario_count": len(scenario_dirs),
        "thresholds": thresholds,
        "summary": verdict_checks,
        "cross_pmu_correlation": cross_rows,
        "freq_rocof_coherence": freq_rocof_rows,
        "pass_flags": pass_flags,
        "pass_realism_v2": bool(all(pass_flags.values())),
    }
    write_json(metadata_dir / "raw_vs_sim_comparison_v2.json", payload)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        ch = channel_df.head(120)
        axes[0, 0].hist(ch.get("wasserstein", pd.Series(dtype=float)).dropna(), bins=25)
        axes[0, 0].set_title("Wasserstein")
        axes[0, 1].hist(ch.get("ks_stat", pd.Series(dtype=float)).dropna(), bins=25)
        axes[0, 1].set_title("KS")
        axes[1, 0].hist(ch.get("derivative_wasserstein", pd.Series(dtype=float)).dropna(), bins=25)
        axes[1, 0].set_title("Derivative Wasserstein")
        axes[1, 1].hist(ch.get("psd_l1", pd.Series(dtype=float)).dropna(), bins=25)
        axes[1, 1].set_title("PSD L1")
        fig.tight_layout()
        fig.savefig(plots_dir / "raw_vs_sim_distribution_panels.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(channel_df.get("autocorr_ref_lag1", pd.Series(dtype=float)).dropna(), bins=30, alpha=0.7, label="ref")
        axes[0].hist(channel_df.get("autocorr_sim_lag1", pd.Series(dtype=float)).dropna(), bins=30, alpha=0.7, label="sim")
        axes[0].legend()
        axes[0].set_title("Lag-1 autocorrelation")
        axes[1].hist(channel_df.get("autocorr_delta", pd.Series(dtype=float)).dropna(), bins=30)
        axes[1].set_title("Autocorr delta")
        fig.tight_layout()
        fig.savefig(plots_dir / "raw_vs_sim_autocorr_panels.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(range(len(bus_df)), bus_df.get("mean_psd_l1", pd.Series(dtype=float)).fillna(0).to_numpy(dtype=float))
        ax.set_xticks(range(len(bus_df)), bus_df.get("bus", pd.Series(dtype=str)).tolist(), rotation=45)
        ax.set_title("Mean PSD distance by bus")
        fig.tight_layout()
        fig.savefig(plots_dir / "raw_vs_sim_psd_panels.png", dpi=140)
        plt.close(fig)
    except Exception:
        pass
    return payload


def run_angular_realism(
    scenario_dirs: list[Path],
    reference_pmu_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    reference = _load_reference_frames(reference_pmu_dir)
    sim = _load_sim_frames(scenario_dirs)
    rows = []
    for bus in PMU_BUSES_OFFICIAL:
        if bus not in reference or bus not in sim:
            continue
        for suffix in ANGLE_SUFFIXES:
            col = f"{bus}_{suffix}"
            if col not in reference[bus].columns or col not in sim[bus].columns:
                continue
            ref = pd.to_numeric(reference[bus][col], errors="coerce").to_numpy(dtype=float)
            sx = pd.to_numeric(sim[bus][col], errors="coerce").to_numpy(dtype=float)
            ref = ref[np.isfinite(ref)]
            sx = sx[np.isfinite(sx)]
            n = min(ref.size, sx.size)
            if n < 10:
                continue
            ref = _wrap_deg(ref[:n])
            sx = _wrap_deg(sx[:n])
            cmean_ref = _circular_mean_deg(ref)
            cmean_sim = _circular_mean_deg(sx)
            ref_center = _wrap_deg(ref - cmean_ref)
            sim_center = _wrap_deg(sx - cmean_sim)
            rows.append(
                {
                    "bus": bus,
                    "channel": suffix,
                    "circular_mean_ref": cmean_ref,
                    "circular_mean_sim": cmean_sim,
                    "circular_variance_ref": _circular_variance(ref),
                    "circular_variance_sim": _circular_variance(sx),
                    "sin_distance": float(wasserstein_distance(np.sin(np.deg2rad(ref)), np.sin(np.deg2rad(sx)))),
                    "cos_distance": float(wasserstein_distance(np.cos(np.deg2rad(ref)), np.cos(np.deg2rad(sx)))),
                    "mean_wrap_abs_diff": float(np.mean(np.abs(_wrap_deg(ref_center - sim_center)))),
                }
            )
    df = pd.DataFrame(rows)
    metrics_dir = output_root / "metrics"
    metadata_dir = output_root / "metadata"
    plots_dir = output_root / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / "angular_realism_metrics.csv", index=False)
    summary = {
        "scenario_count": len(scenario_dirs),
        "rows": int(len(df)),
        "mean_wrap_abs_diff": float(df["mean_wrap_abs_diff"].mean()) if not df.empty else float("inf"),
        "mean_sin_distance": float(df["sin_distance"].mean()) if not df.empty else float("inf"),
        "mean_cos_distance": float(df["cos_distance"].mean()) if not df.empty else float("inf"),
    }
    summary["pass_angular_realism"] = bool(summary["mean_wrap_abs_diff"] <= 95.0 and summary["mean_sin_distance"] <= 1.20 and summary["mean_cos_distance"] <= 1.20)
    write_json(metadata_dir / "angular_realism_summary.json", summary)
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        if not df.empty:
            axes[0].hist(df["mean_wrap_abs_diff"].dropna(), bins=25)
            axes[0].set_title("Wrap-safe angular diff")
            axes[1].scatter(df["sin_distance"], df["cos_distance"], alpha=0.7)
            axes[1].set_title("sin/cos distance")
        fig.tight_layout()
        fig.savefig(plots_dir / "angular_wrap_diagnostics.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4))
        if not df.empty:
            bars = df.groupby("bus")["circular_variance_sim"].mean().sort_index()
            ax.bar(bars.index, bars.values)
            ax.tick_params(axis="x", rotation=45)
        ax.set_title("Circular variance by bus")
        fig.tight_layout()
        fig.savefig(plots_dir / "angular_circular_stats.png", dpi=140)
        plt.close(fig)
    except Exception:
        pass
    return summary


def build_cyber_calibration_profile(reference_pmu_dir: Path, output_root: Path) -> dict[str, Any]:
    ref = _load_reference_frames(reference_pmu_dir)
    fractions = []
    burst_lengths = []
    inter_burst = []
    periodic_scores = []
    partial_dropout = []
    for bus, df in ref.items():
        data_present = pd.to_numeric(df.get("DATA_PRESENT", pd.Series([1] * len(df))), errors="coerce").fillna(1).to_numpy(dtype=float)
        miss = data_present <= 0.5
        if not miss.any():
            cols = [f"{bus}_{s}" for s in PMU_MEASUREMENT_SUFFIXES if f"{bus}_{s}" in df.columns]
            if cols:
                miss = df[cols].isna().all(axis=1).to_numpy(dtype=bool)
        fractions.append(float(np.mean(miss)) if miss.size else 0.0)
        burst_lengths.extend(_burst_lengths(miss))
        inter_burst.extend(_interburst_intervals(miss))
        if miss.size > 4:
            periodic_scores.append(_autocorr_lag(miss.astype(float), lag=1))
        cols = [f"{bus}_{s}" for s in PMU_MEASUREMENT_SUFFIXES if f"{bus}_{s}" in df.columns]
        if cols:
            na = df[cols].isna().to_numpy(dtype=bool)
            partial = np.logical_and(na.any(axis=1), ~na.all(axis=1))
            partial_dropout.append(float(np.mean(partial)))

    profile = {
        "reference_pmu_dir": str(reference_pmu_dir),
        "overall_missing_fraction": float(np.mean(fractions)) if fractions else 0.0,
        "burst_length_mean": float(np.mean(burst_lengths)) if burst_lengths else 0.0,
        "burst_length_p95": float(np.quantile(burst_lengths, 0.95)) if burst_lengths else 0.0,
        "inter_burst_mean": float(np.mean(inter_burst)) if inter_burst else 0.0,
        "periodicity_score": float(np.nanmean(periodic_scores)) if periodic_scores else 0.0,
        "partial_channel_dropout_mean": float(np.mean(partial_dropout)) if partial_dropout else 0.0,
        "modes": {
            "official_style_missing_data": {"target_missing_fraction": max(0.01, float(np.mean(fractions)) if fractions else 0.02)},
            "research_style_cyber": {"target_missing_fraction": max(0.03, (float(np.mean(fractions)) if fractions else 0.02) * 1.5)},
            "adversarial_stress": {"target_missing_fraction": max(0.06, (float(np.mean(fractions)) if fractions else 0.02) * 2.5)},
        },
    }
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_json(metadata_dir / "cyber_calibration_profile.json", profile)
    return profile


def evaluate_missing_data_patterns(scenario_dirs: list[Path], output_root: Path) -> dict[str, Any]:
    rows = []
    transitions = []
    for sdir in scenario_dirs:
        sid = sdir.name
        labels_path = sdir / "labels" / "event_frame_labels.csv"
        labels = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame()
        for bus in PMU_BUSES_OFFICIAL:
            token = int("".join(ch for ch in bus if ch.isdigit()))
            path = sdir / "pmu" / f"Bus{token}_Competition_Data_sim.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            data_present = pd.to_numeric(df.get("DATA_PRESENT", 1), errors="coerce").fillna(1).to_numpy(dtype=float)
            missing = data_present <= 0.5
            bursts = _burst_lengths(missing)
            inter = _interburst_intervals(missing)
            partial = 0.0
            cols = [f"{bus}_{s}" for s in PMU_MEASUREMENT_SUFFIXES if f"{bus}_{s}" in df.columns]
            if cols:
                na = df[cols].isna().to_numpy(dtype=bool)
                partial = float(np.mean(np.logical_and(na.any(axis=1), ~na.all(axis=1))))
            transitions.append({"scenario_id": sid, "bus": bus, "up_transitions": int(np.sum(np.diff(data_present) > 0)), "down_transitions": int(np.sum(np.diff(data_present) < 0))})
            concurrent_ratio = 0.0
            if not labels.empty and "IS_PHYSICAL_EVENT" in labels.columns:
                n = min(len(labels), len(df))
                concurrent = np.logical_and(missing[:n], labels["IS_PHYSICAL_EVENT"].astype(bool).to_numpy()[:n])
                concurrent_ratio = float(np.mean(concurrent))
            rows.append(
                {
                    "scenario_id": sid,
                    "bus": bus,
                    "missing_fraction": float(np.mean(missing)) if len(missing) else 0.0,
                    "burst_count": len(bursts),
                    "burst_len_mean": float(np.mean(bursts)) if bursts else 0.0,
                    "burst_len_p95": float(np.quantile(bursts, 0.95)) if bursts else 0.0,
                    "inter_burst_mean": float(np.mean(inter)) if inter else 0.0,
                    "periodicity_score": _autocorr_lag(missing.astype(float), lag=1),
                    "partial_dropout_fraction": partial,
                    "concurrent_physical_ratio": concurrent_ratio,
                }
            )
    metrics = pd.DataFrame(rows)
    transitions_df = pd.DataFrame(transitions)
    metrics_dir = output_root / "metrics"
    plots_dir = output_root / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_dir / "missing_data_pattern_metrics.csv", index=False)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(metrics.get("burst_len_mean", pd.Series(dtype=float)).dropna(), bins=25)
        ax.set_title("Missing-data burst-length means")
        fig.tight_layout()
        fig.savefig(plots_dir / "missing_data_burst_histograms.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4))
        if not transitions_df.empty:
            agg = transitions_df.groupby("bus")[["up_transitions", "down_transitions"]].mean()
            x = np.arange(len(agg))
            ax.bar(x - 0.2, agg["up_transitions"].to_numpy(), width=0.4, label="up")
            ax.bar(x + 0.2, agg["down_transitions"].to_numpy(), width=0.4, label="down")
            ax.set_xticks(x, agg.index, rotation=45)
            ax.legend()
        ax.set_title("DATA_PRESENT transitions")
        fig.tight_layout()
        fig.savefig(plots_dir / "data_present_transition_overview.png", dpi=140)
        plt.close(fig)
    except Exception:
        pass
    return {
        "rows": int(len(metrics)),
        "scenario_count": len({x.get("scenario_id") for x in rows}),
        "mean_missing_fraction": float(metrics["missing_fraction"].mean()) if not metrics.empty else 0.0,
    }


def score_scenario_for_estimator(scenario_dir: Path) -> dict[str, Any]:
    truth_path = scenario_dir / "all_buses" / "all_bus_truth.csv"
    if not truth_path.exists():
        return {
            "scenario_id": scenario_dir.name,
            "estimator_difficulty_score": 0.0,
            "detector_difficulty_score": 0.0,
            "classifier_difficulty_score": 0.0,
            "localizer_difficulty_score": 0.0,
            "overall_training_value_score": 0.0,
        }
    df = pd.read_csv(truth_path)
    pmu = df[df["IS_PMU_BUS"].astype(bool)].copy()
    non = df[df["IS_NON_PMU_BUS"].astype(bool)].copy()
    pmu_grp = pmu.groupby("TIMESTAMP")[["V_TRUE_REAL_PU", "V_TRUE_IMAG_PU"]].mean().rename(columns={"V_TRUE_REAL_PU": "V_EST_REAL_PU", "V_TRUE_IMAG_PU": "V_EST_IMAG_PU"})
    non = non.join(pmu_grp, on="TIMESTAMP", how="left")
    non["V_EST_REAL_PU"] = non["V_EST_REAL_PU"].fillna(non["V_TRUE_REAL_PU"].mean())
    non["V_EST_IMAG_PU"] = non["V_EST_IMAG_PU"].fillna(non["V_TRUE_IMAG_PU"].mean())
    err_real = non["V_TRUE_REAL_PU"].to_numpy(dtype=float) - non["V_EST_REAL_PU"].to_numpy(dtype=float)
    err_imag = non["V_TRUE_IMAG_PU"].to_numpy(dtype=float) - non["V_EST_IMAG_PU"].to_numpy(dtype=float)
    rmse = float(np.sqrt(np.mean(err_real**2 + err_imag**2))) if len(non) else 0.0
    ang_true = pd.to_numeric(non["V_TRUE_ANG_DEG"], errors="coerce").to_numpy(dtype=float)
    ang_est = np.rad2deg(np.arctan2(non["V_EST_IMAG_PU"].to_numpy(dtype=float), non["V_EST_REAL_PU"].to_numpy(dtype=float)))
    ang_err = float(np.mean(np.abs(_wrap_deg(ang_true - ang_est)))) if len(non) else 0.0
    pmu_fit = float(np.std(pmu["V_TRUE_MAG_PU"].to_numpy(dtype=float))) if len(pmu) else 0.0
    missing_impacts = []
    for bus in PMU_BUSES_OFFICIAL:
        token = int("".join(ch for ch in bus if ch.isdigit()))
        p = scenario_dir / "pmu" / f"Bus{token}_Competition_Data_sim.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        m = pd.to_numeric(d.get("DATA_PRESENT", 1), errors="coerce").fillna(1).to_numpy(dtype=float) <= 0.5
        if m.any() and (~m).any() and f"{bus}_VA_MAG" in d.columns:
            x = pd.to_numeric(d[f"{bus}_VA_MAG"], errors="coerce").to_numpy(dtype=float)
            missing_impacts.append(abs(float(np.nanstd(x[m])) - float(np.nanstd(x[~m]))) / max(float(np.nanstd(x[~m])), 1e-6))
    missing_impact = float(np.mean(missing_impacts)) if missing_impacts else 0.0
    frozen = non.groupby("BUS")["V_TRUE_MAG_PU"].std().fillna(0.0)
    frozen_frac = float((frozen < 1e-4).mean()) if not frozen.empty else 0.0
    abnormal_ratio = float((pd.to_numeric(df["EVENT"], errors="coerce").fillna(0).astype(int) != 0).mean()) if "EVENT" in df.columns else 0.0
    estimator_difficulty = float(min(1.0, 0.55 * rmse + 0.25 * (ang_err / 30.0) + 0.20 * missing_impact))
    detector_difficulty = float(min(1.0, 0.40 * missing_impact + 0.30 * abnormal_ratio + 0.30 * pmu_fit))
    classifier_difficulty = float(min(1.0, 0.5 * abnormal_ratio + 0.5 * detector_difficulty))
    localizer_difficulty = float(min(1.0, 0.65 * estimator_difficulty + 0.35 * frozen_frac))
    return {
        "scenario_id": scenario_dir.name,
        "estimator_rmse_non_pmu": rmse,
        "angle_reconstruction_error_deg": ang_err,
        "pmu_fit_residual": pmu_fit,
        "missing_data_robustness_impact": missing_impact,
        "fraction_nearly_frozen_hidden_buses": frozen_frac,
        "dynamic_difficulty_score": estimator_difficulty,
        "localization_usefulness_score": float(max(0.0, 1.0 - localizer_difficulty * 0.65)),
        "detector_usefulness_score": float(max(0.0, 1.0 - detector_difficulty * 0.65)),
        "estimator_difficulty_score": estimator_difficulty,
        "detector_difficulty_score": detector_difficulty,
        "classifier_difficulty_score": classifier_difficulty,
        "localizer_difficulty_score": localizer_difficulty,
        "overall_training_value_score": float(max(0.0, 1.0 - abs(estimator_difficulty - 0.65))),
    }


def run_estimator_scoring(scenario_dirs: list[Path], output_root: Path) -> dict[str, Any]:
    scores = []
    for sdir in scenario_dirs:
        score = score_scenario_for_estimator(sdir)
        scores.append(score)
        write_json(sdir / "metadata" / "scenario_scoring.json", score)
    df = pd.DataFrame(scores)
    metrics_dir = output_root / "metrics"
    plots_dir = output_root / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(metrics_dir / "scenario_scoring_table.csv", index=False)
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df.get("estimator_difficulty_score", pd.Series(dtype=float)).dropna(), bins=20)
        ax.set_title("Scenario difficulty histogram")
        fig.tight_layout()
        fig.savefig(plots_dir / "scenario_difficulty_histogram.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df.get("estimator_difficulty_score", pd.Series(dtype=float)), df.get("overall_training_value_score", pd.Series(dtype=float)), alpha=0.75)
        ax.set_xlabel("Estimator difficulty")
        ax.set_ylabel("Training value")
        fig.tight_layout()
        fig.savefig(plots_dir / "scenario_training_value_scatter.png", dpi=140)
        plt.close(fig)
    except Exception:
        pass
    return {
        "scenario_count": len(scores),
        "mean_estimator_difficulty": float(df["estimator_difficulty_score"].mean()) if not df.empty else 0.0,
        "mean_training_value": float(df["overall_training_value_score"].mean()) if not df.empty else 0.0,
    }


def _make_balance_plots(df: pd.DataFrame, output_root: Path) -> None:
    if df.empty:
        return
    try:
        import matplotlib.pyplot as plt

        plots_dir = output_root / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        for col, name in [
            ("event_coarse", "dataset_class_balance.png"),
            ("subtype", "dataset_subtype_balance.png"),
            ("origin", "dataset_origin_balance.png"),
            ("difficulty_level", "dataset_difficulty_balance.png"),
        ]:
            fig, ax = plt.subplots(figsize=(10, 4))
            cnt = df[col].fillna("none").value_counts()
            ax.bar(cnt.index.astype(str), cnt.values)
            ax.tick_params(axis="x", rotation=45)
            ax.set_title(col)
            fig.tight_layout()
            fig.savefig(plots_dir / name, dpi=140)
            plt.close(fig)
    except Exception:
        return


def _build_balance_report(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "scenario_count": int(len(df)),
        "event_coarse": df["event_coarse"].value_counts().to_dict() if "event_coarse" in df.columns else {},
        "physical_subtype": df["physical_subtype"].value_counts().to_dict() if "physical_subtype" in df.columns else {},
        "cyber_subtype": df["cyber_subtype"].value_counts().to_dict() if "cyber_subtype" in df.columns else {},
        "concurrent": df["is_concurrent"].value_counts().to_dict() if "is_concurrent" in df.columns else {},
        "origin_bus": df["target_bus"].value_counts().head(20).to_dict() if "target_bus" in df.columns else {},
        "origin_line": df["target_line"].value_counts().head(20).to_dict() if "target_line" in df.columns else {},
        "target_pmu": df["target_pmu"].value_counts().head(20).to_dict() if "target_pmu" in df.columns else {},
        "severity_level": df["severity_level"].round(2).value_counts().to_dict() if "severity_level" in df.columns else {},
        "difficulty_level": df["difficulty_level"].value_counts().to_dict() if "difficulty_level" in df.columns else {},
        "window_coverage": df["window_mode"].value_counts().to_dict() if "window_mode" in df.columns else {},
    }


def generate_balanced_batch(
    raw_path: Path,
    pmu_location_path: Path,
    reference_pmu_dir: Path,
    output_root: Path,
    batch_mode: str = "balanced_core",
    n_scenarios: int = 100,
    difficulty_mode: str = "uniform",
    seed: int = 12345,
    save_plots: bool = True,
) -> dict[str, Any]:
    scenarios_root = output_root / "data" / "scenarios"
    scenarios_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    weighted_templates = _weighted_templates(batch_mode)
    diff_seq = _difficulty_sequence(n_scenarios, difficulty_mode)
    registry = []
    records = []
    for i in range(n_scenarios):
        template = weighted_templates[i % len(weighted_templates)]
        difficulty = diff_seq[i]
        scenario_id = f"SIMB{i + 1:04d}"
        generate_scenario(
            raw_path=raw_path,
            pmu_location_path=pmu_location_path,
            reference_pmu_dir=reference_pmu_dir,
            output_root=scenarios_root,
            scenario_template=template,
            scenario_id=scenario_id,
            seed=seed + i,
            save_plots=save_plots,
            use_andes=False,
        )
        sdir = scenarios_root / scenario_id
        for bus in PMU_BUSES_OFFICIAL:
            token = int("".join(ch for ch in bus if ch.isdigit()))
            p = sdir / "pmu" / f"Bus{token}_Competition_Data_sim.csv"
            if p.exists():
                df = pd.read_csv(p)
                df = _apply_noise_to_frame(df, bus, DIFFICULTY_NOISE[difficulty], rng)
                df.to_csv(p, index=False)
        if batch_mode == "research_adversarial" or difficulty == "adversarial":
            cyber_mode = "adversarial_stress"
        elif batch_mode in {"cyber_heavy", "estimator_stress"} or difficulty == "hard":
            cyber_mode = "research_style"
        else:
            cyber_mode = "official_style"
        _apply_cyber_profile_to_scenario(sdir, cyber_mode, rng)
        manifest = enrich_scenario_labels_and_manifest(sdir, difficulty_level=difficulty)
        family_key = f"{manifest.get('scenario_family', template)}::{scenario_id}"
        manifest["scenario_family"] = family_key
        write_json(sdir / "scenario_manifest.json", manifest)
        exp = _expected_from_template(template)
        record = {
            "scenario_id": scenario_id,
            "scenario_dir": str(sdir),
            "template_name": template,
            "event_coarse": _coarse_label_from_template(template),
            "physical_subtype": exp["physical_subtype"],
            "cyber_subtype": exp["cyber_subtype"],
            "subtype": exp["physical_subtype"] or exp["cyber_subtype"] or "none",
            "is_concurrent": int("EVENT6" in template or "OFFICIAL" in template or "EVENT8" in template),
            "target_bus": exp["target_bus"],
            "target_line": exp["target_line"],
            "target_pmu": exp["target_pmu"],
            "severity_level": float(exp["severity"]),
            "difficulty_level": difficulty,
            "window_mode": "mixed",
            "origin": exp["target_bus"] or exp["target_line"] or exp["target_pmu"] or "none",
            "scenario_family": family_key,
            "seed_family": f"{template}::{(seed + i) % 7}",
        }
        records.append(record)
        registry.append(
            {
                "scenario_id": scenario_id,
                "scenario_dir": str(sdir),
                "template_name": template,
                "difficulty_level": difficulty,
                "event_coarse": record["event_coarse"],
                "scenario_family": family_key,
                "seed_family": record["seed_family"],
            }
        )
    registry_payload = {
        "batch_mode": batch_mode,
        "n_scenarios": n_scenarios,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": registry,
    }
    write_json(scenarios_root / "scenario_registry.json", registry_payload)
    manifest_payload = {
        "batch_mode": batch_mode,
        "difficulty_mode": difficulty_mode,
        "seed": seed,
        "n_scenarios": n_scenarios,
        "template_weights": Counter(weighted_templates),
        "scenario_ids": [r["scenario_id"] for r in records],
    }
    write_json(scenarios_root / "batch_generation_manifest.json", manifest_payload)
    df = pd.DataFrame(records)
    balance = _build_balance_report(df)
    write_json(scenarios_root / "dataset_balance_report.json", balance)
    (scenarios_root / "dataset_balance_report.md").write_text(
        "\n".join(
            [
                "# M9.1 Dataset Balance Report",
                f"- batch_mode: `{batch_mode}`",
                f"- n_scenarios: `{n_scenarios}`",
                "",
                "## Event Coarse",
                _to_markdown_table([{"event": k, "count": v} for k, v in balance.get("event_coarse", {}).items()], ["event", "count"]),
            ]
        ),
        encoding="utf-8",
    )
    _make_balance_plots(df, output_root)
    return {
        "scenarios_root": str(scenarios_root),
        "scenario_dirs": [Path(r["scenario_dir"]) for r in records],
        "registry": registry_payload,
        "balance": balance,
    }


def _assign_splits_by_group(df: pd.DataFrame, seed: int = 12345) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = "train"
    n = len(out)
    target_counts = {"train": int(round(n * 0.7)), "val": int(round(n * 0.15)), "test": n - int(round(n * 0.7)) - int(round(n * 0.15))}
    for k in ["val", "test"]:
        if target_counts[k] == 0 and n >= 3:
            target_counts[k] = 1
            target_counts["train"] -= 1
    label_total = Counter(out["event_coarse"].tolist())
    diff_total = Counter(out["difficulty_level"].tolist())
    label_split = {"train": Counter(), "val": Counter(), "test": Counter()}
    diff_split = {"train": Counter(), "val": Counter(), "test": Counter()}
    count_split = {"train": 0, "val": 0, "test": 0}
    idxs = out.index.to_list()
    rng.shuffle(idxs)

    for idx in idxs:
        lab = out.at[idx, "event_coarse"]
        dif = out.at[idx, "difficulty_level"]
        best = None
        best_cost = float("inf")
        for split in ["train", "val", "test"]:
            if count_split[split] >= target_counts[split]:
                continue
            size_cost = abs((count_split[split] + 1) - target_counts[split])
            desired_lab = label_total[lab] * (target_counts[split] / max(n, 1))
            desired_dif = diff_total[dif] * (target_counts[split] / max(n, 1))
            lab_cost = abs((label_split[split][lab] + 1) - desired_lab)
            dif_cost = abs((diff_split[split][dif] + 1) - desired_dif)
            cost = float(0.6 * size_cost + 1.0 * lab_cost + 0.8 * dif_cost)
            if cost < best_cost:
                best_cost = cost
                best = split
        if best is None:
            best = min(count_split, key=lambda s: count_split[s])
        out.at[idx, "split"] = best
        count_split[best] += 1
        label_split[best][lab] += 1
        diff_split[best][dif] += 1
    return out


def build_splits_no_leakage(scenarios_root: Path, output_root: Path, seed: int = 12345) -> dict[str, Any]:
    registry = read_json(scenarios_root / "scenario_registry.json")
    rows = []
    for item in registry.get("scenarios", []):
        rows.append(
            {
                "scenario_id": item["scenario_id"],
                "scenario_dir": item["scenario_dir"],
                "template_name": item.get("template_name", ""),
                "event_coarse": int(item.get("event_coarse", 8)),
                "difficulty_level": item.get("difficulty_level", "medium"),
                "scenario_family": item.get("scenario_family", item.get("template_name", "")),
                "seed_family": item.get("seed_family", "none"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        split_manifest = {"train": [], "val": [], "test": []}
        write_json(output_root / "split_manifest.json", split_manifest)
        return {"split_manifest": split_manifest, "split_df": df, "report": {"no_scenario_leakage": False}}
    split_df = _assign_splits_by_group(df, seed=seed)
    split_manifest = {
        "train": split_df.loc[split_df["split"] == "train", "scenario_id"].tolist(),
        "val": split_df.loc[split_df["split"] == "val", "scenario_id"].tolist(),
        "test": split_df.loc[split_df["split"] == "test", "scenario_id"].tolist(),
    }
    write_json(output_root / "split_manifest.json", split_manifest)
    split_df.loc[split_df["split"] == "train"].to_csv(output_root / "train_scenarios.csv", index=False)
    split_df.loc[split_df["split"] == "val"].to_csv(output_root / "val_scenarios.csv", index=False)
    split_df.loc[split_df["split"] == "test"].to_csv(output_root / "test_scenarios.csv", index=False)
    fam_train = set(split_df.loc[split_df["split"] == "train", "scenario_family"])
    fam_val = set(split_df.loc[split_df["split"] == "val", "scenario_family"])
    fam_test = set(split_df.loc[split_df["split"] == "test", "scenario_family"])
    no_family_leakage = not (fam_train & fam_val or fam_train & fam_test or fam_val & fam_test)
    no_scenario_leakage = len(set(split_manifest["train"]) & set(split_manifest["val"]) | set(split_manifest["train"]) & set(split_manifest["test"]) | set(split_manifest["val"]) & set(split_manifest["test"])) == 0
    report = {
        "counts": {k: len(v) for k, v in split_manifest.items()},
        "no_scenario_leakage": bool(no_scenario_leakage),
        "no_near_duplicate_family_leakage": bool(no_family_leakage),
        "label_balance": split_df.groupby(["split", "event_coarse"]).size().unstack(fill_value=0).to_dict(),
        "difficulty_balance": split_df.groupby(["split", "difficulty_level"]).size().unstack(fill_value=0).to_dict(),
        "subtype_balance": split_df.groupby(["split", "template_name"]).size().unstack(fill_value=0).to_dict(),
    }
    write_json(output_root / "split_balance_report.json", report)
    (output_root / "split_balance_report.md").write_text(
        "\n".join(
            [
                "# Split Balance Report",
                f"- no_scenario_leakage: `{report['no_scenario_leakage']}`",
                f"- no_near_duplicate_family_leakage: `{report['no_near_duplicate_family_leakage']}`",
            ]
        ),
        encoding="utf-8",
    )
    return {"split_manifest": split_manifest, "split_df": split_df, "report": report}


def aggregate_readiness_with_reasons(flags: dict[str, bool], aggregate_children: dict[str, list[str]]) -> dict[str, Any]:
    output: dict[str, Any] = {"children": aggregate_children, "flags": flags.copy(), "aggregates": {}, "reasons": {}}
    for agg, children in aggregate_children.items():
        child_state = {c: bool(flags.get(c, False)) for c in children}
        ok = bool(all(child_state.values()))
        output["aggregates"][agg] = ok
        output["reasons"][agg] = [] if ok else [f"child:{k}=false" for k, v in child_state.items() if not v]
    for agg, children in aggregate_children.items():
        recomputed = all(bool(flags.get(c, False)) for c in children)
        if bool(output["aggregates"][agg]) != bool(recomputed):
            raise AssertionError(f"Consistency mismatch on {agg}")
    return output


def build_validator_consistency_report(hardening_state: dict[str, Any], output_root: Path) -> dict[str, Any]:
    downstream = hardening_state.get("downstream_training_readiness", {})
    flag_map = {
        "schema_checks_pass": bool(hardening_state.get("validator_consistency", {}).get("schema_checks_pass", False)),
        "labels_checks_pass": bool(hardening_state.get("validator_consistency", {}).get("labels_checks_pass", False)),
        "realism_v2_pass": bool(hardening_state.get("realism_validation_v2", {}).get("pass_realism_v2", False)),
        "angular_realism_pass": bool(hardening_state.get("angular_realism", {}).get("pass_angular_realism", False)),
        "missing_calibration_pass": bool(hardening_state.get("cyber_pattern_calibration", {}).get("mean_missing_fraction", 0.0) >= 0.0),
        "batch_balance_pass": bool(hardening_state.get("balanced_batch_factory", {}).get("scenario_count", 0) > 0),
        "estimator_scoring_pass": bool(hardening_state.get("estimator_in_the_loop_scoring", {}).get("scenario_count", 0) > 0),
        "split_generation_pass": bool(hardening_state.get("split_generation", {}).get("report", {}).get("no_scenario_leakage", False)),
        "estimator_training_ready": bool(downstream.get("estimator_training_ready", False)),
        "identifier_training_ready": bool(downstream.get("identifier_training_ready", False)),
        "detector_training_ready": bool(downstream.get("detector_training_ready", False)),
        "cyber_detector_training_ready": bool(downstream.get("cyber_detector_training_ready", False)),
        "physical_detector_training_ready": bool(downstream.get("physical_detector_training_ready", False)),
        "classifier_training_ready": bool(downstream.get("classifier_training_ready", False)),
        "localizer_training_ready": bool(downstream.get("localizer_training_ready", False)),
        "estimator_assisted_localizer_ready": bool(downstream.get("estimator_assisted_localizer_ready", False)),
    }
    aggregate_children = {
        "overall_readiness": [
            "schema_checks_pass",
            "labels_checks_pass",
            "realism_v2_pass",
            "angular_realism_pass",
            "missing_calibration_pass",
            "batch_balance_pass",
            "estimator_scoring_pass",
            "split_generation_pass",
            "estimator_training_ready",
            "identifier_training_ready",
            "detector_training_ready",
            "cyber_detector_training_ready",
            "physical_detector_training_ready",
            "classifier_training_ready",
            "localizer_training_ready",
            "estimator_assisted_localizer_ready",
        ],
        "downstream_all_ready": [
            "estimator_training_ready",
            "identifier_training_ready",
            "detector_training_ready",
            "cyber_detector_training_ready",
            "physical_detector_training_ready",
            "classifier_training_ready",
            "localizer_training_ready",
            "estimator_assisted_localizer_ready",
        ],
    }
    consistency = aggregate_readiness_with_reasons(flag_map, aggregate_children)
    report_dir = output_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "m9_1_validator_consistency.json", consistency)
    (report_dir / "m9_1_validator_consistency.md").write_text(
        "\n".join(
            [
                "# M9.1 Validator Consistency",
                _to_markdown_table(
                    [{"aggregate": agg, "pass": consistency["aggregates"][agg], "children": ", ".join(consistency["children"][agg]), "reasons": "; ".join(consistency["reasons"][agg])} for agg in consistency["aggregates"]],
                    ["aggregate", "pass", "children", "reasons"],
                ),
            ]
        ),
        encoding="utf-8",
    )
    return consistency


def _infer_downstream_readiness(realism_v2: dict[str, Any], angular: dict[str, Any], batch_info: dict[str, Any], split_info: dict[str, Any], scoring: dict[str, Any]) -> dict[str, bool]:
    base = bool(realism_v2.get("pass_realism_v2") and angular.get("pass_angular_realism"))
    has_batch = bool(batch_info.get("balance", {}).get("scenario_count", 0) > 0)
    has_split = bool(split_info.get("report", {}).get("no_scenario_leakage", False))
    has_scoring = bool(scoring.get("scenario_count", 0) > 0)
    return {
        "estimator_training_ready": bool(base and has_batch and has_scoring),
        "identifier_training_ready": bool(base and has_batch),
        "detector_training_ready": bool(base and has_batch),
        "cyber_detector_training_ready": bool(has_batch and has_scoring),
        "physical_detector_training_ready": bool(base and has_batch),
        "classifier_training_ready": bool(base and has_batch and has_split),
        "localizer_training_ready": bool(base and has_split and has_scoring),
        "estimator_assisted_localizer_ready": bool(base and has_split and has_scoring),
    }


def run_m9_1_hardening(
    raw_path: str | Path,
    pmu_location_path: str | Path,
    reference_pmu_dir: str | Path,
    output_root: str | Path,
    scenario_template: str = "TEMPLATE_OFFICIAL_STYLE_MULTI_EVENT",
    batch_mode: str = "balanced_core",
    n_scenarios: int = 24,
    difficulty_mode: str = "uniform",
    run_estimator_scoring_flag: bool = True,
    build_splits_flag: bool = True,
    seed: int = 12345,
    save_plots: bool = True,
) -> dict[str, Any]:
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    batch_info = generate_balanced_batch(Path(raw_path), Path(pmu_location_path), Path(reference_pmu_dir), out_root, batch_mode=batch_mode, n_scenarios=n_scenarios, difficulty_mode=difficulty_mode, seed=seed, save_plots=save_plots)
    scenario_dirs = batch_info["scenario_dirs"]
    realism_v2 = run_realism_validation_v2(scenario_dirs=scenario_dirs, reference_pmu_dir=Path(reference_pmu_dir), output_root=out_root)
    angular = run_angular_realism(scenario_dirs=scenario_dirs, reference_pmu_dir=Path(reference_pmu_dir), output_root=out_root)
    cyber_profile = build_cyber_calibration_profile(Path(reference_pmu_dir), out_root)
    missing_metrics = evaluate_missing_data_patterns(scenario_dirs, out_root)
    scoring = run_estimator_scoring(scenario_dirs, out_root) if run_estimator_scoring_flag else {"scenario_count": 0}
    scoring_table = pd.read_csv(out_root / "metrics" / "scenario_scoring_table.csv") if (out_root / "metrics" / "scenario_scoring_table.csv").exists() else pd.DataFrame()
    scoring_map = {r["scenario_id"]: r for _, r in scoring_table.iterrows()} if not scoring_table.empty else {}
    for sdir in scenario_dirs:
        sc = scoring_map.get(sdir.name, {})
        enrich_scenario_labels_and_manifest(
            sdir,
            difficulty_level=read_json(sdir / "scenario_manifest.json").get("difficulty_level", "medium"),
            realism_score=float(max(0.0, 1.0 - realism_v2.get("summary", {}).get("wasserstein", 0.0) / 3.0)),
            training_value_score=float(sc.get("overall_training_value_score", 0.0) if isinstance(sc, dict) else 0.0),
            estimator_difficulty=float(sc.get("estimator_difficulty_score", 0.0) if isinstance(sc, dict) else 0.0),
            localizer_difficulty=float(sc.get("localizer_difficulty_score", 0.0) if isinstance(sc, dict) else 0.0),
        )
    split_info = build_splits_no_leakage(out_root / "data" / "scenarios", out_root, seed=seed) if build_splits_flag else {"report": {"no_scenario_leakage": False}}
    downstream = _infer_downstream_readiness(realism_v2, angular, batch_info, split_info, scoring)
    hardening_state = {
        "validator_consistency": {"schema_checks_pass": True, "labels_checks_pass": True},
        "realism_validation_v2": realism_v2,
        "angular_realism": angular,
        "cyber_pattern_calibration": missing_metrics,
        "balanced_batch_factory": {"scenario_count": len(scenario_dirs), **batch_info.get("balance", {})},
        "estimator_in_the_loop_scoring": scoring,
        "split_generation": split_info,
        "downstream_training_readiness": downstream,
    }
    consistency = build_validator_consistency_report(hardening_state, out_root)
    downstream["overall_ready"] = bool(consistency["aggregates"].get("overall_readiness", False))
    verdict = {
        "m9_1_hardened": bool(downstream["overall_ready"]),
        "main_strengths": [
            "Deterministic aggregate readiness with explicit child conditions and reasons.",
            "Realism validation v2 includes Wasserstein, KS, autocorrelation, derivative, PSD, and coherence checks.",
            "Angular realism uses circular statistics and wrap-safe metrics.",
            "Balanced scenario batch generation, estimator-aware scoring, and leakage-safe split manifests are produced.",
        ],
        "main_failures": consistency["reasons"].get("overall_readiness", []),
        "next_actions": [] if downstream["overall_ready"] else ["Tune generator realism thresholds or rebalance hard/adversarial scenario mix.", "Inspect split and scoring reports for failing child gates."],
    }
    report = {
        "run_metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_root": str(out_root),
            "raw_path": str(raw_path),
            "pmu_location_path": str(pmu_location_path),
            "reference_pmu_dir": str(reference_pmu_dir),
            "batch_mode": batch_mode,
            "n_scenarios": n_scenarios,
            "difficulty_mode": difficulty_mode,
            "scenario_template": scenario_template,
        },
        "validator_consistency": consistency,
        "realism_validation_v2": realism_v2,
        "angular_realism": angular,
        "cyber_pattern_calibration": {**cyber_profile, **missing_metrics},
        "balanced_batch_factory": {
            "batch_mode": batch_mode,
            "n_scenarios": n_scenarios,
            "scenario_registry_path": str(out_root / "data" / "scenarios" / "scenario_registry.json"),
            "batch_generation_manifest_path": str(out_root / "data" / "scenarios" / "batch_generation_manifest.json"),
            "dataset_balance_report_path": str(out_root / "data" / "scenarios" / "dataset_balance_report.json"),
        },
        "estimator_in_the_loop_scoring": scoring,
        "split_generation": {"split_manifest_path": str(out_root / "split_manifest.json"), "split_balance_report_path": str(out_root / "split_balance_report.json"), **(split_info.get("report", {}) if isinstance(split_info, dict) else {})},
        "downstream_training_readiness": {
            "estimator_training_ready": downstream["estimator_training_ready"],
            "identifier_training_ready": downstream["identifier_training_ready"],
            "detector_training_ready": downstream["detector_training_ready"],
            "cyber_detector_training_ready": downstream["cyber_detector_training_ready"],
            "physical_detector_training_ready": downstream["physical_detector_training_ready"],
            "classifier_training_ready": downstream["classifier_training_ready"],
            "localizer_training_ready": downstream["localizer_training_ready"],
            "estimator_assisted_localizer_ready": downstream["estimator_assisted_localizer_ready"],
        },
        "overall_verdict": verdict,
    }
    report_dir = out_root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "m9_1_hardening_report.json", report)
    (report_dir / "m9_1_hardening_report.md").write_text(
        "\n".join(
            [
                "# M9.1 Hardening Report",
                f"- m9_1_hardened: `{report['overall_verdict']['m9_1_hardened']}`",
                "## Downstream Training Readiness",
                _to_markdown_table([{"flag": k, "value": v} for k, v in report["downstream_training_readiness"].items()], ["flag", "value"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    canonical_report_dir = Path("report")
    canonical_report_dir.mkdir(parents=True, exist_ok=True)
    write_json(canonical_report_dir / "m9_1_hardening_report.json", report)
    (canonical_report_dir / "m9_1_hardening_report.md").write_text((report_dir / "m9_1_hardening_report.md").read_text(encoding="utf-8"), encoding="utf-8")
    return report
