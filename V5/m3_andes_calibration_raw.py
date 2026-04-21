"""RAW Event-0 ANDES calibration in physical PMU units.

Scope is intentionally limited to RAW0001 normal operation (event label 0).
The script compares all raw event-0 chunks against one ANDES normal-operation
trajectory plus signal-space operating drift and raw PMU noise profiles.
"""

import glob
import json
import math
import os
import shutil
from datetime import datetime, timezone

import andes
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import ks_2samp, t as student_t_dist

from m2_noise_profiling_raw import PROFILE_OUT, RawPMUNoiseLayer

matplotlib.use("Agg")


PMU_BUSES = ["39", "29", "10", "22", "19", "2", "5", "6"]
RAW_CHUNKS_DIR = "output/SCENARIO_RAW0001/chunks"
RAW_PROFILE_FILE = PROFILE_OUT
OUTPUT_DIR = "output/ANDES_CALIBRATION_RAW"
EVENT_LABEL = 0
CALIBRATION_MODE = "both"  # per_chunk, aggregate, both
DRIFT_MODE = "fft_residual_synthesis"  # fft_residual_synthesis, ou_ar1_lowfreq, none
NOISE_MODEL = "auto"  # auto, gaussian, student_t, gmm, bootstrap
CALIBRATION_FIT_MODE = "affine_plus_quantile"  # affine, affine_plus_quantile
RNG_SEED = 20260417
TF = 60.0
TSTEP = 1.0 / 30.0
DRIFT_SCALE_GRID = [0.65, 0.90, 1.10]
NOISE_SCALE_GRID = [0.30, 0.70, 1.00]
NOISE_MODEL_CANDIDATES = ["gaussian", "student_t", "gmm", "bootstrap"]
SPECTRAL_SIGNAL_KEYS = {"Frequency", "IA_mag", "IB_mag", "IC_mag"}

ACTIVE_STATUSES = {"supported", "fallback_profile"}


def wrap_deg(x):
    return ((np.asarray(x, dtype=float) + 180.0) % 360.0) - 180.0


def quality_bucket(ks):
    if not np.isfinite(ks):
        return "invalid_mapping"
    if ks < 0.03:
        return "very_good"
    if ks < 0.08:
        return "usable"
    if ks < 0.15:
        return "needs_tuning"
    return "poor"


def signal_family_from_key(key):
    if key.endswith("_mag") and key[0] == "V":
        return "voltage_mag"
    if key.endswith("_mag") and key[0] == "I":
        return "current_mag"
    if key == "Frequency":
        return "frequency"
    if key == "ROCOF":
        return "frequency"
    if key.endswith("_ang_delta"):
        return "angle_voltage"
    if key.endswith("_ang"):
        return "angle_current"
    return "other"


def build_signal_specs():
    specs = []
    for ph in ["A", "B", "C"]:
        specs.append({
            "signal_key": f"V{ph}_mag",
            "raw_suffix": f"V{ph}_MAG",
            "sim_source": "voltage_mag",
            "support_status": "supported_direct",
            "raw_representation": "raw voltage magnitude in physical dataset units",
            "recommended": True,
            "notes": "ANDES positive-sequence Bus.v mapped to raw magnitude using event-0 calibration.",
        })
    for ph in ["A", "B", "C"]:
        specs.append({
            "signal_key": f"I{ph}_mag",
            "raw_suffix": f"I{ph}_MAG",
            "sim_source": "current_mag_selected",
            "support_status": "supported_derived",
            "raw_representation": "selected positive-sequence equivalent current magnitude in A",
            "recommended": False,
            "notes": "Mapping selected per bus from injection, incident branches, dominant branch, and generator candidates.",
        })
    specs.extend([
        {
            "signal_key": "Frequency",
            "raw_suffix": "Freq",
            "sim_source": "frequency",
            "support_status": "supported_derived",
            "raw_representation": "raw PMU frequency in Hz",
            "recommended": True,
            "notes": "Derived from ANDES bus-angle speed, then calibrated in raw units.",
        },
        {
            "signal_key": "ROCOF",
            "raw_suffix": "ROCOF",
            "sim_source": "rocof",
            "support_status": "supported_derived",
            "raw_representation": "raw PMU ROCOF",
            "recommended": False,
            "notes": "Derived from frequency gradient; often too noisy/flat in normal operation.",
        },
    ])
    for ph in ["A", "B", "C"]:
        specs.append({
            "signal_key": f"V{ph}_ang_delta",
            "raw_suffix": f"V{ph}_ANG",
            "sim_source": "voltage_angle_delta",
            "support_status": "experimental_relative_only",
            "raw_representation": "wrap(raw voltage angle - local event-0 median) in degrees",
            "recommended": False,
            "notes": "Raw absolute PMU angle is unsupported; only wrapped local delta is evaluated as experimental.",
        })
    for ph in ["A", "B", "C"]:
        specs.append({
            "signal_key": f"I{ph}_ang",
            "raw_suffix": f"I{ph}_ANG",
            "sim_source": None,
            "support_status": "unsupported_raw_absolute",
            "raw_representation": "raw current absolute angle",
            "recommended": False,
            "notes": "Current absolute angle has no robust physical mapping in this positive-sequence model.",
        })
    return specs


SIGNAL_SPECS = build_signal_specs()


def discover_event0_chunks():
    return sorted(glob.glob(os.path.join(RAW_CHUNKS_DIR, "*_event_0_*")))


def load_event0_chunks_for_bus(bus_id):
    chunks = []
    for chunk_dir in discover_event0_chunks():
        path = os.path.join(chunk_dir, f"Bus{bus_id}.csv")
        if os.path.exists(path):
            chunks.append({
                "chunk_id": os.path.basename(chunk_dir),
                "path": path,
                "df": pd.read_csv(path),
            })
    return chunks


def raw_col(bus_id, suffix):
    return f"BUS{bus_id}_{suffix}"


def raw_values_from_chunks(chunks, bus_id, suffix, representation="raw"):
    values = []
    col = raw_col(bus_id, suffix)
    for chunk in chunks:
        if col not in chunk["df"].columns:
            continue
        y = pd.to_numeric(chunk["df"][col], errors="coerce").dropna().to_numpy(float)
        if representation == "wrapped_delta" and len(y):
            y = wrap_deg(y - np.median(y))
        if len(y):
            values.append(y)
    return np.concatenate(values) if values else np.array([], dtype=float)


def autocorr(x, lag):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= lag or np.std(x) < 1e-15:
        return 0.0
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def sample_rate_from_t(t):
    t = np.asarray(t, dtype=float)
    if len(t) < 3:
        return 30.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def spectral_features(x, fs=30.0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or np.std(x) < 1e-15:
        return {
            "lowfreq_power": 0.0,
            "total_power": 0.0,
            "lowfreq_power_ratio": 0.0,
            "spectral_centroid_hz": 0.0,
            "dominant_frequency_hz": 0.0,
        }
    centered = x - np.mean(x)
    nperseg = min(512, len(centered))
    freqs, psd = welch(centered, fs=fs, nperseg=nperseg, detrend="constant")
    total = float(np.trapezoid(psd, freqs)) if len(freqs) > 1 else float(np.sum(psd))
    low_mask = (freqs > 0.0) & (freqs <= 0.25)
    low = float(np.trapezoid(psd[low_mask], freqs[low_mask])) if np.any(low_mask) else 0.0
    centroid = float(np.sum(freqs * psd) / max(np.sum(psd), 1e-30))
    fft_freq = np.fft.rfftfreq(len(centered), d=1.0 / fs)
    fft_mag = np.abs(np.fft.rfft(centered)) / max(len(centered), 1)
    dominant_idx = int(np.argmax(fft_mag[1:]) + 1) if len(fft_mag) > 1 else 0
    return {
        "lowfreq_power": low,
        "total_power": total,
        "lowfreq_power_ratio": float(low / max(total, 1e-30)),
        "spectral_centroid_hz": centroid,
        "dominant_frequency_hz": float(fft_freq[dominant_idx]) if dominant_idx else 0.0,
    }


def spectral_mismatch(real, sim, fs=30.0):
    r = spectral_features(real, fs)
    s = spectral_features(sim, fs)
    low_mismatch = abs(s["lowfreq_power"] - r["lowfreq_power"]) / max(r["lowfreq_power"], np.std(real) ** 2, 1e-12)
    centroid_error = abs(s["spectral_centroid_hz"] - r["spectral_centroid_hz"]) / max(r["spectral_centroid_hz"], 1e-6)
    dominant_error = abs(s["dominant_frequency_hz"] - r["dominant_frequency_hz"]) / max(r["dominant_frequency_hz"], 1e-6)
    return {
        "psd_lowfreq_mismatch": float(min(low_mismatch, 10.0)),
        "spectral_centroid_error": float(min(centroid_error, 10.0)),
        "dominant_freq_error": float(min(dominant_error, 10.0)),
        "real_lowfreq_power_ratio": r["lowfreq_power_ratio"],
        "sim_lowfreq_power_ratio": s["lowfreq_power_ratio"],
        "real_spectral_centroid_hz": r["spectral_centroid_hz"],
        "sim_spectral_centroid_hz": s["spectral_centroid_hz"],
        "real_dominant_frequency_hz": r["dominant_frequency_hz"],
        "sim_dominant_frequency_hz": s["dominant_frequency_hz"],
    }


def chunk_stability_penalty(chunks, bus_id, suffix, sim, spec):
    vals = []
    for chunk in chunks:
        col = raw_col(bus_id, suffix)
        if col not in chunk["df"].columns:
            continue
        real = pd.to_numeric(chunk["df"][col], errors="coerce").dropna().to_numpy(float)
        real = transform_real_for_spec(real, spec)
        if len(real) > 0 and len(sim) > 0:
            vals.append(float(ks_2samp(real, sim).statistic))
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def run_andes_normal():
    case_path = andes.get_case("ieee39/ieee39_full.xlsx")
    system = andes.load(case_path)
    print("Running ANDES PFlow...")
    if system.PFlow.run() is False:
        raise RuntimeError("ANDES power flow failed")
    print(f"Running normal-operation TDS ({TF:.0f}s, {1/TSTEP:.0f} Hz)...")
    system.TDS.config.tf = TF
    system.TDS.config.tstep = TSTEP
    if system.TDS.run() is False:
        raise RuntimeError("ANDES TDS failed")
    return system


def get_timeseries(system):
    bus_ids = [str(b) for b in system.Bus.idx.v]
    if hasattr(system.TDS, "get_timeseries"):
        v_df = system.TDS.get_timeseries(system.Bus.v)
        a_df = system.TDS.get_timeseries(system.Bus.a)
        v_df.columns = [str(c) for c in v_df.columns]
        a_df.columns = [str(c) for c in a_df.columns]
        t = v_df.index.to_numpy(float)
        return t, v_df, a_df
    ts = system.dae.ts
    t = np.asarray(ts.t, dtype=float)
    v_df = pd.DataFrame(np.asarray(ts.y[:, system.Bus.v.a], dtype=float), index=t, columns=bus_ids)
    a_df = pd.DataFrame(np.asarray(ts.y[:, system.Bus.a.a], dtype=float), index=t, columns=bus_ids)
    return t, v_df, a_df


def get_bus_vn_kv(system, bus_id):
    bus_ids = [str(b) for b in system.Bus.idx.v]
    if bus_id not in bus_ids:
        return 345.0
    pos = bus_ids.index(bus_id)
    if hasattr(system.Bus, "Vn"):
        try:
            return float(system.Bus.Vn.v[pos])
        except Exception:
            pass
    return 345.0


def get_system_mva(system):
    for attr in ["mva", "MVA", "sbase", "Sbase"]:
        try:
            val = getattr(system.config, attr)
            return float(val)
        except Exception:
            pass
    return 100.0


def current_base_amp(system, bus_id):
    vn_kv = max(get_bus_vn_kv(system, bus_id), 1e-6)
    return get_system_mva(system) * 1e6 / (math.sqrt(3.0) * vn_kv * 1e3)


def build_ybus(system, bus_ids):
    lines = system.Line
    pos = {str(b): i for i, b in enumerate(bus_ids)}
    ybus = np.zeros((len(bus_ids), len(bus_ids)), dtype=complex)

    def val(param, k, default=0.0):
        try:
            return float(param.v[k])
        except Exception:
            return default

    for k in range(lines.n):
        fb, tb = str(lines.bus1.v[k]), str(lines.bus2.v[k])
        if fb not in pos or tb not in pos:
            continue
        fi, ti = pos[fb], pos[tb]
        r, x = val(lines.r, k), val(lines.x, k)
        g = val(lines.g, k) if hasattr(lines, "g") else 0.0
        b = val(lines.b, k) if hasattr(lines, "b") else 0.0
        tap = val(lines.tap, k, 1.0) if hasattr(lines, "tap") else 1.0
        phi = val(lines.phi, k, 0.0) if hasattr(lines, "phi") else 0.0
        if abs(tap) < 1e-12:
            tap = 1.0
        y = 1.0 / (r + 1j * x) if (abs(r) + abs(x)) > 1e-12 else complex(g, 0.0)
        ysh = 0.5 * (g + 1j * b)
        tc = tap * np.exp(1j * phi)
        ybus[fi, fi] += (y + ysh) / abs(tc) ** 2
        ybus[ti, ti] += y + ysh
        ybus[fi, ti] -= y / np.conj(tc)
        ybus[ti, fi] -= y / tc
    return ybus, pos


def compute_bus_injection_current(system, t, v_df, a_df):
    bus_ids = [str(b) for b in system.Bus.idx.v]
    ybus, pos = build_ybus(system, bus_ids)
    cols = [b for b in bus_ids if b in v_df.columns and b in a_df.columns]
    v_complex = v_df[cols].to_numpy(float) * np.exp(1j * a_df[cols].to_numpy(float))
    i_complex = (ybus[np.ix_([pos[c] for c in cols], [pos[c] for c in cols])] @ v_complex.T).T
    out = {}
    for i, bus_id in enumerate(cols):
        out[bus_id] = np.abs(i_complex[:, i]) * current_base_amp(system, bus_id)
    return out


def compute_incident_branch_currents(system, t, v_df, a_df):
    lines = system.Line
    out = defaultdict_list()

    def val(param, k, default=0.0):
        try:
            return float(param.v[k])
        except Exception:
            return default

    for k in range(lines.n):
        fb, tb = str(lines.bus1.v[k]), str(lines.bus2.v[k])
        if fb not in v_df.columns or tb not in v_df.columns:
            continue
        r, x = val(lines.r, k), val(lines.x, k)
        g = val(lines.g, k) if hasattr(lines, "g") else 0.0
        b = val(lines.b, k) if hasattr(lines, "b") else 0.0
        tap = val(lines.tap, k, 1.0) if hasattr(lines, "tap") else 1.0
        phi = val(lines.phi, k, 0.0) if hasattr(lines, "phi") else 0.0
        if abs(tap) < 1e-12:
            tap = 1.0
        y = 1.0 / (r + 1j * x) if (abs(r) + abs(x)) > 1e-12 else complex(g, 0.0)
        ysh = 0.5 * (g + 1j * b)
        tc = tap * np.exp(1j * phi)

        vf = v_df[fb].to_numpy(float) * np.exp(1j * a_df[fb].to_numpy(float))
        vt = v_df[tb].to_numpy(float) * np.exp(1j * a_df[tb].to_numpy(float))
        vf_t = vf / tc
        i_from = (vf_t - vt) * y + ysh * vf_t
        i_to = (vt - vf_t) * y + ysh * vt
        out[fb].append((f"branch_{k}_from_{fb}_to_{tb}", np.abs(i_from) * current_base_amp(system, fb)))
        out[tb].append((f"branch_{k}_to_{tb}_from_{fb}", np.abs(i_to) * current_base_amp(system, tb)))
    return out


def defaultdict_list():
    from collections import defaultdict
    return defaultdict(list)


def compute_generator_current_candidates(system, v_df):
    out = defaultdict_list()
    for model_name in ["PV", "Slack"]:
        if not hasattr(system, model_name):
            continue
        model = getattr(system, model_name)
        if not all(hasattr(model, a) for a in ["bus", "p", "q"]):
            continue
        for k in range(model.n):
            bus_id = str(model.bus.v[k])
            if bus_id not in v_df.columns:
                continue
            p = float(model.p.v[k])
            q = float(model.q.v[k])
            v = np.maximum(v_df[bus_id].to_numpy(float), 1e-6)
            i_pu = np.sqrt(p * p + q * q) / v
            out[bus_id].append((f"generator_{model_name}_{k}", i_pu * current_base_amp(system, bus_id)))
    return out


def build_all_trajectories(system):
    t, v_df, a_df = get_timeseries(system)
    bus_injection = compute_bus_injection_current(system, t, v_df, a_df)
    branch_currents = compute_incident_branch_currents(system, t, v_df, a_df)
    generator_currents = compute_generator_current_candidates(system, v_df)

    trajectories = {}
    for bus_id in PMU_BUSES:
        if bus_id not in v_df.columns:
            continue
        angle = a_df[bus_id].to_numpy(float)
        angle_unwrapped = np.unwrap(angle)
        omega = np.gradient(angle_unwrapped, t)
        freq = 60.0 + omega / (2.0 * np.pi)
        rocof = np.gradient(freq, t)
        trajectories[bus_id] = {
            "t": t,
            "voltage_mag": v_df[bus_id].to_numpy(float),
            "voltage_angle_deg": np.rad2deg(angle),
            "frequency": freq,
            "rocof": rocof,
            "current_candidates": build_current_candidates_for_bus(
                bus_id, bus_injection, branch_currents, generator_currents
            ),
        }
    return trajectories


def build_current_candidates_for_bus(bus_id, bus_injection, branch_currents, generator_currents):
    candidates = {}
    if bus_id in bus_injection:
        candidates["bus_injection_current"] = bus_injection[bus_id]
    for name, arr in branch_currents.get(bus_id, []):
        candidates[f"branch_current:{name}"] = arr
    if branch_currents.get(bus_id):
        name, arr = max(branch_currents[bus_id], key=lambda item: float(np.mean(item[1])))
        candidates[f"dominant_incident_branch_current:{name}"] = arr
        p_name, p_arr = max(branch_currents[bus_id], key=lambda item: float(np.mean(np.square(item[1]))))
        candidates[f"max_power_branch_current:{p_name}"] = p_arr
    for name, arr in generator_currents.get(bus_id, []):
        candidates[f"generator_terminal_current:{name}"] = arr
    return candidates


def percentile_mismatch(real, sim):
    real_p = np.percentile(real, [1, 5, 50, 95, 99])
    sim_p = np.percentile(sim, [1, 5, 50, 95, 99])
    denom = max(float(np.std(real)), float(np.median(np.abs(real_p))), 1e-9)
    return float(np.mean(np.abs(real_p - sim_p) / denom))


def compute_basic_metrics(real, sim, family="current_mag", fs=30.0, chunk_penalty=0.0):
    real = np.asarray(real, dtype=float)
    sim = np.asarray(sim, dtype=float)
    real = real[np.isfinite(real)]
    sim = sim[np.isfinite(sim)]
    if len(real) == 0 or len(sim) == 0:
        return {
            "ks_stat": np.nan,
            "relative_mean_error": np.inf,
            "relative_std_error": np.inf,
            "percentile_error": np.inf,
            "psd_lowfreq_mismatch": np.inf,
            "spectral_centroid_error": np.inf,
            "dominant_freq_error": np.inf,
            "chunk_stability_penalty": np.inf,
            "composite_score": np.inf,
        }
    ks = float(ks_2samp(real, sim).statistic)
    rel_std = float(abs(np.std(sim) - np.std(real)) / (abs(np.std(real)) + 1e-12))
    rel_mean = float(abs(np.mean(sim) - np.mean(real)) / (abs(np.mean(real)) + 1e-12))
    p_err = percentile_mismatch(real, sim)
    spec = spectral_mismatch(real, sim, fs) if family in {"current_mag", "frequency"} else {
        "psd_lowfreq_mismatch": 0.0,
        "spectral_centroid_error": 0.0,
        "dominant_freq_error": 0.0,
    }
    return {
        "ks_stat": ks,
        "relative_mean_error": rel_mean,
        "relative_std_error": rel_std,
        "percentile_error": p_err,
        "psd_lowfreq_mismatch": spec["psd_lowfreq_mismatch"],
        "spectral_centroid_error": spec["spectral_centroid_error"],
        "dominant_freq_error": spec["dominant_freq_error"],
        "chunk_stability_penalty": float(chunk_penalty),
        "composite_score": composite_score(
            ks, rel_mean, rel_std, p_err, family,
            spec["psd_lowfreq_mismatch"], spec["spectral_centroid_error"], spec["dominant_freq_error"], chunk_penalty
        ),
    }


def composite_score(
    ks,
    rel_mean,
    rel_std,
    p_err,
    family,
    psd_lowfreq_mismatch=0.0,
    spectral_centroid_error=0.0,
    dominant_freq_error=0.0,
    chunk_stability_penalty=0.0,
):
    psd = min(float(psd_lowfreq_mismatch), 10.0)
    centroid = min(float(spectral_centroid_error), 10.0)
    dominant = min(float(dominant_freq_error), 10.0)
    stability = min(float(chunk_stability_penalty), 1.0)
    if family == "current_mag":
        return float(
            0.30 * ks + 0.06 * rel_mean + 0.24 * rel_std + 0.18 * p_err
            + 0.12 * psd + 0.05 * centroid + 0.02 * dominant + 0.03 * stability
        )
    if family == "frequency":
        return float(
            0.34 * ks + 0.10 * rel_mean + 0.18 * rel_std + 0.12 * p_err
            + 0.14 * psd + 0.08 * centroid + 0.02 * dominant + 0.02 * stability
        )
    return float(0.55 * ks + 0.15 * rel_mean + 0.15 * rel_std + 0.15 * p_err + 0.02 * stability)


def fit_affine_distribution(sim, real, family):
    sim = np.asarray(sim, dtype=float)
    real = np.asarray(real, dtype=float)
    sim = sim[np.isfinite(sim)]
    real = real[np.isfinite(real)]
    if len(sim) == 0 or len(real) == 0:
        return 1.0, 0.0, "identity_empty"
    if np.std(sim) < 1e-12:
        return 0.0, float(np.median(real)), "median_constant"

    if family == "current_mag":
        qs = [1, 5, 50, 95, 99]
        w = np.asarray([1.5, 1.5, 1.0, 2.0, 2.5])
    else:
        qs = [5, 50, 95]
        w = np.ones(3)
    sx = np.percentile(sim, qs)
    ry = np.percentile(real, qs)
    xbar = np.average(sx, weights=w)
    ybar = np.average(ry, weights=w)
    denom = np.sum(w * (sx - xbar) ** 2)
    if denom < 1e-18:
        return 0.0, float(np.median(real)), "median_constant"
    a = float(np.sum(w * (sx - xbar) * (ry - ybar)) / denom)
    b = float(ybar - a * xbar)
    return a, b, "weighted_percentile_affine"


def estimate_drift_targets_from_real(real, noise_stats, family):
    real = np.asarray(real, dtype=float)
    real = real[np.isfinite(real)]
    total_std = float(np.std(real)) if len(real) else 0.0
    noise_std = float(noise_stats.get("std_dev_abs", noise_stats.get("std_dev_raw", 0.0)))
    if family in {"voltage_mag", "current_mag", "frequency"}:
        drift_std = math.sqrt(max(total_std**2 - min(noise_std, 0.55 * total_std) ** 2, 0.0))
        drift_std = max(drift_std, 0.25 * total_std)
    elif family == "angle_voltage":
        drift_std = max(0.35 * total_std, math.sqrt(max(total_std**2 - min(noise_std, 0.45 * total_std) ** 2, 0.0)))
    else:
        drift_std = 0.0
    empirical_deviation = real - float(np.median(real)) if len(real) else np.array([], dtype=float)
    spectral = estimate_real_spectral_targets(real, noise_stats, fs=30.0)
    return {
        "total_std": total_std,
        "noise_std": noise_std,
        "drift_std": float(drift_std),
        "empirical_deviation": empirical_deviation,
        "lag1": autocorr(real, 1),
        "lag5": autocorr(real, 5),
        "spectral": spectral,
    }


def estimate_real_spectral_targets(real, noise_stats=None, fs=30.0):
    real_spec = spectral_features(real, fs)
    profile_spec = {}
    if isinstance(noise_stats, dict):
        profile_spec = noise_stats.get("psd_summary", {}) or noise_stats.get("fft_low_freq_summary", {}) or {}
    top_freqs = profile_spec.get("top_frequencies_hz", []) if isinstance(profile_spec, dict) else []
    top_amps = profile_spec.get("top_amplitudes", []) if isinstance(profile_spec, dict) else []
    if not top_freqs:
        centered = np.asarray(real, dtype=float) - float(np.nanmean(real))
        if len(centered) > 8:
            freqs = np.fft.rfftfreq(len(centered), d=1.0 / fs)
            amps = np.abs(np.fft.rfft(centered)) / max(len(centered), 1)
            order = np.argsort(amps[1:])[-3:][::-1] + 1
            top_freqs = [float(freqs[i]) for i in order if freqs[i] <= 0.5]
            top_amps = [float(amps[i]) for i in order if freqs[i] <= 0.5]
    return {
        **real_spec,
        "top_frequencies_hz": [float(f) for f in top_freqs[:3] if float(f) > 0.0],
        "top_amplitudes": [float(a) for a in top_amps[:3]],
    }


def synthesize_low_frequency_drift_from_fft(t, target_stats, rng, drift_std):
    spectral = target_stats.get("spectral", {}) or {}
    freqs = spectral.get("top_frequencies_hz", []) or []
    amps = spectral.get("top_amplitudes", []) or []
    t = np.asarray(t, dtype=float)
    if len(t) == 0 or drift_std <= 0:
        return np.zeros(len(t))
    y = np.zeros(len(t))
    for freq, amp in zip(freqs[:3], amps[:3]):
        freq = float(freq)
        if freq <= 0 or freq > 0.5:
            continue
        phase = rng.uniform(0.0, 2.0 * np.pi)
        y += float(amp) * np.sin(2.0 * np.pi * freq * (t - t[0]) + phase)
    if np.std(y) < 1e-15:
        duration = max(float(t[-1] - t[0]), 1.0) if len(t) > 1 else 1.0
        y = np.sin(2.0 * np.pi * (t - t[0]) / duration + rng.uniform(0.0, 2.0 * np.pi))
    y -= np.mean(y)
    if np.std(y) > 1e-15:
        lowfreq_ratio = float(spectral.get("lowfreq_power_ratio", 0.5))
        y *= drift_std * min(max(lowfreq_ratio, 0.25), 0.95) / np.std(y)
    return y


def rank_shape_to_empirical(source, empirical_samples):
    source = np.asarray(source, dtype=float)
    empirical_samples = np.asarray(empirical_samples, dtype=float)
    empirical_samples = empirical_samples[np.isfinite(empirical_samples)]
    if len(source) == 0 or len(empirical_samples) < 8 or np.std(source) < 1e-12:
        return source
    order = np.argsort(source)
    ranks = np.empty(len(source), dtype=float)
    ranks[order] = (np.arange(len(source), dtype=float) + 0.5) / len(source)
    shaped = np.quantile(empirical_samples, ranks)
    return shaped - float(np.mean(shaped))


def apply_operating_drift(signal, t, family, bus_id, target_stats, rng, drift_scale=1.0):
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    if n == 0 or DRIFT_MODE == "none":
        return signal.copy()
    drift_std = float(target_stats.get("drift_std", 0.0)) * float(drift_scale)
    if drift_std <= 0:
        return signal.copy()

    rho_by_family = {
        "voltage_mag": 0.996,
        "current_mag": 0.994,
        "frequency": 0.998,
        "angle_voltage": 0.997,
    }
    rho = rho_by_family.get(family, 0.995)
    white_std = drift_std * math.sqrt(max(0.0, 1.0 - rho**2))
    ar = np.zeros(n)
    white = rng.normal(0.0, white_std, n)
    ar[0] = white[0]
    for i in range(1, n):
        ar[i] = rho * ar[i - 1] + white[i]

    if DRIFT_MODE == "fft_residual_synthesis":
        low = synthesize_low_frequency_drift_from_fft(t, target_stats, rng, drift_std)
    else:
        duration = max(float(t[-1] - t[0]), 1.0) if len(t) > 1 else 1.0
        phase = rng.uniform(0, 2 * np.pi)
        low = 0.35 * drift_std * np.sin(2 * np.pi * t / duration + phase)
    drift = 0.65 * ar + low
    empirical = target_stats.get("empirical_deviation")
    if DRIFT_MODE == "fft_residual_synthesis" and empirical is not None and family in {"current_mag", "frequency"}:
        drift = rank_shape_to_empirical(drift, np.asarray(empirical, dtype=float) * float(drift_scale))
    if np.std(drift) > 1e-12:
        drift *= drift_std / np.std(drift)
    return signal + drift


def fit_residual_distribution(noise_stats, family, requested_model=NOISE_MODEL):
    if requested_model != "auto":
        return requested_model
    preferred = str(noise_stats.get("fitted_distribution_type", "gaussian"))
    kurt = float(noise_stats.get("kurtosis", 0.0))
    skewness = abs(float(noise_stats.get("skewness", 0.0)))
    residual_samples = noise_stats.get("residual_samples", []) or []
    if family in {"current_mag", "frequency"} and residual_samples and (preferred == "bootstrap" or kurt > 3.0):
        return "bootstrap"
    if family in {"current_mag", "frequency"} and (kurt > 1.5 or preferred == "student_t"):
        return "student_t"
    if skewness > 0.6 or preferred == "gmm":
        return "gmm"
    return "gaussian"


def candidate_noise_models(noise_stats, family):
    if NOISE_MODEL != "auto":
        return [fit_residual_distribution(noise_stats, family, NOISE_MODEL)]
    preferred = fit_residual_distribution(noise_stats, family, "auto")
    if family in {"current_mag", "frequency"}:
        modes = [preferred, "bootstrap", "student_t"]
        if abs(float(noise_stats.get("skewness", 0.0))) > 0.6:
            modes.append("gmm")
    else:
        modes = [preferred]
    out = []
    for mode in modes:
        if mode not in out:
            out.append(mode)
    return out


def sample_block_bootstrap(samples, n, rng, block=16):
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return np.zeros(n)
    if len(samples) <= block:
        return rng.choice(samples, size=n, replace=True)
    out = []
    while len(out) < n:
        start = int(rng.integers(0, len(samples) - block + 1))
        out.extend(samples[start:start + block])
    return np.asarray(out[:n], dtype=float)


def sample_residual_process(noise_stats, n, rng, mode, noise_scale=1.0):
    if n <= 0:
        return np.array([], dtype=float)
    sigma = float(noise_stats.get("std_dev_abs", noise_stats.get("std_dev_raw", 1e-6))) * float(noise_scale)
    rho = float(np.clip(noise_stats.get("ar1_rho", 0.0), -0.999, 0.999))

    if mode == "bootstrap":
        innovations = sample_block_bootstrap(noise_stats.get("residual_samples", []), n, rng)
        if np.std(innovations) > 1e-12:
            innovations = innovations * (sigma / np.std(innovations))
    elif mode == "student_t":
        kurt = float(noise_stats.get("kurtosis", 3.0))
        if not np.isfinite(kurt):
            kurt = 3.0
        kurt = max(kurt, 0.1)
        df = float(np.clip(6.0 / kurt + 4.0, 3.0, 30.0))
        innovations = student_t_dist.rvs(df, size=n, random_state=rng)
        innovations = innovations / max(np.std(innovations), 1e-12) * sigma
    elif mode == "gmm":
        tail_prob = float(np.clip(noise_stats.get("p_outlier", 0.01), 0.01, 0.20))
        wide_sigma = max(float(noise_stats.get("outlier_mag_abs", 0.0)), 2.5 * sigma)
        mask = rng.random(n) < tail_prob
        innovations = rng.normal(0.0, sigma, n)
        innovations[mask] = rng.normal(0.0, wide_sigma, int(np.sum(mask)))
    else:
        innovations = rng.normal(0.0, sigma, n)

    white = innovations * math.sqrt(max(0.0, 1.0 - rho**2))
    noise = np.zeros(n)
    noise[0] = white[0]
    for i in range(1, n):
        noise[i] = rho * noise[i - 1] + white[i]

    p = float(np.clip(noise_stats.get("p_outlier", 0.0), 0.0, 1.0))
    mag = float(noise_stats.get("outlier_mag_abs", 0.0)) * float(noise_scale)
    if p > 0 and mag > 0:
        mask = rng.random(n) < p
        signs = rng.choice([-1.0, 1.0], size=n)
        spikes = mask * signs * np.abs(rng.normal(mag, max(mag * 0.25, 1e-12), n))
    else:
        spikes = 0.0
    return noise + spikes


def score_distribution_fit(real_residuals, sampled_residuals):
    real_residuals = np.asarray(real_residuals, dtype=float)
    sampled_residuals = np.asarray(sampled_residuals, dtype=float)
    real_residuals = real_residuals[np.isfinite(real_residuals)]
    sampled_residuals = sampled_residuals[np.isfinite(sampled_residuals)]
    if len(real_residuals) == 0 or len(sampled_residuals) == 0:
        return np.inf
    ks = float(ks_2samp(real_residuals, sampled_residuals).statistic)
    qerr = percentile_mismatch(real_residuals, sampled_residuals)
    return 0.7 * ks + 0.3 * qerr


def fit_quantile_mapping(sim, real, n_quantiles=31):
    sim = np.asarray(sim, dtype=float)
    real = np.asarray(real, dtype=float)
    sim = sim[np.isfinite(sim)]
    real = real[np.isfinite(real)]
    if len(sim) < 8 or len(real) < 8:
        return None
    qs = np.linspace(0.01, 0.99, n_quantiles)
    src = np.quantile(sim, qs)
    dst = np.quantile(real, qs)
    keep = np.r_[True, np.diff(src) > 1e-12]
    if np.sum(keep) < 3:
        return None
    return {"src": src[keep].tolist(), "dst": dst[keep].tolist()}


def apply_quantile_mapping(values, mapping):
    if mapping is None:
        return values
    src = np.asarray(mapping["src"], dtype=float)
    dst = np.asarray(mapping["dst"], dtype=float)
    return np.interp(np.asarray(values, dtype=float), src, dst, left=dst[0], right=dst[-1])


def condition_prepared_to_chunk(prepared, real_chunk, family):
    if family not in {"voltage_mag", "current_mag", "frequency"}:
        out = dict(prepared)
        out["chunk_conditioning_used"] = "no"
        return out
    mapping = fit_quantile_mapping(prepared["sim_noisy"], real_chunk, n_quantiles=21)
    if mapping is None:
        out = dict(prepared)
        out["chunk_conditioning_used"] = "no"
        return out
    out = dict(prepared)
    out["sim_clean"] = apply_quantile_mapping(prepared["sim_clean"], mapping)
    out["sim_drifted"] = apply_quantile_mapping(prepared["sim_drifted"], mapping)
    out["sim_noisy"] = apply_quantile_mapping(prepared["sim_noisy"], mapping)
    out["fit_mode"] = f"{prepared['fit_mode']}+chunk_quantile_conditioning"
    out["chunk_conditioning_used"] = "yes"
    return out


def choose_best_current_mapping_for_bus(bus_id, chunks, traj):
    real_phases = []
    for ph in ["A", "B", "C"]:
        vals = raw_values_from_chunks(chunks, bus_id, f"I{ph}_MAG")
        if len(vals):
            real_phases.append(vals)
    if not real_phases:
        return None, []
    real = np.concatenate(real_phases)

    rows = []
    for name, candidate in traj.get("current_candidates", {}).items():
        if candidate is None or len(candidate) == 0:
            continue
        a, b, fit_mode = fit_affine_distribution(candidate, real, "current_mag")
        prelim = a * np.asarray(candidate, dtype=float) + b
        per_chunk_ks = []
        for chunk in chunks:
            phase_vals = []
            for ph in ["A", "B", "C"]:
                col = raw_col(bus_id, f"I{ph}_MAG")
                if col in chunk["df"].columns:
                    vals = pd.to_numeric(chunk["df"][col], errors="coerce").dropna().to_numpy(float)
                    if len(vals):
                        phase_vals.append(vals)
            if phase_vals:
                per_chunk_ks.append(float(ks_2samp(np.concatenate(phase_vals), prelim).statistic))
        stability = float(np.std(per_chunk_ks)) if len(per_chunk_ks) > 1 else 0.0
        metrics = compute_basic_metrics(real, prelim, "current_mag", fs=30.0, chunk_penalty=stability)
        rows.append({
            "bus_id": bus_id,
            "candidate": name,
            "fit_mode": fit_mode,
            "fit_a": a,
            "fit_b": b,
            "ks_stat": metrics["ks_stat"],
            "relative_mean_error": metrics["relative_mean_error"],
            "relative_std_error": metrics["relative_std_error"],
            "percentile_error": metrics["percentile_error"],
            "psd_lowfreq_mismatch": metrics["psd_lowfreq_mismatch"],
            "spectral_centroid_error": metrics["spectral_centroid_error"],
            "dominant_freq_error": metrics["dominant_freq_error"],
            "chunk_stability_penalty": stability,
            "composite_score": metrics["composite_score"],
        })
    if not rows:
        return None, []
    rows = sorted(rows, key=lambda r: r["composite_score"])
    return rows[0]["candidate"], rows


def select_current_mappings(all_chunks, trajectories):
    selection_rows = []
    diagnostics_rows = []
    chosen = {}
    for bus_id in PMU_BUSES:
        best, rows = choose_best_current_mapping_for_bus(bus_id, all_chunks.get(bus_id, []), trajectories.get(bus_id, {}))
        if best is not None:
            chosen[bus_id] = best
        diagnostics_rows.extend(rows)
        selection_rows.append({
            "bus_id": bus_id,
            "chosen_mapping": best or "unsupported_mapping",
            "score": rows[0]["composite_score"] if rows else np.nan,
            "candidate_scores_json": json.dumps(rows),
            "notes": "Lowest composite score from KS, std error, quantiles, spectral mismatch, and chunk stability.",
        })
    out_path = os.path.join(OUTPUT_DIR, "current_mapping_selection.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(out_path, index=False)
    diag_path = os.path.join(OUTPUT_DIR, "metrics", "current_mapping_diagnostics.csv")
    os.makedirs(os.path.dirname(diag_path), exist_ok=True)
    pd.DataFrame(diagnostics_rows).to_csv(diag_path, index=False)
    plot_current_mapping_diagnostics(diagnostics_rows)
    return chosen


def plot_current_mapping_diagnostics(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    for bus_id, grp in df.groupby("bus_id"):
        grp = grp.sort_values("composite_score").head(12)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        labels = [str(x).replace("branch_current:", "br:").replace("dominant_incident_", "dom_") for x in grp["candidate"]]
        ax.barh(labels[::-1], grp["composite_score"].to_numpy()[::-1])
        ax.set_xlabel("composite score")
        ax.set_title(f"Bus {bus_id} current mapping candidates")
        ax.grid(axis="x", alpha=0.3)
        out_path = os.path.join(OUTPUT_DIR, "plots", "current_mapping", "by_bus", f"Bus{bus_id}.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)


def sim_source_for_spec(spec, traj, current_mapping):
    source = spec["sim_source"]
    if source == "current_mag_selected":
        if current_mapping is None:
            return None
        return traj.get("current_candidates", {}).get(current_mapping)
    if source == "voltage_angle_delta":
        y = traj.get("voltage_angle_deg")
        if y is None:
            return None
        return wrap_deg(y - np.median(y))
    return traj.get(source)


def transform_real_for_spec(values, spec):
    if spec["support_status"] in {"supported_wrapped_delta", "experimental_relative_only"}:
        return wrap_deg(values - np.median(values))
    return values


def prepare_calibrated_series(bus_id, spec, real_aggregate, sim_source, sim_t, noise_layer, rng, chunks):
    family = signal_family_from_key(spec["signal_key"])
    raw_column = raw_col(bus_id, spec["raw_suffix"])
    noise_stats, profile_status = noise_layer.resolve_stats(bus_id, EVENT_LABEL, raw_column)

    a, b, fit_mode = fit_affine_distribution(sim_source, real_aggregate, family)
    sim_clean = a * np.asarray(sim_source, dtype=float) + b
    drift_targets = estimate_drift_targets_from_real(real_aggregate, noise_stats, family)
    fs = sample_rate_from_t(sim_t)
    noise_modes = candidate_noise_models(noise_stats, family)

    best = None
    for drift_scale in DRIFT_SCALE_GRID:
        for noise_scale in NOISE_SCALE_GRID:
            for noise_mode in noise_modes:
                local_seed = (
                    RNG_SEED
                    + int(bus_id) * 1009
                    + sum(ord(c) for c in spec["signal_key"]) * 31
                    + int(drift_scale * 1000) * 17
                    + int(noise_scale * 1000)
                    + sum(ord(c) for c in noise_mode) * 13
                )
                local_rng = np.random.default_rng(local_seed)
                sim_drifted = apply_operating_drift(
                    sim_clean, sim_t, family, bus_id, drift_targets, local_rng, drift_scale=drift_scale
                )
                sim_noisy = sim_drifted + sample_residual_process(
                    noise_stats, len(sim_drifted), local_rng, noise_mode, noise_scale=noise_scale
                )

                quantile_mapping = None
                q_used = "no"
                fit_mode_candidate = fit_mode
                sim_clean_eval = sim_clean
                sim_drifted_eval = sim_drifted
                sim_noisy_eval = sim_noisy
                if CALIBRATION_FIT_MODE == "affine_plus_quantile" and family in {"current_mag", "frequency"}:
                    candidate_mapping = fit_quantile_mapping(sim_noisy, real_aggregate)
                    if candidate_mapping is not None:
                        sim_noisy_q = apply_quantile_mapping(sim_noisy, candidate_mapping)
                        sim_drifted_q = apply_quantile_mapping(sim_drifted, candidate_mapping)
                        sim_clean_q = apply_quantile_mapping(sim_clean, candidate_mapping)
                        base_metrics = full_metrics(
                            real_aggregate, sim_clean, sim_drifted, sim_noisy, family, fs=fs,
                            chunk_stability=0.0
                        )
                        q_metrics = full_metrics(
                            real_aggregate, sim_clean_q, sim_drifted_q, sim_noisy_q, family, fs=fs,
                            chunk_stability=0.0
                        )
                        if q_metrics is not None and (base_metrics is None or q_metrics["composite_score"] < base_metrics["composite_score"]):
                            quantile_mapping = candidate_mapping
                            q_used = "yes"
                            fit_mode_candidate = f"{fit_mode}+quantile_mapping"
                            sim_clean_eval = sim_clean_q
                            sim_drifted_eval = sim_drifted_q
                            sim_noisy_eval = sim_noisy_q

                metrics = full_metrics(real_aggregate, sim_clean_eval, sim_drifted_eval, sim_noisy_eval, family, fs=fs, chunk_stability=0.0)
                score = metrics["composite_score"] if metrics is not None else np.inf
                candidate = {
                    "score": score,
                    "sim_clean": sim_clean_eval,
                    "sim_drifted": sim_drifted_eval,
                    "sim_noisy": sim_noisy_eval,
                    "drift_scale": drift_scale,
                    "noise_scale": noise_scale,
                    "noise_model_used": noise_mode,
                    "quantile_mapping_used": q_used,
                    "quantile_mapping": quantile_mapping,
                    "fit_mode": fit_mode_candidate,
                }
                if best is None or candidate["score"] < best["score"]:
                    best = candidate

    if best is None:
        sim_clean_final = sim_clean.copy()
        sim_drifted = sim_clean.copy()
        sim_noisy = sim_clean.copy()
        drift_scale = 0.0
        noise_scale = 0.0
        noise_model_used = "none"
        quantile_mapping_used = "no"
        fit_mode_final = fit_mode
        stability = 0.0
    else:
        sim_clean_final = best["sim_clean"]
        sim_drifted = best["sim_drifted"]
        sim_noisy = best["sim_noisy"]
        drift_scale = best["drift_scale"]
        noise_scale = best["noise_scale"]
        noise_model_used = best["noise_model_used"]
        quantile_mapping_used = best["quantile_mapping_used"]
        fit_mode_final = best["fit_mode"]
        stability = chunk_stability_penalty(chunks, bus_id, spec["raw_suffix"], sim_noisy, spec)
    return {
        "sim_clean": sim_clean_final,
        "sim_drifted": sim_drifted,
        "sim_noisy": sim_noisy,
        "fit_a": a,
        "fit_b": b,
        "fit_mode": fit_mode_final,
        "profile_status": profile_status,
        "drift_std_target": drift_targets["drift_std"],
        "noise_std_target": drift_targets["noise_std"],
        "drift_scale": drift_scale,
        "noise_scale": noise_scale,
        "noise_model_used": noise_model_used,
        "quantile_mapping_used": quantile_mapping_used,
        "chunk_stability_penalty": stability,
    }


def full_metrics(real, sim_clean, sim_drifted, sim_noisy, family, fs=30.0, chunk_stability=0.0):
    real = np.asarray(real, dtype=float)
    real = real[np.isfinite(real)]
    sim_clean = np.asarray(sim_clean, dtype=float)
    sim_drifted = np.asarray(sim_drifted, dtype=float)
    sim_noisy = np.asarray(sim_noisy, dtype=float)
    if len(real) == 0 or len(sim_noisy) == 0:
        return None
    ks, p = ks_2samp(real, sim_noisy)
    mean_real = float(np.mean(real))
    mean_clean = float(np.mean(sim_clean))
    mean_drifted = float(np.mean(sim_drifted))
    mean_noisy = float(np.mean(sim_noisy))
    std_real = float(np.std(real))
    std_clean = float(np.std(sim_clean))
    std_drifted = float(np.std(sim_drifted))
    std_noisy = float(np.std(sim_noisy))
    rel_mean = float(abs(mean_noisy - mean_real) / (abs(mean_real) + 1e-12))
    rel_std = float(abs(std_noisy - std_real) / (abs(std_real) + 1e-12))
    p_err = percentile_mismatch(real, sim_noisy)
    spec = spectral_mismatch(real, sim_noisy, fs) if family in {"current_mag", "frequency"} else {
        "psd_lowfreq_mismatch": 0.0,
        "spectral_centroid_error": 0.0,
        "dominant_freq_error": 0.0,
        "real_lowfreq_power_ratio": 0.0,
        "sim_lowfreq_power_ratio": 0.0,
        "real_spectral_centroid_hz": 0.0,
        "sim_spectral_centroid_hz": 0.0,
        "real_dominant_frequency_hz": 0.0,
        "sim_dominant_frequency_hz": 0.0,
    }
    score = composite_score(
        float(ks), rel_mean, rel_std, p_err, family,
        spec["psd_lowfreq_mismatch"], spec["spectral_centroid_error"], spec["dominant_freq_error"], chunk_stability
    )

    out = {
        "ks_stat": float(ks),
        "ks_pvalue": float(p),
        "mean_real": mean_real,
        "mean_sim_clean": mean_clean,
        "mean_sim_drifted": mean_drifted,
        "mean_sim_noisy": mean_noisy,
        "std_real": std_real,
        "std_sim_clean": std_clean,
        "std_sim_drifted": std_drifted,
        "std_sim_noisy": std_noisy,
        "relative_mean_error": rel_mean,
        "relative_std_error": rel_std,
        "percentile_error": p_err,
        "autocorr_lag1_real": autocorr(real, 1),
        "autocorr_lag1_sim_noisy": autocorr(sim_noisy, 1),
        "autocorr_lag5_real": autocorr(real, 5),
        "autocorr_lag5_sim_noisy": autocorr(sim_noisy, 5),
        "psd_lowfreq_mismatch": spec["psd_lowfreq_mismatch"],
        "spectral_centroid_error": spec["spectral_centroid_error"],
        "dominant_freq_error": spec["dominant_freq_error"],
        "real_lowfreq_power_ratio": spec["real_lowfreq_power_ratio"],
        "sim_lowfreq_power_ratio": spec["sim_lowfreq_power_ratio"],
        "real_spectral_centroid_hz": spec["real_spectral_centroid_hz"],
        "sim_spectral_centroid_hz": spec["sim_spectral_centroid_hz"],
        "real_dominant_frequency_hz": spec["real_dominant_frequency_hz"],
        "sim_dominant_frequency_hz": spec["sim_dominant_frequency_hz"],
        "chunk_stability_penalty": float(chunk_stability),
        "composite_score": score,
        "num_samples_real": int(len(real)),
        "num_samples_sim": int(len(sim_noisy)),
        "quality_bucket": quality_bucket(float(ks)),
    }
    for label, arr in [("real", real), ("sim_clean", sim_clean), ("sim_drifted", sim_drifted), ("sim_noisy", sim_noisy)]:
        pcts = np.percentile(arr, [1, 5, 50, 95, 99])
        for name, val in zip(["p01", "p05", "p50", "p95", "p99"], pcts):
            out[f"{name}_{label}"] = float(val)
    return out


def make_calibration_plot(bus_id, signal_key, chunk_id, scope, t, real, prepared, metrics, support_status, profile_status, out_path):
    n = min(500, len(t), len(real))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Bus {bus_id} | {signal_key} | {scope}:{chunk_id} | KS={metrics['ks_stat']:.4f} "
        f"p={metrics['ks_pvalue']:.2e}\nSupport={support_status} | Profile={profile_status}",
        fontsize=9,
    )
    axes[0].plot(t[:n], prepared["sim_clean"][:n], lw=1.0, label="ANDES clean fit")
    axes[0].plot(t[:n], prepared["sim_drifted"][:n], lw=0.9, alpha=0.85, label="clean + operating drift")
    axes[0].plot(t[:n], prepared["sim_noisy"][:n], lw=0.8, alpha=0.75, label="drift + raw noise")
    if n > 0:
        axes[0].plot(np.linspace(t[0], t[min(n - 1, len(t) - 1)], n), real[:n], lw=0.8, alpha=0.75, label="raw event0")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    axes[1].hist(real, bins=50, density=True, alpha=0.55, label="raw event0")
    axes[1].hist(prepared["sim_noisy"], bins=50, density=True, alpha=0.55, label="ANDES+drift+noise")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def make_spectral_plot(bus_id, signal_key, t, real, prepared, out_path):
    fs = sample_rate_from_t(t)
    real = np.asarray(real, dtype=float)
    sim = np.asarray(prepared["sim_noisy"], dtype=float)
    real = real[np.isfinite(real)]
    sim = sim[np.isfinite(sim)]
    if len(real) < 8 or len(sim) < 8:
        return
    fr, pr = welch(real - np.mean(real), fs=fs, nperseg=min(512, len(real)), detrend="constant")
    fsim, psim = welch(sim - np.mean(sim), fs=fs, nperseg=min(512, len(sim)), detrend="constant")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(fr, pr + 1e-30, label="raw event0")
    ax.semilogy(fsim, psim + 1e-30, label="ANDES+model")
    ax.set_xlim(0, min(2.0, fs / 2.0))
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD")
    ax.set_title(f"Spectral match | Bus {bus_id} | {signal_key}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def evaluate_supported_signal(bus_id, spec, chunks, traj, current_mapping, noise_layer, rng):
    sim_source = sim_source_for_spec(spec, traj, current_mapping)
    if sim_source is None:
        return [], []

    representation = "wrapped_delta" if spec["support_status"] in {"supported_wrapped_delta", "experimental_relative_only"} else "raw"
    real_aggregate = raw_values_from_chunks(chunks, bus_id, spec["raw_suffix"], representation)
    if len(real_aggregate) == 0:
        return [{
            "bus_id": bus_id,
            "event_label": EVENT_LABEL,
            "signal_key": spec["signal_key"],
            "scope": "aggregate",
            "chunk_id": "ALL_CHUNK0",
            "signal_family": signal_family_from_key(spec["signal_key"]),
            "support_status": spec["support_status"],
            "profile_status": "not_used",
            "status": "skipped",
            "quality_bucket": "invalid_mapping",
            "reason": "raw_column_missing_or_empty",
        }], []

    t = traj["t"]
    prepared = prepare_calibrated_series(bus_id, spec, real_aggregate, sim_source, t, noise_layer, rng, chunks)
    family = signal_family_from_key(spec["signal_key"])
    fs = sample_rate_from_t(t)
    records, plots = [], []

    if CALIBRATION_MODE in {"per_chunk", "both"}:
        for chunk in chunks:
            col = raw_col(bus_id, spec["raw_suffix"])
            if col not in chunk["df"].columns:
                continue
            real = pd.to_numeric(chunk["df"][col], errors="coerce").dropna().to_numpy(float)
            real = transform_real_for_spec(real, spec)
            prepared_scope = condition_prepared_to_chunk(prepared, real, family)
            stability = chunk_stability_penalty(chunks, bus_id, spec["raw_suffix"], prepared_scope["sim_noisy"], spec)
            metrics = full_metrics(
                real, prepared_scope["sim_clean"], prepared_scope["sim_drifted"], prepared_scope["sim_noisy"], family,
                fs=fs, chunk_stability=stability
            )
            if metrics is None:
                continue
            status = "supported" if prepared["profile_status"] == "exact_match" else "fallback_profile"
            record = {
                "bus_id": bus_id,
                "event_label": EVENT_LABEL,
                "event_name": "normal_operation",
                "signal_key": spec["signal_key"],
                "raw_column_used": col,
                "chunk_id": chunk["chunk_id"],
                "scope": "per_chunk",
                "signal_family": family,
                "support_status": spec["support_status"],
                "profile_status": prepared["profile_status"],
                "status": status,
                "fit_mode": prepared_scope["fit_mode"],
                "fit_a": prepared["fit_a"],
                "fit_b": prepared["fit_b"],
                "drift_mode": DRIFT_MODE,
                "drift_std_target": prepared["drift_std_target"],
                "drift_scale": prepared["drift_scale"],
                "noise_std_target": prepared["noise_std_target"],
                "noise_scale": prepared["noise_scale"],
                "noise_model_used": prepared["noise_model_used"],
                "quantile_mapping_used": prepared["quantile_mapping_used"],
                "chunk_conditioning_used": prepared_scope["chunk_conditioning_used"],
                "current_mapping": current_mapping if spec["sim_source"] == "current_mag_selected" else "",
                **metrics,
            }
            records.append(record)
            plots.append((chunk["chunk_id"], "per_chunk", real, prepared_scope, metrics))

    if CALIBRATION_MODE in {"aggregate", "both"}:
        metrics = full_metrics(
            real_aggregate, prepared["sim_clean"], prepared["sim_drifted"], prepared["sim_noisy"], family,
            fs=fs, chunk_stability=prepared["chunk_stability_penalty"]
        )
        if metrics is not None:
            status = "supported" if prepared["profile_status"] == "exact_match" else "fallback_profile"
            records.append({
                "bus_id": bus_id,
                "event_label": EVENT_LABEL,
                "event_name": "normal_operation",
                "signal_key": spec["signal_key"],
                "raw_column_used": raw_col(bus_id, spec["raw_suffix"]),
                "chunk_id": "ALL_CHUNK0",
                "scope": "aggregate",
                "signal_family": family,
                "support_status": spec["support_status"],
                "profile_status": prepared["profile_status"],
                "status": status,
                "fit_mode": prepared["fit_mode"],
                "fit_a": prepared["fit_a"],
                "fit_b": prepared["fit_b"],
                "drift_mode": DRIFT_MODE,
                "drift_std_target": prepared["drift_std_target"],
                "drift_scale": prepared["drift_scale"],
                "noise_std_target": prepared["noise_std_target"],
                "noise_scale": prepared["noise_scale"],
                "noise_model_used": prepared["noise_model_used"],
                "quantile_mapping_used": prepared["quantile_mapping_used"],
                "chunk_conditioning_used": "no",
                "current_mapping": current_mapping if spec["sim_source"] == "current_mag_selected" else "",
                **metrics,
            })
            plots.append(("ALL_CHUNK0", "aggregate", real_aggregate, prepared, metrics))

    plot_payloads = []
    for chunk_id, scope, real, prepared_for_plot, metrics in plots:
        plot_payloads.append((chunk_id, scope, real, prepared_for_plot, metrics))
    return records, plot_payloads


def save_plot_copies(bus_id, spec, plot_payload, traj):
    chunk_id, scope, real, prepared, metrics = plot_payload
    signal_name = spec["signal_key"]
    safe_signal = signal_name.replace("/", "_").replace("\\", "_")
    if scope == "aggregate":
        out_path = os.path.join(OUTPUT_DIR, "plots", "aggregate", f"Bus{bus_id}", f"{safe_signal}.png")
        make_calibration_plot(
            bus_id, signal_name, chunk_id, scope, traj["t"], real, prepared,
            metrics, spec["support_status"], prepared["profile_status"], out_path
        )
        if signal_name in SPECTRAL_SIGNAL_KEYS:
            spectral_path = os.path.join(OUTPUT_DIR, "plots", "spectral", "by_signal", safe_signal, f"Bus{bus_id}.png")
            make_spectral_plot(bus_id, signal_name, traj["t"], real, prepared, spectral_path)
        return

    by_bus = os.path.join(OUTPUT_DIR, "plots", "by_bus", f"Bus{bus_id}", safe_signal, f"{chunk_id}.png")
    by_signal = os.path.join(OUTPUT_DIR, "plots", "by_signal", safe_signal, f"Bus{bus_id}_{chunk_id}.png")
    make_calibration_plot(
        bus_id, signal_name, chunk_id, scope, traj["t"], real, prepared,
        metrics, spec["support_status"], prepared["profile_status"], by_bus
    )
    os.makedirs(os.path.dirname(by_signal), exist_ok=True)
    shutil.copy2(by_bus, by_signal)


def unsupported_record(bus_id, spec):
    return {
        "bus_id": bus_id,
        "event_label": EVENT_LABEL,
        "event_name": "normal_operation",
        "signal_key": spec["signal_key"],
        "raw_column_used": raw_col(bus_id, spec["raw_suffix"]),
        "chunk_id": "ALL_CHUNK0",
        "scope": "aggregate",
        "signal_family": signal_family_from_key(spec["signal_key"]),
        "support_status": spec["support_status"],
        "profile_status": "not_used",
        "status": "unsupported",
        "quality_bucket": "invalid_mapping",
        "reason": spec["notes"],
    }


def summarize_active(active, group_cols):
    if active.empty:
        return pd.DataFrame()
    return (
        active.groupby(group_cols, dropna=False)
        .agg(
            n=("signal_key", "count"),
            ks_mean=("ks_stat", "mean"),
            ks_median=("ks_stat", "median"),
            composite_score_mean=("composite_score", "mean"),
            relative_mean_error_mean=("relative_mean_error", "mean"),
            relative_std_error_mean=("relative_std_error", "mean"),
            psd_lowfreq_mismatch_mean=("psd_lowfreq_mismatch", "mean"),
            spectral_centroid_error_mean=("spectral_centroid_error", "mean"),
            chunk_stability_penalty_mean=("chunk_stability_penalty", "mean"),
            very_good_pct=("quality_bucket", lambda s: float(np.mean(s == "very_good") * 100.0)),
            usable_pct=("quality_bucket", lambda s: float(np.mean(s == "usable") * 100.0)),
            needs_tuning_pct=("quality_bucket", lambda s: float(np.mean(s == "needs_tuning") * 100.0)),
            poor_pct=("quality_bucket", lambda s: float(np.mean(s == "poor") * 100.0)),
        )
        .reset_index()
    )


def write_support_matrix(df):
    active_agg = df[(df["status"].isin(ACTIVE_STATUSES)) & (df["scope"] == "aggregate")]
    rows = []
    for spec in SIGNAL_SPECS:
        sig_rows = active_agg[active_agg["signal_key"] == spec["signal_key"]]
        mean_ks = float(sig_rows["ks_stat"].mean()) if not sig_rows.empty else np.nan
        mean_score = float(sig_rows["composite_score"].mean()) if not sig_rows.empty else np.nan
        supported = spec["support_status"] not in {"unsupported_raw_absolute", "unsupported_physics", "unsupported_mapping"}
        recommended = bool(supported and np.isfinite(mean_ks) and mean_ks < 0.15 and mean_score < 0.30 and spec["recommended"])
        rows.append({
            "signal_name": spec["signal_key"],
            "raw_representation": spec["raw_representation"],
            "supported": "yes" if supported else "no",
            "support_status": spec["support_status"],
            "reason": spec["notes"],
            "recommended_for_training": "yes" if recommended else "no",
            "mean_ks_event0": mean_ks,
            "mean_composite_score_event0": mean_score,
            "notes": spec["notes"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "signal_support_matrix.csv"), index=False)


def write_outputs(records):
    metrics_dir = os.path.join(OUTPUT_DIR, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    df = pd.DataFrame(records)
    long_path = os.path.join(metrics_dir, "calibration_metrics_long.csv")
    df.to_csv(long_path, index=False)

    active = df[df["status"].isin(ACTIVE_STATUSES)].copy()
    summarize_active(active, ["bus_id"]).to_csv(os.path.join(metrics_dir, "calibration_summary_by_bus.csv"), index=False)
    summarize_active(active, ["signal_key"]).to_csv(os.path.join(metrics_dir, "calibration_summary_by_signal.csv"), index=False)
    summarize_active(active[active["scope"] == "per_chunk"], ["chunk_id", "bus_id", "signal_key"]).to_csv(
        os.path.join(metrics_dir, "calibration_summary_chunk0.csv"), index=False
    )
    summarize_active(active, ["signal_family"]).to_csv(
        os.path.join(metrics_dir, "calibration_summary_signal_family.csv"), index=False
    )
    with open(os.path.join(metrics_dir, "calibration_results.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "event_label": EVENT_LABEL,
            "drift_mode": DRIFT_MODE,
            "records": records,
        }, fh, indent=2, default=str)
    write_support_matrix(df)
    return df


def run_raw_event0_calibration():
    np.random.seed(RNG_SEED)
    rng = np.random.default_rng(RNG_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    noise_layer = RawPMUNoiseLayer(RAW_PROFILE_FILE)
    system = run_andes_normal()
    trajectories = build_all_trajectories(system)
    all_chunks = {bus_id: load_event0_chunks_for_bus(bus_id) for bus_id in PMU_BUSES}
    current_mappings = select_current_mappings(all_chunks, trajectories)

    records = []
    for bus_id in PMU_BUSES:
        traj = trajectories.get(bus_id)
        chunks = all_chunks.get(bus_id, [])
        if traj is None or not chunks:
            for spec in SIGNAL_SPECS:
                rec = unsupported_record(bus_id, spec)
                rec["support_status"] = "unsupported_mapping"
                rec["reason"] = "No ANDES trajectory or raw event-0 chunks for bus."
                records.append(rec)
            continue

        for spec in SIGNAL_SPECS:
            if spec["support_status"].startswith("unsupported") or spec["sim_source"] is None:
                records.append(unsupported_record(bus_id, spec))
                continue
            mapping = current_mappings.get(bus_id) if spec["sim_source"] == "current_mag_selected" else None
            sig_records, plot_payloads = evaluate_supported_signal(bus_id, spec, chunks, traj, mapping, noise_layer, rng)
            records.extend(sig_records)
            for payload in plot_payloads:
                save_plot_copies(bus_id, spec, payload, traj)

    df = write_outputs(records)
    active = df[df["status"].isin(ACTIVE_STATUSES)]
    print("=" * 72)
    print("RAW EVENT-0 CALIBRATION SUMMARY")
    print(f"Rows: {len(df)}")
    print(f"Evaluated rows: {len(active)}")
    print(f"Status counts: {df['status'].value_counts(dropna=False).to_dict()}")
    if "profile_status" in df.columns:
        print(f"Profile status counts: {df['profile_status'].value_counts(dropna=False).to_dict()}")
    print(f"Output: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 72)


if __name__ == "__main__":
    run_raw_event0_calibration()
