#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import andes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER CONFIG
# ============================================================
CASE = andes.get_case("ieee39/ieee39_full.xlsx")
RAW_PATH = Path("data/metadata/IEEE_39_Bus_Power_System.raw")
REALISM_RECIPE_PATH = Path("results/pmu_realism_model/json/simulation_copy_recipe.json")

FAULT_BUS = 39
SIM_END = 6000.0
FAULT_START = 1200.0
FAULT_DURATION = 0.02
FAULT_CLEAR = FAULT_START + FAULT_DURATION

NOM_FREQ_HZ = 60.0
INTERNAL_DT = 1.0 / 120.0
EXPORT_DT = 1.0 / 30.0
FORCE_RUN_IF_UNSTABLE = True

ZOOM_PRE_SEC = 5.0
ZOOM_POST_SEC = 5.0
ZOOM_DPI = 1200
FULL_PLOT_DPI = 220

RNG_SEED = 20260415
APPLY_REALISM_RECIPE = True
APPLY_VISIBLE_EVENT_DURATION = True
APPLY_GENERIC_NOISE = True
APPLY_GENERIC_ARTIFACTS = True
INJECT_REFERENCE_BAD_DATA_EVENTS = True
INJECT_REFERENCE_MISSING_EVENTS = True
INJECT_REFERENCE_LABEL6_EVENT = False

RUN_NAME = "SIM001_REALISM_V3"
OUT_DIR = Path("results") / "synth" / "andes" / RUN_NAME
CSV_DIR = OUT_DIR / "csv"
CSV_PMU_CLEAN_DIR = CSV_DIR / "pmu_clean"
CSV_PMU_REALISM_DIR = CSV_DIR / "pmu_realism"
CSV_ALL39_BY_BUS_DIR = CSV_DIR / "all39_clean_by_bus"
JSON_DIR = OUT_DIR / "json"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_PMU_CLEAN_DIR = PLOTS_DIR / "pmu_clean"
PLOTS_PMU_CLEAN_ZOOM_DIR = PLOTS_PMU_CLEAN_DIR / "zoom_fault_window"
PLOTS_PMU_REALISM_DIR = PLOTS_DIR / "pmu_realism"
PLOTS_PMU_REALISM_ZOOM_DIR = PLOTS_PMU_REALISM_DIR / "zoom_fault_window"
PLOTS_ALL39_DIR = PLOTS_DIR / "all39_clean"
PLOTS_ALL39_ZOOM_DIR = PLOTS_ALL39_DIR / "zoom_fault_window"
TXT_DIR = OUT_DIR / "reports"

BUS_MAPPING_CSV = CSV_DIR / "bus_mapping_semantic_to_raw.csv"
PMU_MERGED_CLEAN_CSV = CSV_DIR / "MultiPMU_Competition_Merged_clean.csv"
PMU_MERGED_REALISM_CSV = CSV_DIR / "MultiPMU_Competition_Merged_realism.csv"
ALL39_MERGED_CLEAN_CSV = CSV_DIR / "All39_ANDES_clean_30fps.csv"
YBUS_DIAG_JSON = JSON_DIR / "ybus_diagnostics.json"
RUN_METADATA_JSON = JSON_DIR / "run_metadata.json"
EVENT_LABELS_JSON = JSON_DIR / "event_label_dictionary.json"
PMU_MANIFEST_JSON = JSON_DIR / "pmu_manifest.json"
EVENT_SCHEDULE_JSON = JSON_DIR / "event_schedule_applied.json"
SUMMARY_TXT = TXT_DIR / "summary_report.txt"
YBUS_PNG = PLOTS_DIR / "ybus_abs_difference.png"

PMU_METADATA: Dict[int, Dict[str, Any]] = {
    39: {"pmu_id": 1, "kv": 345.0, "v_pu": 1.0300, "angle_deg": -10.05, "filename": "Bus39_Competition_Data_nanmask.csv"},
    29: {"pmu_id": 2, "kv": 345.0, "v_pu": 1.0499, "angle_deg": +0.75, "filename": "Bus29_Competition_Data_nanmask.csv"},
    10: {"pmu_id": 3, "kv": 345.0, "v_pu": 1.0172, "angle_deg": -5.43, "filename": "Bus10_Competition_Data_nanmask.csv"},
    22: {"pmu_id": 4, "kv": 345.0, "v_pu": 1.0498, "angle_deg": +0.67, "filename": "Bus22_Competition_Data_nanmask.csv"},
    19: {"pmu_id": 5, "kv": 345.0, "v_pu": 1.0499, "angle_deg": -1.02, "filename": "Bus19_Competition_Data_nanmask.csv"},
    2:  {"pmu_id": 6, "kv": 345.0, "v_pu": 1.0487, "angle_deg": -5.75, "filename": "Bus2_Competition_Data_nanmask.csv"},
    5:  {"pmu_id": 7, "kv": 345.0, "v_pu": 1.0053, "angle_deg": -8.61, "filename": "Bus5_Competition_Data_nanmask.csv"},
    6:  {"pmu_id": 8, "kv": 345.0, "v_pu": 1.0077, "angle_deg": -7.95, "filename": "Bus6_Competition_Data_nanmask.csv"},
}
PMU_BUSES_IN_ORDER = [39, 29, 10, 22, 19, 2, 5, 6]

EVENT_LABELS: Dict[int, Dict[str, str]] = {
    0: {"name": "Normal operation", "description": "No disturbance. The system is in steady state or near-steady state. Minor natural fluctuations may be present."},
    1: {"name": "Fault", "description": "A short-circuit event (e.g., three-phase-to-ground). Produces sudden voltage sags and current spikes, typically lasting a few cycles to several seconds."},
    2: {"name": "Line outage", "description": "A transmission line is disconnected (tripped or opened). Causes power-flow redistribution and voltage/angle shifts across the network."},
    3: {"name": "Generation change/outage", "description": "A generator changes its MW output (step change) or trips offline entirely. Causes frequency deviation and system-wide power-flow redistribution."},
    4: {"name": "Load change/drop", "description": "A load suddenly increases, decreases, or disconnects. Similar to generation change but typically produces smaller frequency excursions."},
    5: {"name": "Missing data", "description": "PMU frame missing due to communication failure. All measurements are NaN; DATA_PRESENT = 0. No physical event is occurring."},
    6: {"name": "Missing data + physical event", "description": "Missing data at one PMU concurrent with a physical event elsewhere. Measurements are NaN at the affected PMU; the physical event must be inferred from other PMUs."},
    7: {"name": "Bad data", "description": "Corrupted measurement frame(s): non-physical spikes, jumps, or inconsistent values. The PMU reports data, but the values are unreliable."},
    8: {"name": "Unknown event", "description": "An abnormal pattern that does not match labels 1–7. This is an open-set class for ambiguous or unusual disturbances."},
}

MEAS_COLS = [
    "VA_mag", "VA_ang", "VB_mag", "VB_ang", "VC_mag", "VC_ang",
    "IA_mag", "IA_ang", "IB_mag", "IB_ang", "IC_mag", "IC_ang",
    "Frequency", "ROCOF",
]
ANGLE_COLS = {"VA_ang", "VB_ang", "VC_ang", "IA_ang", "IB_ang", "IC_ang"}
PRIMARY_CLEAN_FEATURES = ["VA_mag", "IA_mag", "Frequency", "ROCOF"]

REFERENCE_BAD_DATA_EVENTS = {
    2: [{"event_id": 7, "start": 3.266, "duration": 0.700}],
    39: [
        {"event_id": 7, "start": 334.366, "duration": 0.734},
        {"event_id": 7, "start": 336.533, "duration": 1.033},
    ],
}
REFERENCE_MISSING_EVENTS = {
    29: [
        {"event_id": 5, "start": 536.566, "duration": 27.034},
        {"event_id": 5, "start": 614.333, "duration": 29.067},
        {"event_id": 5, "start": 2584.666, "duration": 191.400},
        {"event_id": 5, "start": 2935.766, "duration": 40.567},
    ]
}
REFERENCE_LABEL6_EVENTS = {
    29: [{"event_id": 6, "start": 2976.366, "duration": 18.434}],
}


# ============================================================
# HELPERS
# ============================================================
def ensure_dirs() -> None:
    for path in [
        OUT_DIR, CSV_DIR, CSV_PMU_CLEAN_DIR, CSV_PMU_REALISM_DIR, CSV_ALL39_BY_BUS_DIR,
        JSON_DIR, PLOTS_DIR, PLOTS_PMU_CLEAN_DIR, PLOTS_PMU_CLEAN_ZOOM_DIR, PLOTS_PMU_REALISM_DIR, PLOTS_PMU_REALISM_ZOOM_DIR, PLOTS_ALL39_DIR, PLOTS_ALL39_ZOOM_DIR,
        TXT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_idx(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return int(float(x))


def safe_float(x: Any) -> Optional[float]:
    try:
        xf = float(x)
    except Exception:
        return None
    if not np.isfinite(xf):
        return None
    return xf


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return {"real": safe_float(obj.real), "imag": safe_float(obj.imag)}
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=json_default)


def save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def complex_from_mag_ang(mag: np.ndarray, ang_rad: np.ndarray) -> np.ndarray:
    return np.asarray(mag, dtype=float) * np.exp(1j * np.asarray(ang_rad, dtype=float))


def phase_shift(z: np.ndarray, deg: float) -> np.ndarray:
    return np.asarray(z) * np.exp(1j * np.deg2rad(deg))


def safe_angle_deg(z: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.angle(np.asarray(z)))


def wrap_deg(x: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=float) + 180.0) % 360.0) - 180.0


def infer_sampling_frequency(t: np.ndarray) -> Optional[float]:
    if t.size < 2:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def interp_real_with_nan(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    out = np.interp(t_dst, t_src, y_src).astype(float)
    out[t_dst < t_src[0]] = np.nan
    out[t_dst > t_src[-1]] = np.nan
    return out


def interp_complex_with_nan(t_src: np.ndarray, z_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    mag = np.interp(t_dst, t_src, np.abs(z_src))
    ang = np.interp(t_dst, t_src, np.unwrap(np.angle(z_src)))
    out = mag * np.exp(1j * ang)
    mask = (t_dst < t_src[0]) | (t_dst > t_src[-1])
    out = out.astype(complex)
    out[mask] = np.nan + 1j * np.nan
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy(dtype=float)


def rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(np.asarray(x, dtype=float))
    return s.rolling(window=window, center=True, min_periods=1).median().to_numpy(dtype=float)


def mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def parse_raw_bus_mapping(raw_path: Path) -> Tuple[Dict[int, int], Dict[int, int]]:
    text = raw_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    semantic_to_raw: Dict[int, int] = {}
    raw_to_semantic: Dict[int, int] = {}
    in_bus_section = False
    for line in text:
        if not in_bus_section:
            if "IEEE 39 Bus Loadflow" in line:
                in_bus_section = True
            continue
        if line.lstrip().startswith("0 /end bus section"):
            break
        clean = line.split("/")[0].strip()
        if not clean:
            continue
        tokens = [tok.strip() for tok in clean.split(",")]
        if len(tokens) < 2:
            continue
        try:
            raw_id = int(tokens[0])
        except Exception:
            continue
        name = tokens[1].strip().strip("'").strip('"')
        match = re.search(r"BUS(\d+)", name.upper())
        if match is None:
            continue
        semantic_bus = int(match.group(1))
        semantic_to_raw[semantic_bus] = raw_id
        raw_to_semantic[raw_id] = semantic_bus
    if len(semantic_to_raw) < 39:
        raise RuntimeError(f"Could not parse all buses from RAW. Parsed {len(semantic_to_raw)}.")
    return semantic_to_raw, raw_to_semantic


def build_ybus_from_raw(raw_path: Path) -> Tuple[np.ndarray, List[int], Dict[int, int], Dict[int, int]]:
    semantic_to_raw, raw_to_semantic = parse_raw_bus_mapping(raw_path)
    text = raw_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    raw_id_order = sorted(raw_to_semantic.keys())
    raw_id_to_idx = {rid: k for k, rid in enumerate(raw_id_order)}
    nb = len(raw_id_order)
    ybus = np.zeros((nb, nb), dtype=complex)
    in_branch_section = False
    for line in text:
        if not in_branch_section:
            if "0 /end source section starting branch section" in line:
                in_branch_section = True
            continue
        clean = line.split("/")[0].strip()
        if not clean:
            continue
        if clean == "0" or clean.startswith("0 "):
            break
        tokens = [tok.strip() for tok in clean.split(",")]
        if len(tokens) < 11:
            continue
        try:
            raw_i = int(tokens[0])
            raw_j = int(tokens[1])
            r = float(tokens[3])
            x = float(tokens[4])
            b = float(tokens[5])
            tap = float(tokens[9]) if tokens[9] not in ("", "0") else 1.0
            shift_deg = float(tokens[10]) if tokens[10] not in ("",) else 0.0
        except Exception:
            continue
        if raw_i not in raw_id_to_idx or raw_j not in raw_id_to_idx:
            continue
        if abs(r) < 1e-15 and abs(x) < 1e-15:
            continue
        y_series = 1.0 / complex(r, x)
        y_shunt_half = 1j * b / 2.0
        a = tap * np.exp(1j * np.deg2rad(shift_deg))
        ii = raw_id_to_idx[raw_i]
        jj = raw_id_to_idx[raw_j]
        ybus[ii, ii] += (y_series + y_shunt_half) / (a * np.conj(a))
        ybus[jj, jj] += y_series + y_shunt_half
        ybus[ii, jj] += -y_series / np.conj(a)
        ybus[jj, ii] += -y_series / a
    return ybus, raw_id_order, semantic_to_raw, raw_to_semantic


def align_raw_ybus_to_andes_order(ss: Any, raw_path: Path) -> Tuple[np.ndarray, Dict[int, int], Dict[int, int], List[int]]:
    y_raw, raw_id_order, semantic_to_raw, raw_to_semantic = build_ybus_from_raw(raw_path)
    raw_id_to_pos = {rid: pos for pos, rid in enumerate(raw_id_order)}
    andes_bus_order = [normalize_idx(x) for x in ss.Bus.idx.v]
    missing = [b for b in andes_bus_order if b not in semantic_to_raw]
    if missing:
        raise RuntimeError(f"These ANDES buses were not found in RAW mapping: {missing}")
    perm = [raw_id_to_pos[semantic_to_raw[b]] for b in andes_bus_order]
    y_aligned = y_raw[np.ix_(perm, perm)]
    return y_aligned, semantic_to_raw, raw_to_semantic, andes_bus_order


def to_dense_square_matrix(candidate: Any, nb: int) -> Optional[np.ndarray]:
    if candidate is None:
        return None
    try:
        if hasattr(candidate, "toarray"):
            arr = candidate.toarray()
        elif hasattr(candidate, "A"):
            arr = candidate.A
        else:
            arr = np.asarray(candidate)
    except Exception:
        return None
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape != (nb, nb):
        return None
    return arr.astype(complex)


def get_andes_ybus_matrix(ss: Any, nb: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    diag: Dict[str, Any] = {"build_ybus_called": False, "candidate_checks": [], "selected_candidate": None}
    try:
        if hasattr(ss, "build_ybus"):
            ss.build_ybus()
            diag["build_ybus_called"] = True
    except Exception as exc:
        diag["build_ybus_error"] = repr(exc)
    candidates: List[Tuple[str, Any]] = []
    owners = {"system": ss, "dae": getattr(ss, "dae", None), "pflow": getattr(ss, "PFlow", None)}
    names = ["Ybus", "ybus", "Y", "Y_bus", "_Ybus", "_ybus"]
    for owner_name, owner in owners.items():
        if owner is None:
            continue
        for name in names:
            candidates.append((f"{owner_name}.{name}", getattr(owner, name, None)))
    for label, value in candidates:
        arr = to_dense_square_matrix(value, nb)
        diag["candidate_checks"].append({
            "label": label,
            "is_none": value is None,
            "accepted": arr is not None,
            "shape": list(arr.shape) if arr is not None else None,
        })
        if arr is not None:
            diag["selected_candidate"] = label
            return arr, diag
    return None, diag


def compare_ybus(y_ref: np.ndarray, y_test: Optional[np.ndarray]) -> Dict[str, Any]:
    if y_test is None:
        return {
            "andes_ybus_accessible": False,
            "same_matrix": False,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "fro_norm_diff": None,
            "ref_fro_norm": float(np.linalg.norm(y_ref)),
            "relative_fro_error": None,
        }
    diff = y_ref - y_test
    ref_norm = float(np.linalg.norm(y_ref))
    fro_diff = float(np.linalg.norm(diff))
    rel_fro = fro_diff / ref_norm if ref_norm > 0 else None
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    return {
        "andes_ybus_accessible": True,
        "same_matrix": bool(max_abs <= 1e-8 or (rel_fro is not None and rel_fro <= 1e-8)),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "fro_norm_diff": fro_diff,
        "ref_fro_norm": ref_norm,
        "relative_fro_error": rel_fro,
    }


def plot_ybus_difference(diff: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(np.abs(diff), aspect="auto")
    ax.set_title("|Ybus_raw_aligned - Ybus_andes|")
    ax.set_xlabel("Bus index in ANDES order")
    ax.set_ylabel("Bus index in ANDES order")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=FULL_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def get_bus_col(system: Any, bus_no: int) -> int:
    bus_ids = [normalize_idx(x) for x in system.Bus.idx.v]
    if bus_no not in bus_ids:
        raise KeyError(f"Bus {bus_no} not found in ANDES case.")
    return bus_ids.index(bus_no)


def load_realism_recipe(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Realism recipe JSON not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_bus_recipe(recipe: Dict[str, Any], bus: int) -> Dict[str, Any]:
    bus_id = f"Bus{bus}"
    per_bus = recipe.get("per_bus", {})
    if bus_id not in per_bus:
        raise KeyError(f"{bus_id} not present in realism recipe.")
    return per_bus[bus_id]


def estimate_local_frequency_from_voltage_angle(t: np.ndarray, vang_rad: np.ndarray, nominal_hz: float) -> np.ndarray:
    theta = np.unwrap(np.asarray(vang_rad, dtype=float))
    dtheta_dt = np.gradient(theta, t)
    freq = nominal_hz + dtheta_dt / (2.0 * np.pi)
    freq = rolling_median(freq, 5)
    freq = rolling_mean(freq, 5)
    return np.asarray(freq, dtype=float)


def estimate_rocof_from_frequency(t: np.ndarray, freq: np.ndarray) -> np.ndarray:
    rocof = np.gradient(np.asarray(freq, dtype=float), np.asarray(t, dtype=float))
    rocof = rolling_median(rocof, 5)
    rocof = rolling_mean(rocof, 5)
    return np.asarray(rocof, dtype=float)


def sample_innovations(n: int, family: str, scale: float, rng: np.random.Generator) -> np.ndarray:
    scale = float(scale or 0.0)
    if scale <= 0:
        return np.zeros(n, dtype=float)
    family = (family or "gaussian").lower()
    if family == "laplace":
        x = rng.laplace(loc=0.0, scale=scale / np.sqrt(2.0), size=n)
    elif family == "student_t":
        x = rng.standard_t(df=6.0, size=n)
        x = x / max(np.std(x), 1e-9) * scale
    else:
        x = rng.normal(loc=0.0, scale=scale, size=n)
    return x.astype(float)


def sample_ar1_noise(n: int, sigma: float, phi: float, family: str, rng: np.random.Generator) -> np.ndarray:
    sigma = float(sigma or 0.0)
    phi = float(np.clip(float(phi or 0.0), -0.98, 0.98))
    eps = sample_innovations(n, family, sigma, rng)
    y = np.zeros(n, dtype=float)
    for k in range(1, n):
        y[k] = phi * y[k - 1] + np.sqrt(max(1e-12, 1.0 - phi**2)) * eps[k]
    return y


def sanitize_noise_parameters(channel: str, noise_model: Dict[str, Any]) -> Tuple[float, float, str]:
    sigma = float(noise_model.get("residual_std") or 0.0)
    phi = float(noise_model.get("lag1_autocorr") or 0.0)
    family = str(noise_model.get("best_distribution_family") or "gaussian")
    if channel == "Frequency":
        sigma = min(sigma, 0.05)
    elif channel == "ROCOF":
        sigma = min(sigma, 1.0)
    elif channel in ANGLE_COLS:
        sigma = min(sigma, 0.25)
    else:
        sigma = min(sigma, max(0.20 * abs(float(noise_model.get("trend_std") or sigma or 1.0)), sigma))
    phi = float(np.clip(phi, -0.95, 0.95))
    return sigma, phi, family


def inject_artifact_bursts(x: np.ndarray, t: np.ndarray, channel: str, artifact_model: Dict[str, Any], noise_sigma: float, rng: np.random.Generator) -> np.ndarray:
    y = np.array(x, dtype=float, copy=True)
    rate = float(artifact_model.get("artifact_rate_per_sample") or 0.0)
    durations = [float(d) for d in artifact_model.get("artifact_burst_durations_s") or [] if d is not None and d >= 0.0]
    if len(y) == 0 or rate <= 0 or len(durations) == 0:
        return y
    n = len(y)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else EXPORT_DT
    expected_bursts = max(1, min(12, int(round(rate * n * 0.35))))
    for _ in range(expected_bursts):
        dur_s = float(rng.choice(durations))
        dur_n = max(1, int(round(dur_s / max(dt, 1e-9))))
        start = int(rng.integers(0, max(1, n - dur_n)))
        end = min(n, start + dur_n)
        mode = rng.choice(["spike", "step", "drift"])
        amp = float(rng.uniform(4.0, 12.0) * max(noise_sigma, 1e-9))
        if channel in ANGLE_COLS:
            amp = min(max(amp, 0.5), 10.0)
        elif channel == "Frequency":
            amp = min(max(amp, 0.005), 0.15)
        elif channel == "ROCOF":
            amp = min(max(amp, 0.02), 3.0)
        if mode == "spike":
            y[start:end] += sample_innovations(end - start, "student_t", amp, rng)
        elif mode == "step":
            y[start:end] += amp * rng.choice([-1.0, 1.0])
        else:
            ramp = np.linspace(0.0, amp * rng.choice([-1.0, 1.0]), end - start)
            y[start:end] += ramp
    return y


def inject_missing_event(df: pd.DataFrame, start: float, duration_s: float, event_id: int = 5) -> None:
    mask = (df["TIMESTAMP"] >= start) & (df["TIMESTAMP"] < start + duration_s)
    df.loc[mask, MEAS_COLS] = np.nan
    df.loc[mask, "DATA_PRESENT"] = 0
    df.loc[mask, "Event"] = event_id


def inject_bad_data_event(df: pd.DataFrame, bus_recipe: Dict[str, Any], start: float, duration_s: float, rng: np.random.Generator, event_id: int = 7) -> None:
    mask = (df["TIMESTAMP"] >= start) & (df["TIMESTAMP"] < start + duration_s)
    idx = np.where(mask.to_numpy())[0]
    if len(idx) == 0:
        return
    for ch_name in MEAS_COLS:
        if ch_name not in bus_recipe.get("channels", {}):
            continue
        ch_recipe = bus_recipe["channels"][ch_name]
        noise_model = ch_recipe.get("noise_model", {})
        sigma, _, family = sanitize_noise_parameters(ch_name, noise_model)
        amp = max(sigma, 1e-9) * rng.uniform(8.0, 20.0)
        if ch_name in ANGLE_COLS:
            amp = min(max(amp, 1.0), 20.0)
        elif ch_name == "Frequency":
            amp = min(max(amp, 0.01), 0.5)
        elif ch_name == "ROCOF":
            amp = min(max(amp, 0.05), 5.0)
        corruption = sample_innovations(len(idx), family=family, scale=amp, rng=rng)
        current = pd.to_numeric(df.loc[mask, ch_name], errors="coerce").to_numpy(dtype=float)
        df.loc[mask, ch_name] = current + corruption
    df.loc[mask, "DATA_PRESENT"] = 1
    df.loc[mask, "Event"] = event_id


def calibrate_clean_dataframe(df: pd.DataFrame, bus: int, bus_recipe: Optional[Dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    if bus not in PMU_METADATA:
        return out
    meta = PMU_METADATA[bus]
    prefault = out[(out["TIMESTAMP"] >= max(0.0, FAULT_START - 3.0)) & (out["TIMESTAMP"] < FAULT_START)]
    if prefault.empty:
        prefault = out.iloc[: min(len(out), 300)].copy()
    v_target = meta["v_pu"] * meta["kv"] * 1e3 / np.sqrt(3.0)
    va_pref_med = float(np.nanmedian(prefault["VA_mag"]))
    if np.isfinite(va_pref_med) and va_pref_med > 0:
        v_scale = v_target / va_pref_med
        for col in ["VA_mag", "VB_mag", "VC_mag"]:
            out[col] = pd.to_numeric(out[col], errors="coerce") * v_scale
    va_ang_pref = float(np.nanmedian(prefault["VA_ang"]))
    if np.isfinite(va_ang_pref):
        angle_shift = meta["angle_deg"] - va_ang_pref
        for col in ["VA_ang", "VB_ang", "VC_ang", "IA_ang", "IB_ang", "IC_ang"]:
            out[col] = wrap_deg(pd.to_numeric(out[col], errors="coerce") + angle_shift)
    if bus_recipe is not None:
        target_currents = []
        for col in ["IA_mag", "IB_mag", "IC_mag"]:
            try:
                target_currents.append(float(bus_recipe["channels"][col]["baseline"]["median"]))
            except Exception:
                pass
        clean_currents = [
            float(np.nanmedian(prefault[col])) for col in ["IA_mag", "IB_mag", "IC_mag"]
            if np.isfinite(np.nanmedian(prefault[col])) and np.nanmedian(prefault[col]) > 0
        ]
        if target_currents and clean_currents:
            current_scale = float(np.mean(target_currents) / max(np.mean(clean_currents), 1e-9))
            for col in ["IA_mag", "IB_mag", "IC_mag"]:
                out[col] = pd.to_numeric(out[col], errors="coerce") * current_scale
        try:
            target_freq = float(bus_recipe["channels"]["Frequency"]["baseline"]["mean"])
            clean_freq = float(np.nanmedian(prefault["Frequency"]))
            out["Frequency"] = pd.to_numeric(out["Frequency"], errors="coerce") + (target_freq - clean_freq)
        except Exception:
            pass
    return out


def apply_channel_realism(df: pd.DataFrame, bus_recipe: Dict[str, Any], channel: str, rng: np.random.Generator) -> None:
    ch_recipe = bus_recipe.get("channels", {}).get(channel)
    if ch_recipe is None or channel not in df.columns:
        return
    x = pd.to_numeric(df[channel], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return
    noise_model = ch_recipe.get("noise_model", {})
    artifact_model = ch_recipe.get("artifact_model", {})
    sigma, phi, family = sanitize_noise_parameters(channel, noise_model)
    y = np.array(x, copy=True)
    y[valid] = y[valid] + sample_ar1_noise(valid.sum(), sigma=sigma, phi=phi, family=family, rng=rng)
    if APPLY_GENERIC_ARTIFACTS:
        y = inject_artifact_bursts(y, df["TIMESTAMP"].to_numpy(dtype=float), channel, artifact_model, sigma, rng)
    if channel in ANGLE_COLS:
        y = wrap_deg(y)
    elif channel == "Frequency":
        y = np.clip(y, 55.0, 65.0)
    elif channel == "ROCOF":
        y = np.clip(y, -120.0, 120.0)
    df[channel] = y


def build_bus_dataframe_canonical(bus: int, t: np.ndarray, vpu_bus: np.ndarray, ipu_bus: np.ndarray, freq: np.ndarray, rocof: np.ndarray, actual_end: Optional[float] = None) -> pd.DataFrame:
    kv_ll = float(PMU_METADATA[bus]["kv"]) if bus in PMU_METADATA else 345.0
    vbase_ph = kv_ll * 1e3 / np.sqrt(3.0)
    ibase = 100.0e6 / (np.sqrt(3.0) * kv_ll * 1e3)
    va = vpu_bus * vbase_ph
    vb = phase_shift(va, -120.0)
    vc = phase_shift(va, +120.0)
    ia = ipu_bus * ibase
    ib = phase_shift(ia, -120.0)
    ic = phase_shift(ia, +120.0)
    df = pd.DataFrame({
        "TIMESTAMP": np.round(t, 3),
        "VA_mag": np.abs(va),
        "VA_ang": safe_angle_deg(va),
        "VB_mag": np.abs(vb),
        "VB_ang": safe_angle_deg(vb),
        "VC_mag": np.abs(vc),
        "VC_ang": safe_angle_deg(vc),
        "IA_mag": np.abs(ia),
        "IA_ang": safe_angle_deg(ia),
        "IB_mag": np.abs(ib),
        "IB_ang": safe_angle_deg(ib),
        "IC_mag": np.abs(ic),
        "IC_ang": safe_angle_deg(ic),
        "Frequency": freq,
        "ROCOF": rocof,
        "DATA_PRESENT": 1,
        "Event": 0,
    })
    if actual_end is not None:
        df.loc[df["TIMESTAMP"] > actual_end, MEAS_COLS] = np.nan
        df.loc[df["TIMESTAMP"] > actual_end, "DATA_PRESENT"] = 0
    for col in ANGLE_COLS:
        df[col] = wrap_deg(df[col].to_numpy(dtype=float))
    return df


def canonical_to_single_pmu_csv(df: pd.DataFrame, bus: int) -> pd.DataFrame:
    prefix = f"BUS{bus}"
    return pd.DataFrame({
        "TIMESTAMP": np.round(pd.to_numeric(df["TIMESTAMP"], errors="coerce"), 3),
        f"{prefix}_VA_ANG": pd.to_numeric(df["VA_ang"], errors="coerce"),
        f"{prefix}_VA_MAG": pd.to_numeric(df["VA_mag"], errors="coerce"),
        f"{prefix}_VB_ANG": pd.to_numeric(df["VB_ang"], errors="coerce"),
        f"{prefix}_VB_MAG": pd.to_numeric(df["VB_mag"], errors="coerce"),
        f"{prefix}_VC_ANG": pd.to_numeric(df["VC_ang"], errors="coerce"),
        f"{prefix}_VC_MAG": pd.to_numeric(df["VC_mag"], errors="coerce"),
        f"{prefix}_IA_ANG": pd.to_numeric(df["IA_ang"], errors="coerce"),
        f"{prefix}_IA_MAG": pd.to_numeric(df["IA_mag"], errors="coerce"),
        f"{prefix}_IB_ANG": pd.to_numeric(df["IB_ang"], errors="coerce"),
        f"{prefix}_IB_MAG": pd.to_numeric(df["IB_mag"], errors="coerce"),
        f"{prefix}_IC_ANG": pd.to_numeric(df["IC_ang"], errors="coerce"),
        f"{prefix}_IC_MAG": pd.to_numeric(df["IC_mag"], errors="coerce"),
        f"{prefix}_Freq": pd.to_numeric(df["Frequency"], errors="coerce"),
        f"{prefix}_ROCOF": pd.to_numeric(df["ROCOF"], errors="coerce"),
        "DATA_PRESENT": pd.to_numeric(df["DATA_PRESENT"], errors="coerce").fillna(0).astype(int),
        "Event": pd.to_numeric(df["Event"], errors="coerce").fillna(0).astype(int),
    })


def canonical_to_prefixed_wide(df: pd.DataFrame, bus: int, include_meta_prefix: bool = True) -> pd.DataFrame:
    prefix = f"BUS{bus}"
    out = pd.DataFrame({"TIMESTAMP": np.round(pd.to_numeric(df["TIMESTAMP"], errors="coerce"), 3)})
    mapping = {
        "VA_ang": f"{prefix}_VA_ANG",
        "VA_mag": f"{prefix}_VA_MAG",
        "VB_ang": f"{prefix}_VB_ANG",
        "VB_mag": f"{prefix}_VB_MAG",
        "VC_ang": f"{prefix}_VC_ANG",
        "VC_mag": f"{prefix}_VC_MAG",
        "IA_ang": f"{prefix}_IA_ANG",
        "IA_mag": f"{prefix}_IA_MAG",
        "IB_ang": f"{prefix}_IB_ANG",
        "IB_mag": f"{prefix}_IB_MAG",
        "IC_ang": f"{prefix}_IC_ANG",
        "IC_mag": f"{prefix}_IC_MAG",
        "Frequency": f"{prefix}_Freq",
        "ROCOF": f"{prefix}_ROCOF",
    }
    for src, dst in mapping.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce")
    if include_meta_prefix:
        out[f"{prefix}_DATA_PRESENT"] = pd.to_numeric(df["DATA_PRESENT"], errors="coerce").fillna(0).astype(int)
        out[f"{prefix}_Event"] = pd.to_numeric(df["Event"], errors="coerce").fillna(0).astype(int)
    return out


def make_merged_prefixed_frame(frames: Dict[int, pd.DataFrame], prefix_meta: bool = True, include_global_meta: bool = True) -> pd.DataFrame:
    merged: Optional[pd.DataFrame] = None
    for bus in sorted(frames.keys()):
        wide = canonical_to_prefixed_wide(frames[bus], bus, include_meta_prefix=prefix_meta)
        merged = wide if merged is None else merged.merge(wide, on="TIMESTAMP", how="outer")
    assert merged is not None
    merged = merged.sort_values("TIMESTAMP").reset_index(drop=True)
    if include_global_meta:
        event_cols = [c for c in merged.columns if c.endswith("_Event")]
        dp_cols = [c for c in merged.columns if c.endswith("_DATA_PRESENT")]
        if event_cols:
            merged["Global_Event"] = merged[event_cols].max(axis=1, skipna=True).fillna(0).astype(int)
        if dp_cols:
            merged["All_DATA_PRESENT"] = merged[dp_cols].min(axis=1, skipna=True).fillna(0).astype(int)
            merged["Any_DATA_MISSING"] = (merged[dp_cols].min(axis=1, skipna=True).fillna(1) <= 0).astype(int)
    return merged


def add_fault_markers(ax: Any) -> None:
    ax.axvspan(FAULT_START, FAULT_CLEAR, color="tab:red", alpha=0.15)
    ax.axvline(FAULT_START, linestyle="--", linewidth=1.2, color="tab:red")
    ax.axvline(FAULT_CLEAR, linestyle="--", linewidth=1.2, color="tab:red")
    ax.grid(True, alpha=0.3)


def plot_bus_overview(df: pd.DataFrame, bus: int, path: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    axes[0].plot(df["TIMESTAMP"], df["VA_mag"], label="VA_mag")
    axes[0].plot(df["TIMESTAMP"], df["VB_mag"], label="VB_mag", alpha=0.8)
    axes[0].plot(df["TIMESTAMP"], df["VC_mag"], label="VC_mag", alpha=0.8)
    axes[0].set_ylabel("Voltage [V]")
    axes[0].legend(loc="upper right")
    add_fault_markers(axes[0])

    axes[1].plot(df["TIMESTAMP"], df["IA_mag"], label="IA_mag")
    axes[1].plot(df["TIMESTAMP"], df["IB_mag"], label="IB_mag", alpha=0.8)
    axes[1].plot(df["TIMESTAMP"], df["IC_mag"], label="IC_mag", alpha=0.8)
    axes[1].set_ylabel("Current [A]")
    axes[1].legend(loc="upper right")
    add_fault_markers(axes[1])

    axes[2].plot(df["TIMESTAMP"], df["Frequency"], label="Frequency")
    axes[2].plot(df["TIMESTAMP"], df["ROCOF"], label="ROCOF", alpha=0.8)
    axes[2].set_ylabel("Freq / ROCOF")
    axes[2].legend(loc="upper right")
    add_fault_markers(axes[2])

    axes[3].step(df["TIMESTAMP"], df["DATA_PRESENT"], where="post", label="DATA_PRESENT")
    axes[3].step(df["TIMESTAMP"], df["Event"], where="post", label="Event", alpha=0.8)
    axes[3].set_ylabel("Meta")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(loc="upper right")
    add_fault_markers(axes[3])

    fig.suptitle(f"{title_prefix} | Bus {bus}")
    fig.tight_layout()
    plt.savefig(path, dpi=FULL_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)




def add_fault_and_visible_markers(ax: Any, df: Optional[pd.DataFrame] = None) -> None:
    ax.axvspan(FAULT_START, FAULT_CLEAR, color="tab:red", alpha=0.18, label="Physical fault")
    ax.axvline(FAULT_START, linestyle="--", linewidth=1.2, color="tab:red")
    ax.axvline(FAULT_CLEAR, linestyle="--", linewidth=1.2, color="tab:red")
    if df is not None and "Event" in df.columns:
        event_mask = pd.to_numeric(df["Event"], errors="coerce").fillna(0).to_numpy(dtype=int) == 1
        if event_mask.any():
            t = pd.to_numeric(df["TIMESTAMP"], errors="coerce").to_numpy(dtype=float)
            t_event = t[event_mask]
            ax.axvspan(float(t_event[0]), float(t_event[-1]), color="tab:orange", alpha=0.10, label="Visible PMU fault")
    ax.grid(True, alpha=0.3)


def plot_bus_zoom_channels(df: pd.DataFrame, bus: int, out_dir: Path, title_prefix: str, zoom_pre_s: float = ZOOM_PRE_SEC, zoom_post_s: float = ZOOM_POST_SEC, dpi: int = ZOOM_DPI) -> None:
    t0 = FAULT_START - zoom_pre_s
    t1 = FAULT_CLEAR + zoom_post_s
    zoom_df = df[(df["TIMESTAMP"] >= t0) & (df["TIMESTAMP"] <= t1)].copy()
    if zoom_df.empty:
        return

    grouped = {
        "voltage_mag": ["VA_mag", "VB_mag", "VC_mag"],
        "voltage_ang": ["VA_ang", "VB_ang", "VC_ang"],
        "current_mag": ["IA_mag", "IB_mag", "IC_mag"],
        "current_ang": ["IA_ang", "IB_ang", "IC_ang"],
        "frequency": ["Frequency"],
        "rocof": ["ROCOF"],
        "meta": ["DATA_PRESENT", "Event"],
    }

    ylabels = {
        "voltage_mag": "Voltage [V]",
        "voltage_ang": "Angle [deg]",
        "current_mag": "Current [A]",
        "current_ang": "Angle [deg]",
        "frequency": "Frequency [Hz]",
        "rocof": "ROCOF [Hz/s]",
        "meta": "Meta",
    }

    for group_name, cols in grouped.items():
        fig, ax = plt.subplots(figsize=(22, 8))
        for col in cols:
            if col in zoom_df.columns:
                if col in {"DATA_PRESENT", "Event"}:
                    ax.step(zoom_df["TIMESTAMP"], zoom_df[col], where="post", label=col)
                else:
                    ax.plot(zoom_df["TIMESTAMP"], zoom_df[col], linewidth=1.6, label=col)
        add_fault_and_visible_markers(ax, zoom_df)
        ax.set_xlim(float(t0), float(t1))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabels[group_name])
        ax.set_title(f"{title_prefix} | Bus {bus} | {group_name} | zoom [{zoom_pre_s:.1f}s pre, {zoom_post_s:.1f}s post]")
        ax.legend(loc="best", ncol=min(4, len(cols) + 2))
        fig.tight_layout()
        fig.savefig(out_dir / f"Bus{bus}_{group_name}_zoom.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    for col in MEAS_COLS:
        if col not in zoom_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(24, 8))
        ax.plot(zoom_df["TIMESTAMP"], zoom_df[col], linewidth=1.6, label=col)
        add_fault_and_visible_markers(ax, zoom_df)
        ax.set_xlim(float(t0), float(t1))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(col)
        ax.set_title(f"{title_prefix} | Bus {bus} | {col} | high-res zoom [{zoom_pre_s:.1f}s pre, {zoom_post_s:.1f}s post]")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"Bus{bus}_{col}_zoom_hires.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_all39_zoom_heatmap(t: np.ndarray, values: np.ndarray, bus_ids: List[int], path: Path, title: str, colorbar_label: str, zoom_pre_s: float = ZOOM_PRE_SEC, zoom_post_s: float = ZOOM_POST_SEC, dpi: int = ZOOM_DPI) -> None:
    t0 = FAULT_START - zoom_pre_s
    t1 = FAULT_CLEAR + zoom_post_s
    mask = (t >= t0) & (t <= t1)
    if not np.any(mask):
        return
    t_zoom = t[mask]
    vals_zoom = values[:, mask]
    fig, ax = plt.subplots(figsize=(20, 12))
    image = ax.imshow(
        vals_zoom,
        aspect="auto",
        origin="lower",
        extent=[float(t_zoom[0]), float(t_zoom[-1]), float(min(bus_ids)), float(max(bus_ids))],
    )
    ax.axvspan(FAULT_START, FAULT_CLEAR, color="tab:red", alpha=0.18)
    ax.axvline(FAULT_START, linestyle="--", linewidth=1.2, color="tab:red")
    ax.axvline(FAULT_CLEAR, linestyle="--", linewidth=1.2, color="tab:red")
    ax.set_title(title + f" | zoom [{zoom_pre_s:.1f}s pre, {zoom_post_s:.1f}s post]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bus")
    ax.set_yticks(bus_ids)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_all39_heatmap(t: np.ndarray, values: np.ndarray, bus_ids: List[int], path: Path, title: str, colorbar_label: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    image = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        extent=[float(t[0]), float(t[-1]), float(min(bus_ids)), float(max(bus_ids))],
    )
    ax.axvspan(FAULT_START, FAULT_CLEAR, color="tab:red", alpha=0.15)
    ax.axvline(FAULT_START, linestyle="--", linewidth=1.2, color="tab:red")
    ax.axvline(FAULT_CLEAR, linestyle="--", linewidth=1.2, color="tab:red")
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Bus")
    ax.set_yticks(bus_ids)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    plt.tight_layout()
    plt.savefig(path, dpi=FULL_PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def build_all39_feature_matrices(frames: Dict[int, pd.DataFrame], feature: str) -> Tuple[np.ndarray, List[int], np.ndarray]:
    bus_ids = sorted(frames.keys())
    t = frames[bus_ids[0]]["TIMESTAMP"].to_numpy(dtype=float)
    vals = []
    for bus in bus_ids:
        vals.append(pd.to_numeric(frames[bus][feature], errors="coerce").to_numpy(dtype=float))
    return t, bus_ids, np.vstack(vals)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    ensure_dirs()
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"RAW file not found: {RAW_PATH.resolve()}")
    realism_recipe = load_realism_recipe(REALISM_RECIPE_PATH) if APPLY_REALISM_RECIPE else None

    ss = andes.load(CASE, setup=False)
    if ss is None:
        raise RuntimeError(f"ANDES could not load case: {CASE}")
    ss.add("Fault", bus=FAULT_BUS, tf=FAULT_START, tc=FAULT_CLEAR)
    ss.setup()
    ss.PFlow.run()
    if not ss.PFlow.converged:
        raise RuntimeError("Power flow did not converge.")

    ss.TDS.config.tf = SIM_END
    ss.TDS.config.tstep = INTERNAL_DT
    ss.TDS.config.fixt = 1
    ss.TDS.config.shrinkt = 1
    ss.TDS.config.method = "trapezoid"
    ss.TDS.config.save_every = 1
    ss.TDS.config.criteria = 0 if FORCE_RUN_IF_UNSTABLE else 1
    ss.TDS.run()
    exit_code = getattr(ss.TDS, "exit_code", 0)
    if exit_code != 0:
        raise RuntimeError(f"TDS failed with exit_code={exit_code}")

    t_raw = np.asarray(ss.dae.ts.t, dtype=float)
    if t_raw.size == 0:
        raise RuntimeError("No time samples were saved by ANDES.")
    actual_end = float(t_raw[-1])
    if actual_end < SIM_END - 1e-6:
        warnings.warn(
            f"Simulation stopped early at t={actual_end:.4f} s. The exported CSVs will be padded with NaNs until tf."
        )

    vmag_all = np.asarray(ss.dae.ts.y[:, ss.Bus.v.a], dtype=float)
    vang_all = np.asarray(ss.dae.ts.y[:, ss.Bus.a.a], dtype=float)

    ybus_raw_aligned, semantic_to_raw, raw_to_semantic, andes_bus_order = align_raw_ybus_to_andes_order(ss, RAW_PATH)
    mapping_df = pd.DataFrame({
        "andes_position": np.arange(len(andes_bus_order)),
        "andes_semantic_bus": andes_bus_order,
        "raw_internal_bus": [semantic_to_raw[b] for b in andes_bus_order],
    })
    mapping_df.to_csv(BUS_MAPPING_CSV, index=False)

    ybus_andes, andes_ybus_probe = get_andes_ybus_matrix(ss, nb=len(andes_bus_order))
    ybus_compare = compare_ybus(ybus_raw_aligned, ybus_andes)
    if ybus_andes is not None:
        plot_ybus_difference(ybus_raw_aligned - ybus_andes, YBUS_PNG)

    if ybus_compare["andes_ybus_accessible"] and ybus_compare["same_matrix"]:
        ybus_used = ybus_andes
        ybus_source_used = "andes_internal"
    else:
        ybus_used = ybus_raw_aligned
        ybus_source_used = "raw_aligned_to_andes_order"

    vpu_all_raw = complex_from_mag_ang(vmag_all, vang_all)
    try:
        ipu_all_raw = vpu_all_raw @ ybus_used.T
        current_available = True
    except Exception as exc:
        warnings.warn(f"Could not compute current proxy from selected Ybus: {exc}")
        ipu_all_raw = np.full_like(vpu_all_raw, np.nan + 1j * np.nan)
        current_available = False

    t_out = np.round(np.arange(0.0, SIM_END + 1e-12, EXPORT_DT), 3)
    visible_fault_duration = None
    if realism_recipe is not None:
        try:
            event_lib = realism_recipe.get("global", {}).get("event_library", {})
            visible_fault_duration = float(event_lib["1"].get("duration_median_s") or event_lib["1"].get("duration_mean_s"))
        except Exception:
            visible_fault_duration = None
    fault_visible_duration = visible_fault_duration if (visible_fault_duration is not None and APPLY_VISIBLE_EVENT_DURATION) else FAULT_DURATION

    all39_clean_frames: Dict[int, pd.DataFrame] = {}
    pmu_clean_frames: Dict[int, pd.DataFrame] = {}
    pmu_realism_frames: Dict[int, pd.DataFrame] = {}
    applied_events: Dict[str, List[Dict[str, Any]]] = {}

    for bus in andes_bus_order:
        bus_col = get_bus_col(ss, bus)
        vpu_bus_raw = vpu_all_raw[:, bus_col]
        ipu_bus_raw = ipu_all_raw[:, bus_col]
        freq_raw = estimate_local_frequency_from_voltage_angle(t_raw, vang_all[:, bus_col], NOM_FREQ_HZ)
        rocof_raw = estimate_rocof_from_frequency(t_raw, freq_raw)

        vpu_bus_pmu = interp_complex_with_nan(t_raw, vpu_bus_raw, t_out)
        ipu_bus_pmu = interp_complex_with_nan(t_raw, ipu_bus_raw, t_out)
        freq_pmu = interp_real_with_nan(t_raw, freq_raw, t_out)
        rocof_pmu = interp_real_with_nan(t_raw, rocof_raw, t_out)

        clean_df = build_bus_dataframe_canonical(bus, t_out, vpu_bus_pmu, ipu_bus_pmu, freq_pmu, rocof_pmu, actual_end=actual_end)
        clean_df.loc[(clean_df["TIMESTAMP"] >= FAULT_START) & (clean_df["TIMESTAMP"] < FAULT_START + fault_visible_duration), "Event"] = 1

        bus_recipe = None
        if realism_recipe is not None and bus in PMU_METADATA:
            try:
                bus_recipe = get_bus_recipe(realism_recipe, bus)
            except Exception:
                bus_recipe = None
        clean_df = calibrate_clean_dataframe(clean_df, bus, bus_recipe)
        all39_clean_frames[bus] = clean_df.copy()

        if bus in PMU_METADATA:
            pmu_clean_frames[bus] = clean_df.copy()
            clean_single = canonical_to_single_pmu_csv(clean_df, bus)
            clean_single.to_csv(CSV_PMU_CLEAN_DIR / PMU_METADATA[bus]["filename"], index=False)
            plot_bus_overview(clean_df, bus, PLOTS_PMU_CLEAN_DIR / f"Bus{bus}_clean_overview.png", title_prefix="Clean ANDES + local PMU observation")
            plot_bus_zoom_channels(clean_df, bus, PLOTS_PMU_CLEAN_ZOOM_DIR, title_prefix="Clean ANDES + local PMU observation")

            realism_df = clean_df.copy()
            applied_bus_events: List[Dict[str, Any]] = [{
                "event_id": 1,
                "event_name": EVENT_LABELS[1]["name"],
                "bus": bus,
                "start": FAULT_START,
                "physical_duration_s": FAULT_DURATION,
                "visible_duration_s": fault_visible_duration,
                "source": "ANDES_fault + PMU_visible_duration",
            }]

            if APPLY_REALISM_RECIPE and APPLY_GENERIC_NOISE and bus_recipe is not None:
                rng = np.random.default_rng(RNG_SEED + int(bus))
                for ch in MEAS_COLS:
                    apply_channel_realism(realism_df, bus_recipe, ch, rng)

                if INJECT_REFERENCE_BAD_DATA_EVENTS and bus in REFERENCE_BAD_DATA_EVENTS:
                    for item in REFERENCE_BAD_DATA_EVENTS[bus]:
                        inject_bad_data_event(realism_df, bus_recipe, start=item["start"], duration_s=item["duration"], rng=rng, event_id=7)
                        applied_bus_events.append({
                            "event_id": 7,
                            "event_name": EVENT_LABELS[7]["name"],
                            "bus": bus,
                            "start": item["start"],
                            "duration_s": item["duration"],
                            "source": "reference_bad_data_overlay",
                        })

                if INJECT_REFERENCE_MISSING_EVENTS and bus in REFERENCE_MISSING_EVENTS:
                    for item in REFERENCE_MISSING_EVENTS[bus]:
                        inject_missing_event(realism_df, start=item["start"], duration_s=item["duration"], event_id=5)
                        applied_bus_events.append({
                            "event_id": 5,
                            "event_name": EVENT_LABELS[5]["name"],
                            "bus": bus,
                            "start": item["start"],
                            "duration_s": item["duration"],
                            "source": "reference_missing_data_overlay",
                        })

                if INJECT_REFERENCE_LABEL6_EVENT and bus in REFERENCE_LABEL6_EVENTS:
                    for item in REFERENCE_LABEL6_EVENTS[bus]:
                        inject_missing_event(realism_df, start=item["start"], duration_s=item["duration"], event_id=6)
                        applied_bus_events.append({
                            "event_id": 6,
                            "event_name": EVENT_LABELS[6]["name"],
                            "bus": bus,
                            "start": item["start"],
                            "duration_s": item["duration"],
                            "source": "reference_missing_plus_physical_overlay",
                        })

            for col in ANGLE_COLS:
                realism_df[col] = wrap_deg(pd.to_numeric(realism_df[col], errors="coerce"))
            realism_df["DATA_PRESENT"] = pd.to_numeric(realism_df["DATA_PRESENT"], errors="coerce").fillna(0).astype(int)
            realism_df["Event"] = pd.to_numeric(realism_df["Event"], errors="coerce").fillna(0).astype(int)

            pmu_realism_frames[bus] = realism_df
            realism_single = canonical_to_single_pmu_csv(realism_df, bus)
            realism_single.to_csv(CSV_PMU_REALISM_DIR / PMU_METADATA[bus]["filename"], index=False)
            plot_bus_overview(realism_df, bus, PLOTS_PMU_REALISM_DIR / f"Bus{bus}_realism_overview.png", title_prefix="Post-realism synthetic PMU")
            plot_bus_zoom_channels(realism_df, bus, PLOTS_PMU_REALISM_ZOOM_DIR, title_prefix="Post-realism synthetic PMU")
            applied_events[f"Bus{bus}"] = applied_bus_events

    # Export all 39 buses clean wide and per-bus
    for bus, frame in all39_clean_frames.items():
        canonical_to_prefixed_wide(frame, bus, include_meta_prefix=True).to_csv(CSV_ALL39_BY_BUS_DIR / f"Bus{bus}_ANDES_clean_30fps.csv", index=False)
    all39_merged = make_merged_prefixed_frame(all39_clean_frames, prefix_meta=True, include_global_meta=True)
    all39_merged.to_csv(ALL39_MERGED_CLEAN_CSV, index=False)

    # Export merged PMU frames (clean and realism)
    pmu_clean_merged = make_merged_prefixed_frame(pmu_clean_frames, prefix_meta=True, include_global_meta=True)
    pmu_realism_merged = make_merged_prefixed_frame(pmu_realism_frames, prefix_meta=True, include_global_meta=True)
    pmu_clean_merged.to_csv(PMU_MERGED_CLEAN_CSV, index=False)
    pmu_realism_merged.to_csv(PMU_MERGED_REALISM_CSV, index=False)

    # All-bus plots before realism (clean ANDES for all 39 buses)
    for feature, label in [
        ("VA_mag", "Voltage magnitude [V]"),
        ("IA_mag", "Current magnitude [A]"),
        ("Frequency", "Frequency [Hz]"),
        ("ROCOF", "ROCOF [Hz/s]"),
    ]:
        t_plot, buses_plot, vals_plot = build_all39_feature_matrices(all39_clean_frames, feature)
        plot_all39_heatmap(
            t_plot,
            vals_plot,
            buses_plot,
            PLOTS_ALL39_DIR / f"All39_{feature}_clean_heatmap.png",
            title=f"All 39 buses clean ANDES export | {feature}",
            colorbar_label=label,
        )
        plot_all39_zoom_heatmap(
            t_plot,
            vals_plot,
            buses_plot,
            PLOTS_ALL39_ZOOM_DIR / f"All39_{feature}_clean_zoom_heatmap.png",
            title=f"All 39 buses clean ANDES export | {feature}",
            colorbar_label=label,
        )

    ybus_diag = {
        "raw_alignment_order": andes_bus_order,
        "raw_to_semantic_bus": raw_to_semantic,
        "semantic_to_raw_bus": semantic_to_raw,
        "andes_internal_probe": andes_ybus_probe,
        "comparison": ybus_compare,
        "selected_ybus_source": ybus_source_used,
        "notes": [
            "Current phasors are nodal current injection proxies obtained from V @ Ybus^T, not branch current measurements.",
            "ANDES internal Ybus is preferred only when it is accessible and numerically consistent with the RAW-aligned Ybus.",
        ],
    }
    manifest = {
        "pmu_buses_in_order": PMU_BUSES_IN_ORDER,
        "pmu_metadata": PMU_METADATA,
        "event_labels": EVENT_LABELS,
        "meas_columns_canonical": MEAS_COLS,
        "single_csv_header_style": "TIMESTAMP, BUSxx_* phasor/freq columns, DATA_PRESENT, Event",
        "merged_csv_header_style": "TIMESTAMP plus fully prefixed BUSxx_* columns including BUSxx_DATA_PRESENT and BUSxx_Event",
    }
    metadata = {
        "case": str(CASE),
        "raw_path": str(RAW_PATH),
        "realism_recipe_path": str(REALISM_RECIPE_PATH),
        "fault_bus": FAULT_BUS,
        "sim_end_requested_sec": SIM_END,
        "actual_simulated_end_sec": actual_end,
        "fault_start_sec": FAULT_START,
        "fault_clear_sec": FAULT_CLEAR,
        "fault_duration_sec": FAULT_DURATION,
        "fault_visible_duration_sec": fault_visible_duration,
        "internal_dt_sec": INTERNAL_DT,
        "export_dt_sec": EXPORT_DT,
        "export_sampling_hz": infer_sampling_frequency(t_out),
        "pmu_buses": PMU_BUSES_IN_ORDER,
        "all39_buses": andes_bus_order,
        "current_proxy_available": current_available,
        "current_proxy_source": ybus_source_used,
        "apply_realism_recipe": APPLY_REALISM_RECIPE,
        "apply_generic_noise": APPLY_GENERIC_NOISE,
        "apply_generic_artifacts": APPLY_GENERIC_ARTIFACTS,
        "inject_reference_bad_data_events": INJECT_REFERENCE_BAD_DATA_EVENTS,
        "inject_reference_missing_events": INJECT_REFERENCE_MISSING_EVENTS,
        "inject_reference_label6_event": INJECT_REFERENCE_LABEL6_EVENT,
        "outputs": {
            "csv_pmu_clean_dir": str(CSV_PMU_CLEAN_DIR),
            "csv_pmu_realism_dir": str(CSV_PMU_REALISM_DIR),
            "csv_all39_by_bus_dir": str(CSV_ALL39_BY_BUS_DIR),
            "pmu_merged_clean_csv": str(PMU_MERGED_CLEAN_CSV),
            "pmu_merged_realism_csv": str(PMU_MERGED_REALISM_CSV),
            "all39_merged_clean_csv": str(ALL39_MERGED_CLEAN_CSV),
            "plots_pmu_clean_dir": str(PLOTS_PMU_CLEAN_DIR),
            "plots_pmu_clean_zoom_dir": str(PLOTS_PMU_CLEAN_ZOOM_DIR),
            "plots_pmu_realism_dir": str(PLOTS_PMU_REALISM_DIR),
            "plots_pmu_realism_zoom_dir": str(PLOTS_PMU_REALISM_ZOOM_DIR),
            "plots_all39_dir": str(PLOTS_ALL39_DIR),
            "plots_all39_zoom_dir": str(PLOTS_ALL39_ZOOM_DIR),
        },
    }

    save_json(YBUS_DIAG_JSON, ybus_diag)
    save_json(RUN_METADATA_JSON, metadata)
    save_json(EVENT_LABELS_JSON, EVENT_LABELS)
    save_json(PMU_MANIFEST_JSON, manifest)
    save_json(EVENT_SCHEDULE_JSON, applied_events)

    summary_lines = [
        f"Run name: {RUN_NAME}",
        f"Output root: {OUT_DIR}",
        f"PMU buses exported clean: {', '.join(str(b) for b in PMU_BUSES_IN_ORDER)}",
        f"PMU buses exported realism: {', '.join(str(b) for b in PMU_BUSES_IN_ORDER)}",
        f"All 39 buses exported clean in wide + per-bus form.",
        f"Fault simulated in ANDES at Bus {FAULT_BUS} from {FAULT_START:.3f}s to {FAULT_CLEAR:.3f}s.",
        f"Visible fault duration used in PMU labels: {fault_visible_duration:.3f} s",
        f"Selected Ybus source: {ybus_source_used}",
        f"Ybus same_matrix: {ybus_compare['same_matrix']}",
        f"Fixed merged-PMU collision by prefixing BUSxx_DATA_PRESENT and BUSxx_Event in merged CSVs.",
        f"Added high-resolution zoom plots (+/- {ZOOM_PRE_SEC:.1f}s around the fault) for each PMU bus and each metric.",
    ]
    save_text(SUMMARY_TXT, "\n".join(summary_lines) + "\n")

    print("Done.")
    for line in summary_lines:
        print(line)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
