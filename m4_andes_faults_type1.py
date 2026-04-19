from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone

import andes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import m3_andes_calibration_raw as m3
from m2_noise_profiling_raw import PROFILE_OUT, RawPMUNoiseLayer


# ============================================================================
# CONFIG
# ============================================================================

FAULT_BUS = "39"
FAULT_BUS_OPTIONS = [str(i) for i in range(1, 40)]

PRE_FAULT_S = 5.0
FAULT_START_S = 5.0
FAULT_DURATION_S = 0.10
POST_FAULT_S = 5.0

FAULT_CLEAR_S = FAULT_START_S + FAULT_DURATION_S
SIM_TF = PRE_FAULT_S + FAULT_DURATION_S + POST_FAULT_S
SIM_TSTEP = 1.0 / 30.0

FAULT_RF = 0.0
FAULT_XF = 1e-3
SYSTEM_BASE_MVA = 100.0

RUN_NAME_TEMPLATE = "SIM0001_BUS{fault_bus}_Event1"
BASE_OUTPUT_DIR = "output"

EVENT_LABEL_MODE = "windowed"   # "windowed" or "file_constant"
DATA_PRESENT_DEFAULT = 1

PMU_BUSES = ["39", "29", "10", "22", "19", "2", "5", "6"]

EVENT0_PROFILE_PATH = PROFILE_OUT
EVENT0_CURRENT_MAPPING_CSV = "output/ANDES_CALIBRATION_RAW/current_mapping_selection.csv"
EVENT0_SUPPORT_MATRIX_CSV = "output/ANDES_CALIBRATION_RAW/signal_support_matrix.csv"
EVENT0_CALIBRATION_JSON = "output/ANDES_CALIBRATION_RAW/metrics/calibration_results.json"

FAULT_DRIFT_SCALE = 1.0
FAULT_NOISE_SCALE = 1.0
FAULT_SPIKE_SCALE = 1.0

RNG_SEED = 20260418

# angle policy
DISABLE_ANGLE_NOISE = True
ANGLE_TINY_JITTER_DEG = 0.0
ANGLE_SUFFIXES = {"VA_ANG", "VB_ANG", "VC_ANG", "IA_ANG", "IB_ANG", "IC_ANG"}

# plot config
GENERATE_PLOTS = True
PLOT_INCLUDE_CLEAN_REFERENCE = True
PLOT_DPI = 140
PLOT_FIGSIZE = (11, 4.5)

PLOT_SUFFIXES = [
    "VA_ANG", "VA_MAG",
    "VB_ANG", "VB_MAG",
    "VC_ANG", "VC_MAG",
    "IA_ANG", "IA_MAG",
    "IB_ANG", "IB_MAG",
    "IC_ANG", "IC_MAG",
    "Freq", "ROCOF",
]

ESTIMATION_PLOT_SUFFIXES = PLOT_SUFFIXES.copy()

# estimator config
ESTIMATION_MODE = "ybus_temporal_regularized"  # ybus_static | ybus_temporal_regularized
ESTIMATION_REG_EPS = 1e-8
ESTIMATION_TEMPORAL_LAMBDA = 5e-2
ESTIMATION_PRIOR_LAMBDA = 1e-3
ESTIMATION_USE_POSITIVE_SEQUENCE = True
ESTIMATION_SMOOTH_VOLTAGE_WINDOW = 5
ESTIMATION_SMOOTH_CURRENT_WINDOW = 7
ESTIMATION_FREQ_CLIP = (58.5, 61.5)
ESTIMATION_ROCOF_CLIP = (-20.0, 20.0)
CURRENT_ESTIMATION_MODE = "nodal_injection_from_ybus"


# ============================================================================
# HELPERS
# ============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def wrap_deg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return ((x + 180.0) % 360.0) - 180.0


def complex_phase_deg(z: np.ndarray) -> np.ndarray:
    return wrap_deg(np.rad2deg(np.angle(np.asarray(z, dtype=complex))))


def current_run_output_dir(fault_bus: str) -> str:
    return os.path.join(BASE_OUTPUT_DIR, RUN_NAME_TEMPLATE.format(fault_bus=fault_bus))


def simulation_output_dir(fault_bus: str) -> str:
    return os.path.join(current_run_output_dir(fault_bus), "simulation")


def estimated_output_dir(fault_bus: str) -> str:
    return os.path.join(current_run_output_dir(fault_bus), "estimated")


def event_series(t: np.ndarray, mode: str = EVENT_LABEL_MODE) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if mode == "file_constant":
        return np.ones(len(t), dtype=int)
    return ((t >= FAULT_START_S) & (t <= FAULT_CLEAR_S)).astype(int)


def raw_col(bus_id: str, suffix: str) -> str:
    return f"BUS{bus_id}_{suffix}"


def current_base_amp_from_kv(kv_ll: float, system_base_mva: float = SYSTEM_BASE_MVA) -> float:
    kv_ll = float(kv_ll)
    if abs(kv_ll) < 1e-12:
        kv_ll = 345.0
    return system_base_mva * 1e6 / (np.sqrt(3.0) * kv_ll * 1e3)


def voltage_base_phase_volts_from_kv(kv_ll: float) -> float:
    kv_ll = float(kv_ll)
    if abs(kv_ll) < 1e-12:
        kv_ll = 345.0
    return kv_ll * 1e3 / np.sqrt(3.0)


def suffix_to_family(suffix: str) -> str:
    if suffix in {"VA_MAG", "VB_MAG", "VC_MAG"}:
        return "voltage_mag"
    if suffix in {"IA_MAG", "IB_MAG", "IC_MAG"}:
        return "current_mag"
    if suffix == "Freq":
        return "frequency"
    if suffix == "ROCOF":
        return "rocof"
    if suffix in {"VA_ANG", "VB_ANG", "VC_ANG"}:
        return "angle_voltage"
    if suffix in {"IA_ANG", "IB_ANG", "IC_ANG"}:
        return "angle_current"
    return "other"


def robust_rocof(freq_hz: np.ndarray, t: np.ndarray) -> np.ndarray:
    freq_hz = np.asarray(freq_hz, dtype=float)
    t = np.asarray(t, dtype=float)

    if len(freq_hz) < 3:
        return np.zeros_like(freq_hz)

    rocof = np.gradient(freq_hz, t)
    rocof = pd.Series(rocof).rolling(window=5, center=True, min_periods=1).median().to_numpy(dtype=float)
    rocof = np.clip(rocof, ESTIMATION_ROCOF_CLIP[0], ESTIMATION_ROCOF_CLIP[1])
    return rocof


def ensure_fault_bus_valid(fault_bus: str) -> str:
    fault_bus = str(fault_bus)
    if fault_bus not in FAULT_BUS_OPTIONS:
        raise ValueError(f"FAULT_BUS must be in 1..39, got {fault_bus!r}")
    return fault_bus


def is_angle_suffix(raw_suffix: str) -> bool:
    return raw_suffix in ANGLE_SUFFIXES or str(raw_suffix).endswith("_ANG")


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.mean(np.abs(a - b)))


def rel_rmse(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.std(a) + 1e-12)
    return float(rmse(a, b) / denom)


def angle_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return wrap_deg(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))


def rolling_complex_mean(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=complex)
    if window <= 1 or len(x) == 0:
        return x.copy()
    real = pd.Series(np.real(x)).rolling(window=window, center=True, min_periods=1).mean().to_numpy(float)
    imag = pd.Series(np.imag(x)).rolling(window=window, center=True, min_periods=1).mean().to_numpy(float)
    return real + 1j * imag


def positive_sequence_from_abc(va: complex, vb: complex, vc: complex) -> complex:
    a = np.exp(1j * 2.0 * np.pi / 3.0)
    return (va + a * vb + (a ** 2) * vc) / 3.0


# ============================================================================
# LOAD EVENT-0 ARTIFACTS
# ============================================================================

def load_csv_if_exists(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Could not read CSV {path}: {exc}")
    return None


def load_json_if_exists(path: str) -> dict | None:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            print(f"[WARN] Could not read JSON {path}: {exc}")
    return None


def aggregate_family_defaults_from_profiles(noise_layer: RawPMUNoiseLayer) -> dict:
    families = {
        "voltage_mag": [],
        "current_mag": [],
        "frequency": [],
        "rocof": [],
        "angle_voltage": [],
        "angle_current": [],
    }

    profiles = getattr(noise_layer, "profiles", {}) or {}
    evt0 = profiles.get("0", {}) if isinstance(profiles, dict) else {}

    for bus_id, bus_payload in evt0.items():
        if not isinstance(bus_payload, dict):
            continue
        for raw_name, profile_obj in bus_payload.items():
            profile = profile_obj if isinstance(profile_obj, dict) else {}
            family = None

            name_upper = str(raw_name).upper()
            if name_upper.endswith(("VA_MAG", "VB_MAG", "VC_MAG")):
                family = "voltage_mag"
            elif name_upper.endswith(("IA_MAG", "IB_MAG", "IC_MAG")):
                family = "current_mag"
            elif name_upper.endswith("FREQ"):
                family = "frequency"
            elif name_upper.endswith("ROCOF"):
                family = "rocof"
            elif name_upper.endswith(("VA_ANG", "VB_ANG", "VC_ANG")):
                family = "angle_voltage"
            elif name_upper.endswith(("IA_ANG", "IB_ANG", "IC_ANG")):
                family = "angle_current"

            if family is not None:
                families[family].append(profile)

    out = {}
    for family, items in families.items():
        if not items:
            out[family] = {}
            continue

        stds = []
        outliers = []
        rhos = []
        meds = []
        fitted = []
        residual_samples = []

        for p in items:
            stds.append(float(p.get("std_dev_abs", p.get("std_dev_raw", 0.0))))
            outliers.append(float(p.get("outlier_mag_abs", 0.0)))
            rhos.append(float(p.get("ar1_rho", 0.0)))

            eda = p.get("eda_stats", {}) or {}
            meds.append(float(eda.get("median", eda.get("p50", 0.0))))

            fitted.append(str(p.get("fitted_distribution_type", "gaussian")))
            rs = np.asarray(p.get("residual_samples", []), dtype=float)
            rs = rs[np.isfinite(rs)]
            if len(rs):
                residual_samples.append(rs)

        out[family] = {
            "std_dev_abs": float(np.median(stds)) if stds else 0.0,
            "outlier_mag_abs": float(np.median(outliers)) if outliers else 0.0,
            "ar1_rho": float(np.median(rhos)) if rhos else 0.0,
            "fitted_distribution_type": max(set(fitted), key=fitted.count) if fitted else "gaussian",
            "eda_stats": {
                "median": float(np.median(meds)) if meds else 0.0,
                "std": float(np.median(stds)) if stds else 0.0,
            },
            "residual_samples": np.concatenate(residual_samples).tolist()[:5000] if residual_samples else [],
            "profile_source": "family_default_from_event0_pmuses",
        }
    return out


def load_event0_artifacts() -> dict:
    artifacts = {
        "noise_layer": None,
        "family_defaults": {},
        "current_mappings": {},
        "support": {},
        "aggregate_records": {},
    }

    try:
        noise_layer = RawPMUNoiseLayer(EVENT0_PROFILE_PATH)
        artifacts["noise_layer"] = noise_layer
        artifacts["family_defaults"] = aggregate_family_defaults_from_profiles(noise_layer)
    except Exception as exc:
        print(f"[WARN] Could not load RawPMUNoiseLayer: {exc}")

    current_map_df = load_csv_if_exists(EVENT0_CURRENT_MAPPING_CSV)
    if current_map_df is not None and not current_map_df.empty and "bus_id" in current_map_df.columns:
        for _, row in current_map_df.iterrows():
            artifacts["current_mappings"][str(row["bus_id"])] = str(row.get("chosen_mapping", "bus_injection_current"))

    support_df = load_csv_if_exists(EVENT0_SUPPORT_MATRIX_CSV)
    if support_df is not None and not support_df.empty and "signal_name" in support_df.columns:
        for _, row in support_df.iterrows():
            artifacts["support"][str(row["signal_name"])] = row.to_dict()

    cal_json = load_json_if_exists(EVENT0_CALIBRATION_JSON)
    if cal_json is not None:
        for rec in cal_json.get("records", []):
            if rec.get("scope") == "aggregate":
                artifacts["aggregate_records"][(str(rec.get("bus_id")), str(rec.get("signal_key")))] = rec

    return artifacts


def resolve_profile_for_bus_signal(
    artifacts: dict,
    bus_id: str,
    raw_suffix: str,
) -> tuple[dict | None, str]:
    noise_layer = artifacts.get("noise_layer")
    if noise_layer is not None:
        raw_name = raw_col(bus_id, raw_suffix)
        try:
            profile, status = noise_layer.resolve_stats(bus_id, 0, raw_name)
            if profile is not None:
                return profile, f"exact_or_fallback:{status}"
        except Exception:
            pass

    family = suffix_to_family(raw_suffix)
    family_profile = artifacts.get("family_defaults", {}).get(family, {})
    if family_profile:
        return family_profile, "family_default"

    return None, "no_profile"


# ============================================================================
# ANDES FAULT SIMULATION
# ============================================================================

def build_fault_scenario(system, fault_bus: str):
    return system.add(
        "Fault",
        bus=int(fault_bus),
        tf=float(FAULT_START_S),
        tc=float(FAULT_CLEAR_S),
        rf=float(FAULT_RF),
        xf=float(FAULT_XF),
    )


def run_fault_simulation_andes(fault_bus: str):
    fault_bus = ensure_fault_bus_valid(fault_bus)

    system = andes.load(
        andes.get_case("ieee39/ieee39_full.xlsx"),
        setup=False,
        no_output=True,
    )

    build_fault_scenario(system, fault_bus)
    system.setup()

    print(f"[INFO] Running ANDES PFlow with fault at bus {fault_bus}...")
    ok_pf = system.PFlow.run()
    if ok_pf is False:
        raise RuntimeError("ANDES PFlow failed")

    system.TDS.config.tf = SIM_TF
    system.TDS.config.tstep = SIM_TSTEP
    if hasattr(system.TDS.config, "criteria"):
        system.TDS.config.criteria = 0

    print(
        f"[INFO] Running TDS: pre={PRE_FAULT_S:.3f}s, "
        f"fault={FAULT_DURATION_S:.3f}s, post={POST_FAULT_S:.3f}s, bus={fault_bus}"
    )
    ok_tds = system.TDS.run()
    if ok_tds is False:
        print("[WARN] ANDES TDS returned False; continuing with available trajectory if any.")

    return system


# ============================================================================
# NETWORK EXTRACTION
# ============================================================================

def get_bus_kv_map(system) -> dict[str, float]:
    out = {}
    try:
        for idx, kv in zip(system.Bus.idx.v, system.Bus.Vn.v):
            out[str(idx)] = float(kv)
    except Exception:
        for idx in system.Bus.idx.v:
            out[str(idx)] = 345.0
    return out


def build_ybus_from_lines(system):
    bus_ids = [str(b) for b in system.Bus.idx.v]
    bus_pos = {bus: i for i, bus in enumerate(bus_ids)}
    ybus = np.zeros((len(bus_ids), len(bus_ids)), dtype=complex)

    if not hasattr(system, "Line"):
        return None, bus_pos

    lines = system.Line
    if getattr(lines, "n", 0) == 0:
        return None, bus_pos

    def _v(param, i, default=0.0):
        try:
            return float(param.v[i])
        except Exception:
            return default

    try:
        for k in range(lines.n):
            fb = str(lines.bus1.v[k])
            tb = str(lines.bus2.v[k])
            if fb not in bus_pos or tb not in bus_pos:
                continue

            fi = bus_pos[fb]
            ti = bus_pos[tb]

            r = _v(lines.r, k, 0.0)
            x = _v(lines.x, k, 0.0)
            g = _v(lines.g, k, 0.0) if hasattr(lines, "g") else 0.0
            b = _v(lines.b, k, 0.0) if hasattr(lines, "b") else 0.0
            tap = _v(lines.tap, k, 1.0) if hasattr(lines, "tap") else 1.0
            phi = _v(lines.phi, k, 0.0) if hasattr(lines, "phi") else 0.0

            if abs(tap) < 1e-12:
                tap = 1.0

            if abs(r) + abs(x) <= 1e-12:
                continue

            y = 1.0 / complex(r, x)
            ysh = 0.5 * complex(g, b)
            tc = tap * np.exp(1j * phi)

            ybus[fi, fi] += (y + ysh) / (abs(tc) ** 2)
            ybus[ti, ti] += y + ysh
            ybus[fi, ti] -= y / np.conj(tc)
            ybus[ti, fi] -= y / tc

        return ybus, bus_pos
    except Exception as exc:
        print(f"[WARN] Failed to build Ybus: {exc}")
        return None, bus_pos


def build_fault_shunt_current(system, t, v_df, a_df, fault_bus: str, kv_map: dict[str, float]) -> dict[str, np.ndarray]:
    if fault_bus not in v_df.columns or fault_bus not in a_df.columns:
        return {}

    zf = complex(FAULT_RF, FAULT_XF)
    if abs(zf) < 1e-12:
        zf = complex(0.0, 1e-4)

    v_complex_pu = v_df[fault_bus].to_numpy(float) * np.exp(1j * a_df[fault_bus].to_numpy(float))
    mask = (np.asarray(t) >= FAULT_START_S) & (np.asarray(t) <= FAULT_CLEAR_S)
    i_pu = np.zeros(len(t), dtype=complex)
    i_pu[mask] = v_complex_pu[mask] / zf

    kv_ll = kv_map.get(fault_bus, 345.0)
    i_base_amp = current_base_amp_from_kv(kv_ll)
    return {fault_bus: i_pu * i_base_amp}


def extract_all_bus_signals(system, fault_bus: str) -> tuple[dict[str, dict], dict]:
    t, v_df, a_df = m3.get_timeseries(system)
    v_df.columns = [str(c) for c in v_df.columns]
    a_df.columns = [str(c) for c in a_df.columns]

    kv_map = get_bus_kv_map(system)
    ybus, bus_pos = build_ybus_from_lines(system)

    bus_ids = [str(b) for b in system.Bus.idx.v]
    bus_ids_common = [b for b in bus_ids if b in v_df.columns and b in a_df.columns]

    current_complex_map = {}
    if ybus is not None:
        v_complex_pu = v_df[bus_ids_common].to_numpy(dtype=float) * np.exp(1j * a_df[bus_ids_common].to_numpy(dtype=float))
        idx = [bus_pos[b] for b in bus_ids_common]
        ysub = ybus[np.ix_(idx, idx)]
        i_complex_pu = (ysub @ v_complex_pu.T).T
        for i, bus_id in enumerate(bus_ids_common):
            kv_ll = kv_map.get(bus_id, 345.0)
            i_base_amp = current_base_amp_from_kv(kv_ll)
            current_complex_map[bus_id] = i_complex_pu[:, i] * i_base_amp

    fault_shunt_map = build_fault_shunt_current(system, t, v_df, a_df, fault_bus, kv_map)

    signals_by_bus = {}
    for bus_id in bus_ids_common:
        kv_ll = kv_map.get(bus_id, 345.0)

        v_pu = v_df[bus_id].to_numpy(dtype=float)
        angle_rad = a_df[bus_id].to_numpy(dtype=float)
        angle_deg = np.rad2deg(angle_rad)

        v_phase_volts = v_pu * kv_ll * 1e3 / np.sqrt(3.0)

        freq = 60.0 + np.gradient(np.unwrap(angle_rad), t) / (2.0 * np.pi)
        freq = pd.Series(freq).rolling(window=5, center=True, min_periods=1).median().to_numpy(dtype=float)
        freq = np.clip(freq, ESTIMATION_FREQ_CLIP[0], ESTIMATION_FREQ_CLIP[1])
        rocof = robust_rocof(freq, t)

        i_complex = current_complex_map.get(bus_id, None)
        if i_complex is None:
            i_mag = np.zeros_like(v_phase_volts)
            i_ang = wrap_deg(angle_deg.copy())
        else:
            if bus_id in fault_shunt_map:
                i_complex = i_complex + fault_shunt_map[bus_id]
            i_mag = np.abs(i_complex)
            i_ang = wrap_deg(np.rad2deg(np.angle(i_complex)))

        signals_by_bus[bus_id] = {
            "t": np.asarray(t, dtype=float),

            "VA_MAG": v_phase_volts.copy(),
            "VB_MAG": v_phase_volts.copy(),
            "VC_MAG": v_phase_volts.copy(),

            "VA_ANG": wrap_deg(angle_deg),
            "VB_ANG": wrap_deg(angle_deg - 120.0),
            "VC_ANG": wrap_deg(angle_deg + 120.0),

            "IA_MAG": i_mag.copy(),
            "IB_MAG": i_mag.copy(),
            "IC_MAG": i_mag.copy(),

            "IA_ANG": wrap_deg(i_ang),
            "IB_ANG": wrap_deg(i_ang - 120.0),
            "IC_ANG": wrap_deg(i_ang + 120.0),

            "Freq": freq,
            "ROCOF": rocof,
        }

    meta = {
        "t": np.asarray(t, dtype=float),
        "v_df": v_df,
        "a_df": a_df,
        "kv_map": kv_map,
        "ybus": ybus,
        "bus_pos": bus_pos,
        "bus_ids_common": bus_ids_common,
    }
    return signals_by_bus, meta


# ============================================================================
# APPLY EVENT-0 MEASUREMENT MODEL
# ============================================================================

def calibration_record_for(bus_id: str, raw_suffix: str, artifacts: dict) -> dict:
    signal_key_map = {
        "VA_MAG": "VA_mag",
        "VB_MAG": "VB_mag",
        "VC_MAG": "VC_mag",
        "IA_MAG": "IA_mag",
        "IB_MAG": "IB_mag",
        "IC_MAG": "IC_mag",
        "Freq": "Frequency",
        "ROCOF": "ROCOF",
    }
    key = signal_key_map.get(raw_suffix, None)
    if key is None:
        return {}
    return artifacts.get("aggregate_records", {}).get((str(bus_id), key), {})


def profile_median(profile: dict | None, default: float = 0.0) -> float:
    if not profile:
        return float(default)
    eda = profile.get("eda_stats", {}) or {}
    return float(eda.get("median", eda.get("p50", default)))


def profile_std(profile: dict | None, default: float = 0.0) -> float:
    if not profile:
        return float(default)
    eda = profile.get("eda_stats", {}) or {}
    return float(eda.get("std", default))


def convert_clean_to_event0_center(clean_arr: np.ndarray, raw_suffix: str, profile: dict | None) -> tuple[np.ndarray, str]:
    clean_arr = np.asarray(clean_arr, dtype=float)

    if raw_suffix in {"VA_MAG", "VB_MAG", "VC_MAG", "IA_MAG", "IB_MAG", "IC_MAG", "Freq", "ROCOF"}:
        if profile:
            med_target = profile_median(profile, default=np.median(clean_arr))
            med_clean = float(np.median(clean_arr))
            if abs(med_clean) > 1e-12:
                if raw_suffix in {"VA_MAG", "VB_MAG", "VC_MAG", "IA_MAG", "IB_MAG", "IC_MAG"}:
                    return clean_arr * (med_target / med_clean), "median_scale_to_event0"
                else:
                    return clean_arr + (med_target - med_clean), "median_shift_to_event0"
        return clean_arr, "physical_units_no_profile"

    return wrap_deg(clean_arr), "angle_clean_only"


def event0_drift_targets_from_profile(profile: dict | None) -> dict:
    if not profile:
        return {
            "total_std": 0.0,
            "noise_std": 0.0,
            "drift_std": 0.0,
            "empirical_deviation": np.array([]),
            "spectral": {},
        }

    total_std = profile_std(profile, default=0.0)
    noise_std = float(profile.get("std_dev_abs", profile.get("std_dev_raw", 0.0)))
    drift_std = math.sqrt(max(total_std**2 - min(noise_std, 0.55 * total_std) ** 2, 0.0))
    drift_std = max(drift_std, 0.25 * total_std) if total_std > 0 else 0.0

    residual = np.asarray(profile.get("residual_samples", []), dtype=float)
    residual = residual[np.isfinite(residual)]

    return {
        "total_std": float(total_std),
        "noise_std": float(noise_std),
        "drift_std": float(drift_std),
        "empirical_deviation": residual,
        "spectral": profile.get("psd_summary", {}) or profile.get("fft_low_freq_summary", {}) or {},
    }


def scaled_noise_profile(profile: dict | None) -> dict:
    if not profile:
        return {}
    out = dict(profile)
    out["outlier_mag_abs"] = float(out.get("outlier_mag_abs", 0.0)) * FAULT_SPIKE_SCALE
    return out


def apply_event0_measurement_model(
    bus_id: str,
    raw_suffix: str,
    clean_arr: np.ndarray,
    t: np.ndarray,
    artifacts: dict,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    clean_arr = np.asarray(clean_arr, dtype=float)
    profile, profile_status = resolve_profile_for_bus_signal(artifacts, bus_id, raw_suffix)

    if DISABLE_ANGLE_NOISE and is_angle_suffix(raw_suffix):
        clean_wrapped = wrap_deg(clean_arr)
        if ANGLE_TINY_JITTER_DEG > 0.0:
            jitter = rng.normal(0.0, ANGLE_TINY_JITTER_DEG, size=len(clean_wrapped))
            noisy = wrap_deg(clean_wrapped + jitter)
        else:
            noisy = clean_wrapped.copy()
        return clean_wrapped, noisy, "angle_clean_only", "disabled_for_angles"

    family = suffix_to_family(raw_suffix)
    cal_record = calibration_record_for(bus_id, raw_suffix, artifacts)

    if profile is None:
        return clean_arr, clean_arr.copy(), profile_status, "identity_no_profile"

    drift_scale = float(cal_record.get("drift_scale", 1.0)) * FAULT_DRIFT_SCALE if cal_record else FAULT_DRIFT_SCALE
    noise_scale = float(cal_record.get("noise_scale", 1.0)) * FAULT_NOISE_SCALE if cal_record else FAULT_NOISE_SCALE
    noise_model = str(cal_record.get("noise_model_used", "")) if cal_record else ""

    if not noise_model or noise_model == "nan":
        noise_model = m3.fit_residual_distribution(profile, family, "auto")

    targets = event0_drift_targets_from_profile(profile)
    drifted = m3.apply_operating_drift(
        clean_arr,
        t,
        family,
        bus_id,
        targets,
        rng,
        drift_scale=drift_scale,
    )

    noise = m3.sample_residual_process(
        scaled_noise_profile(profile),
        len(clean_arr),
        rng,
        noise_model,
        noise_scale=noise_scale,
    )

    noisy = drifted + noise

    if raw_suffix == "Freq":
        noisy = np.clip(noisy, ESTIMATION_FREQ_CLIP[0], ESTIMATION_FREQ_CLIP[1])
    elif raw_suffix == "ROCOF":
        noisy = np.clip(noisy, ESTIMATION_ROCOF_CLIP[0], ESTIMATION_ROCOF_CLIP[1])

    return clean_arr, noisy, profile_status, noise_model


# ============================================================================
# PLOTS
# ============================================================================

def spans_from_event(t: np.ndarray, event: np.ndarray) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    t = np.asarray(t, dtype=float)
    event = np.asarray(event, dtype=int)

    if len(t) == 0:
        return [], []

    spans0 = []
    spans1 = []

    start_idx = 0
    current_val = event[0]

    for i in range(1, len(event)):
        if event[i] != current_val:
            span = (float(t[start_idx]), float(t[i - 1]))
            if current_val == 0:
                spans0.append(span)
            else:
                spans1.append(span)
            start_idx = i
            current_val = event[i]

    span = (float(t[start_idx]), float(t[-1]))
    if current_val == 0:
        spans0.append(span)
    else:
        spans1.append(span)

    return spans0, spans1


def add_event_spans(ax, t: np.ndarray, event: np.ndarray):
    spans0, spans1 = spans_from_event(t, event)

    first0 = True
    first1 = True

    for a, b in spans0:
        ax.axvspan(a, b, alpha=0.08, label="Event 0" if first0 else None)
        first0 = False

    for a, b in spans1:
        ax.axvspan(a, b, alpha=0.18, label="Event 1" if first1 else None)
        first1 = False


def plot_one_signal(
    bus_id: str,
    suffix: str,
    t: np.ndarray,
    clean_arr: np.ndarray,
    noisy_arr: np.ndarray,
    event_arr: np.ndarray,
    out_dir: str,
):
    ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)

    add_event_spans(ax, t, event_arr)

    if PLOT_INCLUDE_CLEAN_REFERENCE:
        ax.plot(t, clean_arr, linewidth=1.0, alpha=0.75, label="clean/reference")

    ax.plot(t, noisy_arr, linewidth=1.0, label="simulated")

    ax.axvline(FAULT_START_S, linestyle="--", linewidth=1.0, label="fault start")
    ax.axvline(FAULT_CLEAR_S, linestyle="--", linewidth=1.0, label="fault clear")

    ax.set_title(f"BUS{bus_id} - {suffix}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(suffix)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"BUS{bus_id}_{suffix}.png"), dpi=PLOT_DPI)
    plt.close(fig)


def generate_bus_plots(
    bus_id: str,
    t: np.ndarray,
    event_arr: np.ndarray,
    clean_map: dict[str, np.ndarray],
    noisy_map: dict[str, np.ndarray],
    base_out_dir: str,
):
    bus_plot_dir = os.path.join(base_out_dir, "plots", f"BUS{bus_id}")
    ensure_dir(bus_plot_dir)

    for suffix in PLOT_SUFFIXES:
        if suffix not in clean_map or suffix not in noisy_map:
            continue
        plot_one_signal(
            bus_id=bus_id,
            suffix=suffix,
            t=t,
            clean_arr=clean_map[suffix],
            noisy_arr=noisy_map[suffix],
            event_arr=event_arr,
            out_dir=bus_plot_dir,
        )


def plot_estimated_signal(
    bus_id: str,
    suffix: str,
    t: np.ndarray,
    sim_arr: np.ndarray,
    est_arr: np.ndarray,
    event_arr: np.ndarray,
    out_dir: str,
):
    ensure_dir(out_dir)

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    add_event_spans(ax, t, event_arr)

    ax.plot(t, sim_arr, linewidth=1.0, alpha=0.85, label="simulation")
    ax.plot(t, est_arr, linewidth=1.0, label="estimated")

    ax.axvline(FAULT_START_S, linestyle="--", linewidth=1.0, label="fault start")
    ax.axvline(FAULT_CLEAR_S, linestyle="--", linewidth=1.0, label="fault clear")

    ax.set_title(f"BUS{bus_id} - {suffix} (simulation vs estimated)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(suffix)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"BUS{bus_id}_{suffix}.png"), dpi=PLOT_DPI)
    plt.close(fig)


def generate_estimated_bus_plots(
    bus_id: str,
    t: np.ndarray,
    event_arr: np.ndarray,
    simulation_df: pd.DataFrame,
    estimated_df: pd.DataFrame,
    base_out_dir: str,
):
    bus_plot_dir = os.path.join(base_out_dir, "plots", f"BUS{bus_id}")
    ensure_dir(bus_plot_dir)

    for suffix in ESTIMATION_PLOT_SUFFIXES:
        sim_col = raw_col(bus_id, suffix)
        est_col = raw_col(bus_id, suffix)
        if sim_col not in simulation_df.columns or est_col not in estimated_df.columns:
            continue

        plot_estimated_signal(
            bus_id=bus_id,
            suffix=suffix,
            t=t,
            sim_arr=simulation_df[sim_col].to_numpy(dtype=float),
            est_arr=estimated_df[est_col].to_numpy(dtype=float),
            event_arr=event_arr,
            out_dir=bus_plot_dir,
        )


# ============================================================================
# BUILD SIMULATION DATAFRAMES
# ============================================================================

def build_competition_dataframe_for_bus(
    bus_id: str,
    bus_signals: dict,
    artifacts: dict,
    rng: np.random.Generator,
):
    t = np.asarray(bus_signals["t"], dtype=float)
    ev = event_series(t, EVENT_LABEL_MODE)

    suffixes = [
        "VA_ANG", "VA_MAG",
        "VB_ANG", "VB_MAG",
        "VC_ANG", "VC_MAG",
        "IA_ANG", "IA_MAG",
        "IB_ANG", "IB_MAG",
        "IC_ANG", "IC_MAG",
        "Freq", "ROCOF",
    ]

    data = {"TIMESTAMP": t}
    clean_map = {}
    noisy_map = {}

    for suffix in suffixes:
        clean_arr = np.asarray(bus_signals[suffix], dtype=float)

        profile, _ = resolve_profile_for_bus_signal(artifacts, bus_id, suffix)
        clean_centered, _ = convert_clean_to_event0_center(clean_arr, suffix, profile)

        clean_plot, noisy, _, _ = apply_event0_measurement_model(
            bus_id=bus_id,
            raw_suffix=suffix,
            clean_arr=clean_centered,
            t=t,
            artifacts=artifacts,
            rng=rng,
        )

        if suffix.endswith("_ANG"):
            clean_plot = wrap_deg(clean_plot)
            noisy = wrap_deg(noisy)

        clean_plot = np.asarray(clean_plot, dtype=float)
        noisy = np.asarray(noisy, dtype=float)

        if np.any(~np.isfinite(noisy)):
            noisy = (
                pd.Series(noisy)
                .interpolate(limit_direction="both")
                .bfill()
                .ffill()
                .to_numpy(dtype=float)
            )

        if np.any(~np.isfinite(clean_plot)):
            clean_plot = (
                pd.Series(clean_plot)
                .interpolate(limit_direction="both")
                .bfill()
                .ffill()
                .to_numpy(dtype=float)
            )

        data[raw_col(bus_id, suffix)] = noisy
        clean_map[suffix] = clean_plot
        noisy_map[suffix] = noisy

    data["DATA_PRESENT"] = np.full(len(t), DATA_PRESENT_DEFAULT, dtype=int)
    data["Event"] = ev

    return pd.DataFrame(data), clean_map, noisy_map


# ============================================================================
# EXPORT SIMULATION
# ============================================================================

def export_simulation_outputs(
    signals_by_bus: dict[str, dict],
    artifacts: dict,
    fault_bus: str,
) -> tuple[str, dict[str, pd.DataFrame]]:
    out_dir = simulation_output_dir(fault_bus)
    ensure_dir(out_dir)

    simulation_dfs: dict[str, pd.DataFrame] = {}

    for bus_id in sorted(signals_by_bus.keys(), key=lambda x: int(x)):
        rng = np.random.default_rng(RNG_SEED + int(bus_id))
        df, clean_map, noisy_map = build_competition_dataframe_for_bus(
            bus_id, signals_by_bus[bus_id], artifacts, rng
        )

        simulation_dfs[bus_id] = df.copy()

        out_path = os.path.join(out_dir, f"BUS{bus_id}_Competition_Data_nanmask.csv")
        df.to_csv(out_path, index=False)

        if GENERATE_PLOTS:
            generate_bus_plots(
                bus_id=bus_id,
                t=df["TIMESTAMP"].to_numpy(dtype=float),
                event_arr=df["Event"].to_numpy(dtype=int),
                clean_map=clean_map,
                noisy_map=noisy_map,
                base_out_dir=out_dir,
            )

    return out_dir, simulation_dfs


# ============================================================================
# ESTIMATION USING ONLY 8 PMUS + YBUS
# ============================================================================

def build_observed_positive_sequence_vector(
    simulation_dfs: dict[str, pd.DataFrame],
    kv_map: dict[str, float],
    pmu_available: list[str],
    ti: int,
) -> np.ndarray:
    vk = []
    for bus_id in pmu_available:
        df = simulation_dfs[bus_id]

        va = float(df.iloc[ti][raw_col(bus_id, "VA_MAG")]) * np.exp(1j * np.deg2rad(float(df.iloc[ti][raw_col(bus_id, "VA_ANG")])))
        vb = float(df.iloc[ti][raw_col(bus_id, "VB_MAG")]) * np.exp(1j * np.deg2rad(float(df.iloc[ti][raw_col(bus_id, "VB_ANG")])))
        vc = float(df.iloc[ti][raw_col(bus_id, "VC_MAG")]) * np.exp(1j * np.deg2rad(float(df.iloc[ti][raw_col(bus_id, "VC_ANG")])))

        v_complex_phase = positive_sequence_from_abc(va, vb, vc) if ESTIMATION_USE_POSITIVE_SEQUENCE else va
        v_base = voltage_base_phase_volts_from_kv(kv_map.get(bus_id, 345.0))
        vk.append(v_complex_phase / v_base)

    return np.asarray(vk, dtype=complex)


def solve_unknown_voltages_static(
    Yuu: np.ndarray,
    Yuk: np.ndarray,
    Vk: np.ndarray,
) -> np.ndarray:
    rhs = -Yuk @ Vk
    Yuu_reg = Yuu + ESTIMATION_REG_EPS * np.eye(len(Yuu), dtype=complex)
    try:
        return np.linalg.solve(Yuu_reg, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(Yuu_reg, rhs, rcond=None)[0]


def solve_unknown_voltages_temporal(
    Yuu: np.ndarray,
    Yuk: np.ndarray,
    Vk: np.ndarray,
    Vu_prev: np.ndarray | None,
    Vu_prior: np.ndarray | None,
    lambda_time: float,
    lambda_prior: float,
) -> np.ndarray:
    A = Yuu
    b = -Yuk @ Vk
    n = A.shape[1]

    H = A.conj().T @ A + (ESTIMATION_REG_EPS + lambda_time + lambda_prior) * np.eye(n, dtype=complex)
    rhs = A.conj().T @ b

    if Vu_prev is not None:
        rhs = rhs + lambda_time * Vu_prev
    if Vu_prior is not None:
        rhs = rhs + lambda_prior * Vu_prior

    try:
        return np.linalg.solve(H, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(H, rhs, rcond=None)[0]


def smooth_estimated_voltage_matrix(V_est: np.ndarray) -> np.ndarray:
    if ESTIMATION_SMOOTH_VOLTAGE_WINDOW <= 1:
        return V_est
    out = np.zeros_like(V_est, dtype=complex)
    for bi in range(V_est.shape[1]):
        out[:, bi] = rolling_complex_mean(V_est[:, bi], ESTIMATION_SMOOTH_VOLTAGE_WINDOW)
    return out


def build_estimated_voltage_matrix_from_pmuses(
    simulation_dfs: dict[str, pd.DataFrame],
    meta: dict,
) -> np.ndarray:
    ybus = meta["ybus"]
    kv_map = meta["kv_map"]
    bus_ids = meta["bus_ids_common"]
    bus_pos = {b: i for i, b in enumerate(bus_ids)}
    t = meta["t"]

    if ybus is None:
        raise RuntimeError("YBUS is not available; cannot run YBUS-based estimator.")

    pmu_available = [b for b in PMU_BUSES if b in simulation_dfs and b in bus_pos]
    unknown = [b for b in bus_ids if b not in pmu_available]

    if not pmu_available:
        raise RuntimeError("No PMU buses available for estimation.")

    K = [bus_pos[b] for b in pmu_available]
    U = [bus_pos[b] for b in unknown]

    V_est = np.zeros((len(t), len(bus_ids)), dtype=complex)

    for ti in range(len(t)):
        Vk = build_observed_positive_sequence_vector(simulation_dfs, kv_map, pmu_available, ti)
        for j, b in enumerate(pmu_available):
            V_est[ti, bus_pos[b]] = Vk[j]

    if len(U) == 0:
        return smooth_estimated_voltage_matrix(V_est)

    Yuu = ybus[np.ix_(U, U)]
    Yuk = ybus[np.ix_(U, K)]

    Vu_prev = None
    for ti in range(len(t)):
        Vk = V_est[ti, K]
        Vu_static = solve_unknown_voltages_static(Yuu, Yuk, Vk)

        if ESTIMATION_MODE == "ybus_temporal_regularized":
            Vu = solve_unknown_voltages_temporal(
                Yuu=Yuu,
                Yuk=Yuk,
                Vk=Vk,
                Vu_prev=Vu_prev,
                Vu_prior=Vu_static,
                lambda_time=ESTIMATION_TEMPORAL_LAMBDA,
                lambda_prior=ESTIMATION_PRIOR_LAMBDA,
            )
        else:
            Vu = Vu_static

        for j, b in enumerate(unknown):
            V_est[ti, bus_pos[b]] = Vu[j]
        Vu_prev = Vu.copy()

    return smooth_estimated_voltage_matrix(V_est)


def compute_estimated_currents_from_voltage(
    V_est: np.ndarray,
    ybus: np.ndarray,
) -> np.ndarray:
    I_est = np.zeros_like(V_est, dtype=complex)
    for ti in range(V_est.shape[0]):
        I_est[ti, :] = ybus @ V_est[ti, :]

    if ESTIMATION_SMOOTH_CURRENT_WINDOW > 1:
        for bi in range(I_est.shape[1]):
            I_est[:, bi] = rolling_complex_mean(I_est[:, bi], ESTIMATION_SMOOTH_CURRENT_WINDOW)

    return I_est


def build_estimated_bus_dataframes(
    simulation_dfs: dict[str, pd.DataFrame],
    meta: dict,
    fault_bus: str,
) -> dict[str, pd.DataFrame]:
    t = meta["t"]
    kv_map = meta["kv_map"]
    ybus = meta["ybus"]
    bus_ids = meta["bus_ids_common"]
    bus_pos = {b: i for i, b in enumerate(bus_ids)}

    V_est = build_estimated_voltage_matrix_from_pmuses(simulation_dfs, meta)

    estimated_dfs: dict[str, pd.DataFrame] = {}

    if ybus is None:
        raise RuntimeError("YBUS missing; cannot compute estimated currents.")

    I_est = compute_estimated_currents_from_voltage(V_est, ybus)

    est_angles_rad = np.unwrap(np.angle(V_est), axis=0)
    est_freq = np.zeros_like(est_angles_rad, dtype=float)
    for bi in range(est_angles_rad.shape[1]):
        est_freq[:, bi] = 60.0 + np.gradient(est_angles_rad[:, bi], t) / (2.0 * np.pi)
        est_freq[:, bi] = (
            pd.Series(est_freq[:, bi])
            .rolling(window=5, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )
        est_freq[:, bi] = np.clip(est_freq[:, bi], ESTIMATION_FREQ_CLIP[0], ESTIMATION_FREQ_CLIP[1])

    est_rocof = np.zeros_like(est_freq, dtype=float)
    for bi in range(est_freq.shape[1]):
        est_rocof[:, bi] = robust_rocof(est_freq[:, bi], t)

    ev = event_series(t, EVENT_LABEL_MODE)

    for bus_id in bus_ids:
        if bus_id in PMU_BUSES and bus_id in simulation_dfs:
            estimated_dfs[bus_id] = simulation_dfs[bus_id].copy()
            continue

        bi = bus_pos[bus_id]
        kv_ll = kv_map.get(bus_id, 345.0)
        v_base = voltage_base_phase_volts_from_kv(kv_ll)
        i_base = current_base_amp_from_kv(kv_ll)

        v_complex = V_est[:, bi]
        i_complex = I_est[:, bi]

        va_mag = np.abs(v_complex) * v_base
        va_ang = complex_phase_deg(v_complex)

        ia_mag = np.abs(i_complex) * i_base
        ia_ang = complex_phase_deg(i_complex)

        df = pd.DataFrame({
            "TIMESTAMP": t,

            raw_col(bus_id, "VA_ANG"): va_ang,
            raw_col(bus_id, "VA_MAG"): va_mag,
            raw_col(bus_id, "VB_ANG"): wrap_deg(va_ang - 120.0),
            raw_col(bus_id, "VB_MAG"): va_mag.copy(),
            raw_col(bus_id, "VC_ANG"): wrap_deg(va_ang + 120.0),
            raw_col(bus_id, "VC_MAG"): va_mag.copy(),

            raw_col(bus_id, "IA_ANG"): ia_ang,
            raw_col(bus_id, "IA_MAG"): ia_mag,
            raw_col(bus_id, "IB_ANG"): wrap_deg(ia_ang - 120.0),
            raw_col(bus_id, "IB_MAG"): ia_mag.copy(),
            raw_col(bus_id, "IC_ANG"): wrap_deg(ia_ang + 120.0),
            raw_col(bus_id, "IC_MAG"): ia_mag.copy(),

            raw_col(bus_id, "Freq"): est_freq[:, bi],
            raw_col(bus_id, "ROCOF"): est_rocof[:, bi],

            "DATA_PRESENT": np.full(len(t), DATA_PRESENT_DEFAULT, dtype=int),
            "Event": ev,
        })

        estimated_dfs[bus_id] = df

    return estimated_dfs


# ============================================================================
# ESTIMATION METRICS / REPORT
# ============================================================================

def compute_estimation_metrics(
    simulation_dfs: dict[str, pd.DataFrame],
    estimated_dfs: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []

    for bus_id in sorted(simulation_dfs.keys(), key=lambda x: int(x)):
        sim_df = simulation_dfs[bus_id]
        est_df = estimated_dfs[bus_id]

        for suffix in PLOT_SUFFIXES:
            sim_col = raw_col(bus_id, suffix)
            est_col = raw_col(bus_id, suffix)
            if sim_col not in sim_df.columns or est_col not in est_df.columns:
                continue

            sim = sim_df[sim_col].to_numpy(dtype=float)
            est = est_df[est_col].to_numpy(dtype=float)

            if is_angle_suffix(suffix):
                diff = angle_diff_deg(sim, est)
                metric_rmse = float(np.sqrt(np.mean(diff**2)))
                metric_mae = float(np.mean(np.abs(diff)))
                metric_corr = safe_corr(sim, est)
                rows.append({
                    "bus_id": bus_id,
                    "signal": suffix,
                    "source": "pmu_passthrough" if bus_id in PMU_BUSES else "ybus_estimated",
                    "rmse": metric_rmse,
                    "mae": metric_mae,
                    "relative_rmse": np.nan,
                    "corr": metric_corr,
                    "max_abs_error": float(np.max(np.abs(diff))),
                })
            else:
                rows.append({
                    "bus_id": bus_id,
                    "signal": suffix,
                    "source": "pmu_passthrough" if bus_id in PMU_BUSES else "ybus_estimated",
                    "rmse": rmse(sim, est),
                    "mae": mae(sim, est),
                    "relative_rmse": rel_rmse(sim, est),
                    "corr": safe_corr(sim, est),
                    "max_abs_error": float(np.max(np.abs(sim - est))),
                })

    metrics_long = pd.DataFrame(rows)

    summary_bus = (
        metrics_long.groupby("bus_id")[["rmse", "mae", "relative_rmse", "corr"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("bus_id", key=lambda s: s.astype(int))
    )

    summary_signal = (
        metrics_long.groupby("signal")[["rmse", "mae", "relative_rmse", "corr"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("signal")
    )

    return metrics_long, summary_bus, summary_signal


def export_estimation_report(
    fault_bus: str,
    simulation_dfs: dict[str, pd.DataFrame],
    estimated_dfs: dict[str, pd.DataFrame],
) -> None:
    out_dir = estimated_output_dir(fault_bus)
    ensure_dir(out_dir)

    metrics_long, summary_bus, summary_signal = compute_estimation_metrics(simulation_dfs, estimated_dfs)

    metrics_long.to_csv(os.path.join(out_dir, "estimation_metrics_long.csv"), index=False)
    summary_bus.to_csv(os.path.join(out_dir, "estimation_summary_by_bus.csv"), index=False)
    summary_signal.to_csv(os.path.join(out_dir, "estimation_summary_by_signal.csv"), index=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fault_bus": str(fault_bus),
        "pmu_buses_used": PMU_BUSES,
        "estimation_mode": ESTIMATION_MODE,
        "use_positive_sequence": ESTIMATION_USE_POSITIVE_SEQUENCE,
        "temporal_lambda": ESTIMATION_TEMPORAL_LAMBDA,
        "prior_lambda": ESTIMATION_PRIOR_LAMBDA,
        "current_estimation_mode": CURRENT_ESTIMATION_MODE,
        "method": "YBUS partition solve with PMU positive-sequence voltages as knowns; non-PMU buses estimated with temporal regularization",
        "notes": [
            "PMU buses in estimated/ are passed through from simulation/",
            "Non-PMU buses are reconstructed using YBUS and only the 8 PMU buses as measurements",
            "Observed PMU voltage is built from positive-sequence phasor using VA/VB/VC",
            "Temporal regularization is applied across consecutive frames for non-PMU voltage estimation",
            "Currents are computed from nodal injection I = YBUS * V_est and should be interpreted as nodal-injection-like currents",
            "Frequency is derived from estimated voltage angle trajectory",
            "ROCOF is derived from estimated frequency",
        ],
        "summary_by_bus": summary_bus.to_dict(orient="records"),
        "summary_by_signal": summary_signal.to_dict(orient="records"),
    }

    with open(os.path.join(out_dir, "estimation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)


# ============================================================================
# EXPORT ESTIMATED
# ============================================================================

def export_estimated_outputs(
    estimated_dfs: dict[str, pd.DataFrame],
    simulation_dfs: dict[str, pd.DataFrame],
    fault_bus: str,
) -> str:
    out_dir = estimated_output_dir(fault_bus)
    ensure_dir(out_dir)

    for bus_id in sorted(estimated_dfs.keys(), key=lambda x: int(x)):
        df = estimated_dfs[bus_id]
        out_path = os.path.join(out_dir, f"BUS{bus_id}_Competition_Data_nanmask.csv")
        df.to_csv(out_path, index=False)

        if GENERATE_PLOTS and bus_id in simulation_dfs:
            generate_estimated_bus_plots(
                bus_id=bus_id,
                t=df["TIMESTAMP"].to_numpy(dtype=float),
                event_arr=df["Event"].to_numpy(dtype=int),
                simulation_df=simulation_dfs[bus_id],
                estimated_df=df,
                base_out_dir=out_dir,
            )

    export_estimation_report(fault_bus, simulation_dfs, estimated_dfs)
    return out_dir


# ============================================================================
# MAIN
# ============================================================================

def main():
    fault_bus = ensure_fault_bus_valid(FAULT_BUS)
    root_out_dir = current_run_output_dir(fault_bus)
    ensure_dir(root_out_dir)

    print("[INFO] Loading Event-0 artifacts...")
    artifacts = load_event0_artifacts()

    print(f"[INFO] Running fault simulation for bus {fault_bus}...")
    system = run_fault_simulation_andes(fault_bus)

    print("[INFO] Extracting all 39 bus signals...")
    signals_by_bus, meta = extract_all_bus_signals(system, fault_bus)

    print("[INFO] Exporting simulation/...")
    sim_dir, simulation_dfs = export_simulation_outputs(signals_by_bus, artifacts, fault_bus)

    print(f"[INFO] Building estimated/ using estimator mode={ESTIMATION_MODE}...")
    estimated_dfs = build_estimated_bus_dataframes(simulation_dfs, meta, fault_bus)

    print("[INFO] Exporting estimated/...")
    est_dir = export_estimated_outputs(estimated_dfs, simulation_dfs, fault_bus)

    run_info = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fault_bus": str(fault_bus),
        "pre_fault_s": PRE_FAULT_S,
        "fault_start_s": FAULT_START_S,
        "fault_duration_s": FAULT_DURATION_S,
        "fault_clear_s": FAULT_CLEAR_S,
        "post_fault_s": POST_FAULT_S,
        "sim_tf": SIM_TF,
        "sim_tstep": SIM_TSTEP,
        "fault_rf": FAULT_RF,
        "fault_xf": FAULT_XF,
        "event_label_mode": EVENT_LABEL_MODE,
        "data_present_default": DATA_PRESENT_DEFAULT,
        "plots_generated": GENERATE_PLOTS,
        "disable_angle_noise": DISABLE_ANGLE_NOISE,
        "angle_tiny_jitter_deg": ANGLE_TINY_JITTER_DEG,
        "estimation_mode": ESTIMATION_MODE,
        "estimation_temporal_lambda": ESTIMATION_TEMPORAL_LAMBDA,
        "estimation_prior_lambda": ESTIMATION_PRIOR_LAMBDA,
        "estimation_use_positive_sequence": ESTIMATION_USE_POSITIVE_SEQUENCE,
        "current_estimation_mode": CURRENT_ESTIMATION_MODE,
        "folders": {
            "root": os.path.abspath(root_out_dir),
            "simulation": os.path.abspath(sim_dir),
            "estimated": os.path.abspath(est_dir),
        },
        "pmu_buses_used_for_estimation": PMU_BUSES,
        "notes": [
            "simulation/ contains ANDES + Event0 measurement model for all 39 buses",
            "estimated/ contains PMU passthrough for the 8 PMU buses and temporally-regularized YBUS estimates for the remaining buses",
            "Voltage observations use PMU positive-sequence reconstruction from VA/VB/VC",
            "Angle channels remain low-noise / clean approximations by policy",
            "Current channels in estimated/ are nodal-injection-derived approximations, not guaranteed to match local PMU branch current exactly",
        ],
    }

    with open(os.path.join(root_out_dir, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump(run_info, fh, indent=2)

    print("=" * 84)
    print("SIMULATION + ESTIMATION TYPE-1 GENERATION FINISHED")
    print(f"Fault bus: {fault_bus}")
    print(f"Root output: {os.path.abspath(root_out_dir)}")
    print(f"Simulation folder: {os.path.abspath(sim_dir)}")
    print(f"Estimated folder: {os.path.abspath(est_dir)}")
    print(f"Bus files generated in simulation/: {len(simulation_dfs)}")
    print(f"Bus files generated in estimated/: {len(estimated_dfs)}")
    print("=" * 84)


if __name__ == "__main__":
    main()
