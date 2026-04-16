import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import andes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER CONFIG
# ============================================================
CASE = andes.get_case("ieee39/ieee39_full.xlsx")
RAW_PATH = Path("data/metadata/IEEE_39_Bus_Power_System.raw")

BUS = 39
BUS_BASE_KV_LL = 345.0
SYS_BASE_MVA = 100.0
NOM_FREQ_HZ = 60.0

SIM_END = 6000.0
FAULT_START = 1200.0          # 20th minute
FAULT_DURATION = 0.02         # 20 ms
FAULT_CLEAR = FAULT_START + FAULT_DURATION

INTERNAL_DT = 1.0 / 120.0
EXPORT_DT = 1.0 / 30.0

FORCE_RUN_IF_UNSTABLE = True

# Zoom around event
ZOOM_PRE_SEC = 1.0
ZOOM_POST_SEC = 1.0

# Noise / EDA windows
NOISE_WINDOW_PRE_SEC = 1.0
POST_EVENT_WINDOW_SEC = 1.0

# Output folder structure
OUT_DIR = Path("results") / "synth" / "andes" / "SIM001"
CSV_DIR = OUT_DIR / "csv"
JSON_DIR = OUT_DIR / "json"
PLOTS_DIR = OUT_DIR / "plots"
TXT_DIR = OUT_DIR / "reports"

PMU_CSV = CSV_DIR / "Bus39_Competition_Data_andes_like.csv"
RAW_ZOOM_CSV = CSV_DIR / "Bus39_fault_window_raw_highres.csv"
PMU_ZOOM_CSV = CSV_DIR / "Bus39_fault_window_pmu_30fps.csv"
BUS_MAPPING_CSV = CSV_DIR / "bus_mapping_semantic_to_raw.csv"
EDA_FLAT_CSV = CSV_DIR / "eda_summary_flat.csv"

RUN_METADATA_JSON = JSON_DIR / "run_metadata.json"
YBUS_DIAG_JSON = JSON_DIR / "ybus_diagnostics.json"
MISSING_JSON = JSON_DIR / "missing_data_report.json"
GLOBAL_STATS_JSON = JSON_DIR / "signal_statistics_global.json"
PREFAULT_STATS_JSON = JSON_DIR / "signal_statistics_prefault.json"
ZOOM_STATS_JSON = JSON_DIR / "signal_statistics_fault_zoom.json"
NOISE_JSON = JSON_DIR / "noise_analysis.json"
EVENT_METRICS_JSON = JSON_DIR / "event_response_metrics.json"
CORR_JSON = JSON_DIR / "correlation_summary.json"
SUMMARY_TXT = TXT_DIR / "summary_report.txt"

FULL_V_PNG = PLOTS_DIR / "full_voltage.png"
FULL_I_PNG = PLOTS_DIR / "full_current.png"
FULL_F_PNG = PLOTS_DIR / "full_frequency_rocof.png"
ZOOM_V_PNG = PLOTS_DIR / "zoom_voltage_raw_highres.png"
ZOOM_I_PNG = PLOTS_DIR / "zoom_current_raw_highres.png"
ZOOM_F_PNG = PLOTS_DIR / "zoom_frequency_rocof_raw_highres.png"
DIST_PNG = PLOTS_DIR / "eda_distributions.png"
BOX_PNG = PLOTS_DIR / "eda_prefault_vs_zoom_boxplot.png"
PSD_PNG = PLOTS_DIR / "noise_psd_prefault.png"
CORR_PNG = PLOTS_DIR / "core_feature_correlation.png"
YBUS_PNG = PLOTS_DIR / "ybus_abs_difference.png"


# ============================================================
# HELPERS
# ============================================================
def ensure_dirs() -> None:
    for path in (OUT_DIR, CSV_DIR, JSON_DIR, PLOTS_DIR, TXT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def normalize_idx(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return int(float(x))


def complex_from_mag_ang(mag: np.ndarray, ang_rad: np.ndarray) -> np.ndarray:
    return mag * np.exp(1j * ang_rad)


def interp_real_with_nan(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    y = np.interp(t_dst, t_src, y_src)
    y = y.astype(float)
    y[t_dst < t_src[0]] = np.nan
    y[t_dst > t_src[-1]] = np.nan
    return y


def interp_complex_with_nan(t_src: np.ndarray, z_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    mag = np.interp(t_dst, t_src, np.abs(z_src))
    ang = np.interp(t_dst, t_src, np.unwrap(np.angle(z_src)))
    out = mag * np.exp(1j * ang)
    out = out.astype(complex)
    mask = (t_dst < t_src[0]) | (t_dst > t_src[-1])
    out[mask] = np.nan + 1j * np.nan
    return out


def phase_shift(z: np.ndarray, deg: float) -> np.ndarray:
    return z * np.exp(1j * np.deg2rad(deg))


def safe_angle_deg(z: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.angle(z))


def add_fault_markers(ax: Any, x_start: float, x_clear: float) -> None:
    ax.axvspan(x_start, x_clear, alpha=0.15)
    ax.axvline(x_start, linestyle="--", linewidth=1.2, label="Fault start")
    ax.axvline(x_clear, linestyle="--", linewidth=1.2, label="Fault clear")
    ax.grid(True, alpha=0.3)


def get_bus_col(system: Any, bus_no: int) -> int:
    bus_ids = [normalize_idx(x) for x in system.Bus.idx.v]
    if bus_no not in bus_ids:
        raise KeyError(f"Bus {bus_no} not found in ANDES case.")
    return bus_ids.index(bus_no)


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
        raise RuntimeError(
            f"Could not parse all 39 buses from RAW. Parsed {len(semantic_to_raw)}."
        )

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

    if arr is None:
        return None

    arr = np.asarray(arr)
    if arr.ndim != 2:
        return None
    if arr.shape != (nb, nb):
        return None
    return arr.astype(complex)


def get_andes_ybus_matrix(ss: Any, nb: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "build_ybus_called": False,
        "candidate_checks": [],
        "selected_candidate": None,
    }

    try:
        if hasattr(ss, "build_ybus"):
            ss.build_ybus()
            diag["build_ybus_called"] = True
    except Exception as exc:
        diag["build_ybus_error"] = repr(exc)

    candidates: List[Tuple[str, Any]] = []
    owners = {
        "system": ss,
        "dae": getattr(ss, "dae", None),
        "pflow": getattr(ss, "PFlow", None),
    }
    names = [
        "Ybus", "ybus", "Y", "Y_bus", "_Ybus", "_ybus", "Bbus", "B"
    ]

    for owner_name, owner in owners.items():
        if owner is None:
            continue
        for name in names:
            value = getattr(owner, name, None)
            candidates.append((f"{owner_name}.{name}", value))

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


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return None
    if np.isnan(xf) or np.isinf(xf):
        return None
    return xf


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, (np.ndarray,)):
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


def build_bus_dataframe(
    t: np.ndarray,
    vpu_bus: np.ndarray,
    ipu_bus: np.ndarray,
    freq: np.ndarray,
    rocof: np.ndarray,
    actual_end: Optional[float] = None,
) -> pd.DataFrame:
    vbase_ph = BUS_BASE_KV_LL * 1e3 / np.sqrt(3.0)
    ibase = SYS_BASE_MVA * 1e6 / (np.sqrt(3.0) * BUS_BASE_KV_LL * 1e3)

    va = vpu_bus * vbase_ph
    vb = phase_shift(va, -120.0)
    vc = phase_shift(va, +120.0)

    ia = ipu_bus * ibase
    ib = phase_shift(ia, -120.0)
    ic = phase_shift(ia, +120.0)

    event = np.zeros_like(t, dtype=int)
    event[(t >= FAULT_START) & (t < FAULT_CLEAR)] = 1

    data_present = np.ones_like(t, dtype=int)
    if actual_end is not None:
        data_present[t > actual_end] = 0

    df = pd.DataFrame({
        "TIMESTAMP": np.round(t, 6),
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
        "DATA_PRESENT": data_present,
        "Event": event,
    })

    return df


def finite_series(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    return arr[np.isfinite(arr)]


def window_df(df: pd.DataFrame, t0: float, t1: float) -> pd.DataFrame:
    return df[(df["TIMESTAMP"] >= t0) & (df["TIMESTAMP"] <= t1)].copy()


def _moment_stats(x: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if x.size < 3:
        return None, None
    mean = float(np.mean(x))
    centered = x - mean
    std = float(np.std(centered, ddof=0))
    if std <= 0:
        return 0.0, 0.0
    skew = float(np.mean((centered / std) ** 3))
    kurt = float(np.mean((centered / std) ** 4) - 3.0)
    return skew, kurt


def _autocorr(x: np.ndarray, lag: int) -> Optional[float]:
    if x.size <= lag or lag <= 0:
        return None
    x0 = x[:-lag]
    x1 = x[lag:]
    if x0.size < 2 or np.std(x0) == 0 or np.std(x1) == 0:
        return None
    return float(np.corrcoef(x0, x1)[0, 1])


def summarize_series(series: pd.Series, time_index: Optional[np.ndarray] = None) -> Dict[str, Any]:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = x[np.isfinite(x)]

    if finite.size == 0:
        return {
            "count_total": int(x.size),
            "count_finite": 0,
            "missing_count": int(np.isnan(x).sum()),
            "missing_ratio": float(np.isnan(x).mean()) if x.size else None,
        }

    skew, kurt = _moment_stats(finite)

    out: Dict[str, Any] = {
        "count_total": int(x.size),
        "count_finite": int(finite.size),
        "missing_count": int(np.isnan(x).sum()),
        "missing_ratio": float(np.isnan(x).mean()) if x.size else None,
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "min": float(np.min(finite)),
        "q01": float(np.quantile(finite, 0.01)),
        "q05": float(np.quantile(finite, 0.05)),
        "q25": float(np.quantile(finite, 0.25)),
        "median": float(np.median(finite)),
        "q75": float(np.quantile(finite, 0.75)),
        "q95": float(np.quantile(finite, 0.95)),
        "q99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
        "peak_to_peak": float(np.ptp(finite)),
        "rms": float(np.sqrt(np.mean(finite ** 2))),
        "abs_max": float(np.max(np.abs(finite))),
        "mad": float(np.median(np.abs(finite - np.median(finite)))),
        "skewness": skew,
        "excess_kurtosis": kurt,
        "lag1_autocorr": _autocorr(finite, 1),
        "lag5_autocorr": _autocorr(finite, 5),
    }

    if time_index is not None and len(time_index) == len(x):
        mask = np.isfinite(x)
        t = np.asarray(time_index, dtype=float)[mask]
        y = x[mask]
        if y.size >= 2:
            dt = np.diff(t)
            dy = np.diff(y)
            valid = np.isfinite(dt) & np.isfinite(dy) & (np.abs(dt) > 0)
            slope = dy[valid] / dt[valid] if np.any(valid) else np.array([])
            if slope.size:
                out["slope_mean"] = float(np.mean(slope))
                out["slope_std"] = float(np.std(slope, ddof=1)) if slope.size > 1 else 0.0
                out["slope_abs_max"] = float(np.max(np.abs(slope)))

    return out


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    t = df["TIMESTAMP"].to_numpy(dtype=float) if "TIMESTAMP" in df.columns else None
    for col in df.columns:
        if col == "TIMESTAMP":
            continue
        out[col] = summarize_series(df[col], time_index=t)
    return out


def linear_detrend(t: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    mask = np.isfinite(t) & np.isfinite(x)
    t2 = t[mask]
    x2 = x[mask]
    if x2.size < 3:
        return np.array([], dtype=float), {"detrend": "insufficient_data"}

    coeff = np.polyfit(t2, x2, 1)
    trend = coeff[0] * t2 + coeff[1]
    resid = x2 - trend
    return resid, {
        "detrend": "linear",
        "slope": float(coeff[0]),
        "intercept": float(coeff[1]),
    }


def dominant_fft_components(x: np.ndarray, fs: float, top_k: int = 5) -> List[Dict[str, Any]]:
    finite = x[np.isfinite(x)]
    if finite.size < 4 or fs <= 0:
        return []

    y = finite - np.mean(finite)
    n = y.size
    spec = np.fft.rfft(y)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.abs(spec)
    if amp.size <= 1:
        return []

    amp[0] = 0.0
    order = np.argsort(amp)[::-1][:top_k]
    out = []
    for idx in order:
        out.append({
            "frequency_hz": float(freq[idx]),
            "amplitude": float(amp[idx]),
        })
    return out


def noise_metrics(df: pd.DataFrame, fs: float, columns: List[str]) -> Dict[str, Any]:
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    out: Dict[str, Any] = {
        "sampling_frequency_hz": fs,
        "window_start": safe_float(df["TIMESTAMP"].min()) if len(df) else None,
        "window_end": safe_float(df["TIMESTAMP"].max()) if len(df) else None,
        "channels": {},
    }

    for col in columns:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        resid, detrend_info = linear_detrend(t, x)
        diffs = np.diff(x[np.isfinite(x)]) if np.isfinite(x).sum() >= 2 else np.array([], dtype=float)
        channel = {
            **detrend_info,
            "noise_std": safe_float(np.std(resid, ddof=1)) if resid.size > 1 else None,
            "noise_rms": safe_float(np.sqrt(np.mean(resid ** 2))) if resid.size else None,
            "noise_mad": safe_float(np.median(np.abs(resid - np.median(resid)))) if resid.size else None,
            "noise_peak_to_peak": safe_float(np.ptp(resid)) if resid.size else None,
            "diff_std": safe_float(np.std(diffs, ddof=1)) if diffs.size > 1 else None,
            "dominant_fft_components": dominant_fft_components(resid, fs=fs, top_k=5),
        }
        out["channels"][col] = channel

    return out


def compute_missing_data_report(df: pd.DataFrame) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": {},
        "data_present": {},
    }

    for col in df.columns:
        nan_count = int(df[col].isna().sum())
        report["columns"][col] = {
            "nan_count": nan_count,
            "nan_ratio": float(df[col].isna().mean()),
        }

    if "DATA_PRESENT" in df.columns:
        mask = pd.to_numeric(df["DATA_PRESENT"], errors="coerce").fillna(0).astype(int).to_numpy()
        ts = df["TIMESTAMP"].to_numpy(dtype=float)
        gaps: List[Dict[str, Any]] = []
        start_idx: Optional[int] = None
        for i, val in enumerate(mask):
            if val == 0 and start_idx is None:
                start_idx = i
            elif val == 1 and start_idx is not None:
                end_idx = i - 1
                gaps.append({
                    "start_time": float(ts[start_idx]),
                    "end_time": float(ts[end_idx]),
                    "samples": int(end_idx - start_idx + 1),
                    "duration_sec": float(ts[end_idx] - ts[start_idx]) if end_idx > start_idx else 0.0,
                })
                start_idx = None
        if start_idx is not None:
            end_idx = len(mask) - 1
            gaps.append({
                "start_time": float(ts[start_idx]),
                "end_time": float(ts[end_idx]),
                "samples": int(end_idx - start_idx + 1),
                "duration_sec": float(ts[end_idx] - ts[start_idx]) if end_idx > start_idx else 0.0,
            })

        report["data_present"] = {
            "valid_count": int(np.sum(mask == 1)),
            "missing_count": int(np.sum(mask == 0)),
            "missing_ratio": float(np.mean(mask == 0)),
            "gap_count": len(gaps),
            "gaps": gaps,
        }

    return report


def compare_windows(prefault_df: pd.DataFrame, postfault_df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in columns:
        pre = finite_series(prefault_df[col])
        post = finite_series(postfault_df[col])
        if pre.size == 0 or post.size == 0:
            out[col] = {"status": "insufficient_data"}
            continue
        pre_mean = float(np.mean(pre))
        post_mean = float(np.mean(post))
        delta = post_mean - pre_mean
        pre_std = float(np.std(pre, ddof=1)) if pre.size > 1 else 0.0
        z_like = delta / pre_std if pre_std > 0 else None
        out[col] = {
            "prefault_mean": pre_mean,
            "postfault_mean": post_mean,
            "delta_mean": delta,
            "relative_delta": delta / pre_mean if abs(pre_mean) > 1e-12 else None,
            "prefault_std": pre_std,
            "event_to_noise_ratio": z_like,
            "prefault_max": float(np.max(pre)),
            "postfault_max": float(np.max(post)),
            "prefault_min": float(np.min(pre)),
            "postfault_min": float(np.min(post)),
        }
    return out


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items: List[Tuple[str, Any]] = []
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def save_flattened_csv(path: Path, data: Dict[str, Any]) -> None:
    flat = flatten_dict(data)
    df = pd.DataFrame({
        "metric": list(flat.keys()),
        "value": list(flat.values()),
    })
    df.to_csv(path, index=False)


def plot_voltage(df: pd.DataFrame, path: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(df["TIMESTAMP"], df["VA_mag"], label="VA_mag")
    axes[0].plot(df["TIMESTAMP"], df["VB_mag"], label="VB_mag")
    axes[0].plot(df["TIMESTAMP"], df["VC_mag"], label="VC_mag")
    axes[0].set_ylabel("Voltage magnitude [V]")
    axes[0].set_title(f"{title_prefix} voltage phasors vs time")
    add_fault_markers(axes[0], FAULT_START, FAULT_CLEAR)
    axes[0].legend()

    axes[1].plot(df["TIMESTAMP"], df["VA_ang"], label="VA_ang")
    axes[1].plot(df["TIMESTAMP"], df["VB_ang"], label="VB_ang")
    axes[1].plot(df["TIMESTAMP"], df["VC_ang"], label="VC_ang")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Voltage angle [deg]")
    add_fault_markers(axes[1], FAULT_START, FAULT_CLEAR)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_current(df: pd.DataFrame, path: Path, title_prefix: str, current_note: str = "") -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(df["TIMESTAMP"], df["IA_mag"], label="IA_mag")
    axes[0].plot(df["TIMESTAMP"], df["IB_mag"], label="IB_mag")
    axes[0].plot(df["TIMESTAMP"], df["IC_mag"], label="IC_mag")
    axes[0].set_ylabel("Current magnitude [A]")
    axes[0].set_title(f"{title_prefix} current phasors vs time{current_note}")
    add_fault_markers(axes[0], FAULT_START, FAULT_CLEAR)
    axes[0].legend()

    axes[1].plot(df["TIMESTAMP"], df["IA_ang"], label="IA_ang")
    axes[1].plot(df["TIMESTAMP"], df["IB_ang"], label="IB_ang")
    axes[1].plot(df["TIMESTAMP"], df["IC_ang"], label="IC_ang")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Current angle [deg]")
    add_fault_markers(axes[1], FAULT_START, FAULT_CLEAR)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_freq_rocof(df: pd.DataFrame, path: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(df["TIMESTAMP"], df["Frequency"], label="Frequency")
    axes[0].set_ylabel("Frequency [Hz]")
    axes[0].set_title(f"{title_prefix} frequency / ROCOF vs time")
    add_fault_markers(axes[0], FAULT_START, FAULT_CLEAR)
    axes[0].legend()

    axes[1].plot(df["TIMESTAMP"], df["ROCOF"], label="ROCOF")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("ROCOF [Hz/s]")
    add_fault_markers(axes[1], FAULT_START, FAULT_CLEAR)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(df: pd.DataFrame, path: Path) -> None:
    cols = ["VA_mag", "IA_mag", "Frequency", "ROCOF"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, col in zip(axes, cols):
        vals = finite_series(df[col])
        if vals.size:
            ax.hist(vals, bins=50)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_box_prefault_vs_zoom(prefault_df: pd.DataFrame, zoom_df: pd.DataFrame, path: Path) -> None:
    cols = ["VA_mag", "IA_mag", "Frequency", "ROCOF"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, col in zip(axes, cols):
        pre = finite_series(prefault_df[col])
        zoom = finite_series(zoom_df[col])
        data = []
        labels = []
        if pre.size:
            data.append(pre)
            labels.append("prefault")
        if zoom.size:
            data.append(zoom)
            labels.append("fault_zoom")
        if data:
            ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_noise_psd(prefault_df: pd.DataFrame, path: Path, fs: float) -> None:
    cols = ["VA_mag", "IA_mag", "Frequency", "ROCOF"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for ax, col in zip(axes, cols):
        vals = pd.to_numeric(prefault_df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size >= 4 and fs > 0:
            y = vals - np.mean(vals)
            n = y.size
            freq = np.fft.rfftfreq(n, d=1.0 / fs)
            psd = np.abs(np.fft.rfft(y)) ** 2 / max(n, 1)
            ax.plot(freq, psd)
        ax.set_title(f"Prefault PSD - {col}")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> Dict[str, Any]:
    cols = ["VA_mag", "VA_ang", "IA_mag", "IA_ang", "Frequency", "ROCOF"]
    corr = df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title("Core feature correlation")
    fig.colorbar(im, ax=ax)

    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "columns": cols,
        "correlation_matrix": corr.round(6).to_dict(),
    }


def plot_ybus_difference(diff: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(np.abs(diff), aspect="auto")
    ax.set_title("|Ybus_raw_aligned - Ybus_andes|")
    ax.set_xlabel("Bus index in ANDES order")
    ax.set_ylabel("Bus index in ANDES order")
    fig.colorbar(image, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def infer_sampling_frequency(t: np.ndarray) -> Optional[float]:
    if t.size < 2:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def main() -> None:
    ensure_dirs()

    if not RAW_PATH.exists():
        raise FileNotFoundError(f"RAW file not found: {RAW_PATH.resolve()}")

    # ------------------------------------------------------------
    # LOAD, ADD FAULT, RUN TDS
    # ------------------------------------------------------------
    ss = andes.load(CASE, setup=False)
    if ss is None:
        raise RuntimeError(f"ANDES could not load case: {CASE}")

    ss.add("Fault", bus=BUS, tf=FAULT_START, tc=FAULT_CLEAR)
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

    # ------------------------------------------------------------
    # EXTRACT RAW TIME SERIES
    # ------------------------------------------------------------
    t_raw = np.asarray(ss.dae.ts.t, dtype=float)
    if t_raw.size == 0:
        raise RuntimeError("No time samples were saved by ANDES.")

    actual_end = float(t_raw[-1])
    if actual_end < SIM_END - 1e-6:
        warnings.warn(
            f"Simulation stopped early at t={actual_end:.4f} s. "
            "The PMU CSV will be padded with NaNs until the requested tf."
        )

    vmag_all = np.asarray(ss.dae.ts.y[:, ss.Bus.v.a], dtype=float)
    vang_all = np.asarray(ss.dae.ts.y[:, ss.Bus.a.a], dtype=float)

    bus_col = get_bus_col(ss, BUS)
    vpu_bus_raw = complex_from_mag_ang(vmag_all[:, bus_col], vang_all[:, bus_col])

    freq_raw = np.full_like(t_raw, NOM_FREQ_HZ, dtype=float)
    if hasattr(ss, "GENROU") and hasattr(ss.GENROU, "omega") and len(ss.GENROU.omega.a) > 0:
        omega_raw = np.asarray(ss.dae.ts.x[:, ss.GENROU.omega.a], dtype=float)
        freq_raw = NOM_FREQ_HZ * omega_raw.mean(axis=1)

    rocof_raw = np.gradient(freq_raw, t_raw) if len(t_raw) >= 2 else np.full_like(freq_raw, np.nan)

    # ------------------------------------------------------------
    # YBUS ALIGNMENT / VALIDATION AGAINST ANDES
    # ------------------------------------------------------------
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
        diff = ybus_raw_aligned - ybus_andes
        plot_ybus_difference(diff, YBUS_PNG)
    else:
        diff = None

    # Prefer ANDES Ybus if accessible and close; otherwise keep RAW aligned.
    if ybus_compare["andes_ybus_accessible"] and ybus_compare["same_matrix"]:
        ybus_used = ybus_andes
        ybus_source_used = "andes_internal"
    else:
        ybus_used = ybus_raw_aligned
        ybus_source_used = "raw_aligned_to_andes_order"

    # ------------------------------------------------------------
    # CURRENT PROXY USING SELECTED YBUS
    # ------------------------------------------------------------
    try:
        vpu_all = complex_from_mag_ang(vmag_all, vang_all)
        ipu_all = vpu_all @ ybus_used.T
        ipu_bus_raw = ipu_all[:, bus_col]
        current_available = True
    except Exception as exc:
        warnings.warn(
            "Could not compute current proxy from selected Ybus. "
            f"Current columns will be NaN. Original error: {exc}"
        )
        ipu_bus_raw = np.full_like(vpu_bus_raw, np.nan + 1j * np.nan)
        current_available = False

    # ------------------------------------------------------------
    # RESAMPLE TO PMU-LIKE 30 FPS GRID
    # ------------------------------------------------------------
    t_out = np.round(np.arange(0.0, SIM_END + 1e-12, EXPORT_DT), 6)

    vpu_bus_pmu = interp_complex_with_nan(t_raw, vpu_bus_raw, t_out)
    ipu_bus_pmu = interp_complex_with_nan(t_raw, ipu_bus_raw, t_out)
    freq_pmu = interp_real_with_nan(t_raw, freq_raw, t_out)
    rocof_pmu = interp_real_with_nan(t_raw, rocof_raw, t_out)

    # ------------------------------------------------------------
    # DATAFRAMES
    # ------------------------------------------------------------
    df_pmu = build_bus_dataframe(
        t=t_out,
        vpu_bus=vpu_bus_pmu,
        ipu_bus=ipu_bus_pmu,
        freq=freq_pmu,
        rocof=rocof_pmu,
        actual_end=actual_end,
    )
    df_pmu["TIMESTAMP"] = df_pmu["TIMESTAMP"].round(3)

    df_raw = build_bus_dataframe(
        t=t_raw,
        vpu_bus=vpu_bus_raw,
        ipu_bus=ipu_bus_raw,
        freq=freq_raw,
        rocof=rocof_raw,
        actual_end=actual_end,
    )

    zoom_start = max(0.0, FAULT_START - ZOOM_PRE_SEC)
    zoom_end = min(actual_end, FAULT_CLEAR + ZOOM_POST_SEC)
    prefault_start = max(0.0, FAULT_START - NOISE_WINDOW_PRE_SEC)
    prefault_end = FAULT_START
    postfault_start = FAULT_CLEAR
    postfault_end = min(actual_end, FAULT_CLEAR + POST_EVENT_WINDOW_SEC)

    df_zoom_raw = window_df(df_raw, zoom_start, zoom_end)
    df_zoom_pmu = window_df(df_pmu, zoom_start, zoom_end)
    df_prefault_raw = window_df(df_raw, prefault_start, prefault_end)
    df_postfault_raw = window_df(df_raw, postfault_start, postfault_end)

    # ------------------------------------------------------------
    # EXPORT CSVs
    # ------------------------------------------------------------
    df_pmu.to_csv(PMU_CSV, index=False)
    df_zoom_raw.to_csv(RAW_ZOOM_CSV, index=False)
    df_zoom_pmu.to_csv(PMU_ZOOM_CSV, index=False)

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------
    current_note = ""
    if not current_available:
        current_note = " (current proxy unavailable -> NaN fallback)"
    elif ybus_source_used != "andes_internal":
        current_note = " (current proxy from RAW-aligned Ybus)"

    plot_voltage(df_pmu, FULL_V_PNG, "Full-duration")
    plot_current(df_pmu, FULL_I_PNG, "Full-duration", current_note=current_note)
    plot_freq_rocof(df_pmu, FULL_F_PNG, "Full-duration")

    plot_voltage(df_zoom_raw, ZOOM_V_PNG, "Fault zoom raw/high-res")
    plot_current(df_zoom_raw, ZOOM_I_PNG, "Fault zoom raw/high-res", current_note=current_note)
    plot_freq_rocof(df_zoom_raw, ZOOM_F_PNG, "Fault zoom raw/high-res")

    plot_distributions(df_pmu, DIST_PNG)
    plot_box_prefault_vs_zoom(df_prefault_raw, df_zoom_raw, BOX_PNG)
    fs_prefault = infer_sampling_frequency(df_prefault_raw["TIMESTAMP"].to_numpy(dtype=float)) or 0.0
    plot_noise_psd(df_prefault_raw, PSD_PNG, fs=fs_prefault)
    corr_summary = plot_correlation_heatmap(df_pmu, CORR_PNG)

    # ------------------------------------------------------------
    # STATISTICS / EDA
    # ------------------------------------------------------------
    global_stats = summarize_dataframe(df_pmu)
    prefault_stats = summarize_dataframe(df_prefault_raw)
    zoom_stats = summarize_dataframe(df_zoom_raw)
    missing_report = compute_missing_data_report(df_pmu)
    noise_report = noise_metrics(
        df_prefault_raw,
        fs=fs_prefault,
        columns=["VA_mag", "VA_ang", "IA_mag", "IA_ang", "Frequency", "ROCOF"],
    )
    event_metrics = {
        "fault_window": {
            "zoom_start": zoom_start,
            "fault_start": FAULT_START,
            "fault_clear": FAULT_CLEAR,
            "zoom_end": zoom_end,
            "fault_duration_sec": FAULT_DURATION,
        },
        "prefault_vs_postfault": compare_windows(
            df_prefault_raw,
            df_postfault_raw,
            columns=["VA_mag", "VA_ang", "IA_mag", "IA_ang", "Frequency", "ROCOF"],
        ),
        "samples": {
            "raw_total": int(len(df_raw)),
            "pmu_total": int(len(df_pmu)),
            "prefault_raw": int(len(df_prefault_raw)),
            "zoom_raw": int(len(df_zoom_raw)),
            "postfault_raw": int(len(df_postfault_raw)),
        },
    }

    ybus_diag = {
        "bus_of_interest": BUS,
        "andes_bus_order": andes_bus_order,
        "bus_mapping_preview": mapping_df.head(10).to_dict(orient="records"),
        "semantic_to_raw": semantic_to_raw,
        "raw_to_semantic": raw_to_semantic,
        "selected_ybus_source": ybus_source_used,
        "andes_probe": andes_ybus_probe,
        "comparison": ybus_compare,
        "notes": [
            "Current proxy is computed from nodal current injection I = Ybus * V using the selected static Ybus.",
            "If the internal ANDES Ybus is accessible and numerically matches the RAW-aligned Ybus, it is preferred.",
            "If the internal ANDES Ybus is not accessible or differs beyond tolerance, the script falls back to the RAW-aligned Ybus and records the discrepancy.",
            "This proxy does not reconstruct switched topology snapshots during the fault unless ANDES exposes them explicitly.",
        ],
    }

    metadata = {
        "case": str(CASE),
        "raw_path": str(RAW_PATH),
        "bus": BUS,
        "bus_base_kv_ll": BUS_BASE_KV_LL,
        "system_base_mva": SYS_BASE_MVA,
        "nominal_frequency_hz": NOM_FREQ_HZ,
        "sim_end_requested_sec": SIM_END,
        "actual_simulated_end_sec": actual_end,
        "fault_start_sec": FAULT_START,
        "fault_clear_sec": FAULT_CLEAR,
        "fault_duration_sec": FAULT_DURATION,
        "internal_dt_sec": INTERNAL_DT,
        "export_dt_sec": EXPORT_DT,
        "zoom_pre_sec": ZOOM_PRE_SEC,
        "zoom_post_sec": ZOOM_POST_SEC,
        "current_proxy_available": current_available,
        "current_proxy_source": ybus_source_used,
        "raw_sampling_frequency_hz": infer_sampling_frequency(t_raw),
        "pmu_sampling_frequency_hz": infer_sampling_frequency(df_pmu["TIMESTAMP"].to_numpy(dtype=float)),
        "outputs": {
            "pmu_csv": str(PMU_CSV),
            "raw_zoom_csv": str(RAW_ZOOM_CSV),
            "pmu_zoom_csv": str(PMU_ZOOM_CSV),
            "bus_mapping_csv": str(BUS_MAPPING_CSV),
            "eda_flat_csv": str(EDA_FLAT_CSV),
            "plots_dir": str(PLOTS_DIR),
            "json_dir": str(JSON_DIR),
        },
    }

    eda_summary = {
        "metadata": metadata,
        "ybus": ybus_diag,
        "missing_data": missing_report,
        "event_metrics": event_metrics,
        "noise": noise_report,
        "correlation": corr_summary,
        "global_stats_subset": {
            k: global_stats[k]
            for k in ["VA_mag", "VA_ang", "IA_mag", "IA_ang", "Frequency", "ROCOF", "DATA_PRESENT", "Event"]
            if k in global_stats
        },
        "prefault_stats_subset": {
            k: prefault_stats[k]
            for k in ["VA_mag", "VA_ang", "IA_mag", "IA_ang", "Frequency", "ROCOF"]
            if k in prefault_stats
        },
        "zoom_stats_subset": {
            k: zoom_stats[k]
            for k in ["VA_mag", "VA_ang", "IA_mag", "IA_ang", "Frequency", "ROCOF"]
            if k in zoom_stats
        },
    }

    # ------------------------------------------------------------
    # SAVE REPORTS
    # ------------------------------------------------------------
    save_json(RUN_METADATA_JSON, metadata)
    save_json(YBUS_DIAG_JSON, ybus_diag)
    save_json(MISSING_JSON, missing_report)
    save_json(GLOBAL_STATS_JSON, global_stats)
    save_json(PREFAULT_STATS_JSON, prefault_stats)
    save_json(ZOOM_STATS_JSON, zoom_stats)
    save_json(NOISE_JSON, noise_report)
    save_json(EVENT_METRICS_JSON, event_metrics)
    save_json(CORR_JSON, corr_summary)
    save_flattened_csv(EDA_FLAT_CSV, eda_summary)

    summary_lines = [
        "ANDES IEEE39 Bus 39 synthesis summary",
        "=" * 50,
        f"Output root: {OUT_DIR}",
        f"Requested tf: {SIM_END:.6f} s",
        f"Actual simulated end: {actual_end:.6f} s",
        f"Fault window: [{FAULT_START:.6f}, {FAULT_CLEAR:.6f}] s",
        f"Zoom window: [{zoom_start:.6f}, {zoom_end:.6f}] s",
        f"Current proxy available: {current_available}",
        f"Current proxy source: {ybus_source_used}",
        f"ANDES Ybus accessible: {ybus_compare['andes_ybus_accessible']}",
        f"Ybus same matrix: {ybus_compare['same_matrix']}",
        f"Ybus max abs diff: {ybus_compare['max_abs_diff']}",
        f"Ybus relative Fro error: {ybus_compare['relative_fro_error']}",
        f"PMU CSV: {PMU_CSV}",
        f"Raw zoom CSV: {RAW_ZOOM_CSV}",
        f"Plots: {PLOTS_DIR}",
        f"JSON reports: {JSON_DIR}",
    ]
    save_text(SUMMARY_TXT, "\n".join(summary_lines) + "\n")

    print(f"Actual simulated end time: {actual_end:.6f} s")
    print(f"Saved PMU CSV: {PMU_CSV}")
    print(f"Saved raw zoom CSV: {RAW_ZOOM_CSV}")
    print(f"Saved PMU zoom CSV: {PMU_ZOOM_CSV}")
    print(f"Saved plots to: {PLOTS_DIR}")
    print(f"Saved JSON reports to: {JSON_DIR}")
    print(f"Saved summary report: {SUMMARY_TXT}")
    print("Done.")


if __name__ == "__main__":
    main()
