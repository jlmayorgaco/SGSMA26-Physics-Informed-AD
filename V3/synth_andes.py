import re
import warnings
from pathlib import Path

import andes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

OUT_CSV = "Bus39_Competition_Data_andes_like.csv"
OUT_V_PNG = "bus39_voltage_vs_t.png"
OUT_I_PNG = "bus39_current_vs_t.png"
OUT_F_PNG = "bus39_freq_rocof_vs_t.png"


# ============================================================
# HELPERS
# ============================================================
def normalize_idx(x):
    try:
        return int(x)
    except Exception:
        return int(float(x))


def complex_from_mag_ang(mag, ang_rad):
    return mag * np.exp(1j * ang_rad)


def interp_real_with_nan(t_src, y_src, t_dst):
    y = np.interp(t_dst, t_src, y_src)
    y[t_dst < t_src[0]] = np.nan
    y[t_dst > t_src[-1]] = np.nan
    return y


def interp_complex_with_nan(t_src, z_src, t_dst):
    mag = np.interp(t_dst, t_src, np.abs(z_src))
    ang = np.interp(t_dst, t_src, np.unwrap(np.angle(z_src)))
    out = mag * np.exp(1j * ang)
    mask = (t_dst < t_src[0]) | (t_dst > t_src[-1])
    out = out.astype(complex)
    out[mask] = np.nan + 1j * np.nan
    return out


def phase_shift(z, deg):
    return z * np.exp(1j * np.deg2rad(deg))


def safe_angle_deg(z):
    return np.rad2deg(np.angle(z))


def add_fault_markers(ax):
    ax.axvspan(FAULT_START, FAULT_CLEAR, alpha=0.15)
    ax.axvline(FAULT_START, linestyle="--", linewidth=1.2, label="Fault start")
    ax.axvline(FAULT_CLEAR, linestyle="--", linewidth=1.2, label="Fault clear")
    ax.grid(True, alpha=0.3)


def get_bus_col(system, bus_no):
    bus_ids = [normalize_idx(x) for x in system.Bus.idx.v]
    if bus_no not in bus_ids:
        raise KeyError(f"Bus {bus_no} not found in ANDES case.")
    return bus_ids.index(bus_no)


def parse_raw_bus_mapping(raw_path: Path):
    """
    Parse the BUS section of the RAW file.

    Returns
    -------
    semantic_to_raw : dict[int, int]
        e.g. 39 -> 33 because line says raw-id 33 is 'BUS39'
    raw_to_semantic : dict[int, int]
        inverse mapping
    """
    text = raw_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    semantic_to_raw = {}
    raw_to_semantic = {}

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
        m = re.search(r"BUS(\d+)", name.upper())
        if m is None:
            continue

        semantic_bus = int(m.group(1))
        semantic_to_raw[semantic_bus] = raw_id
        raw_to_semantic[raw_id] = semantic_bus

    if len(semantic_to_raw) < 39:
        raise RuntimeError(
            f"Could not parse all 39 buses from RAW. Parsed {len(semantic_to_raw)}."
        )

    return semantic_to_raw, raw_to_semantic


def build_ybus_from_raw(raw_path: Path):
    """
    Build Ybus from the RAW branch section using the standard pi-model plus
    off-nominal tap and phase shift when present.

    Returns
    -------
    Y_raw : ndarray complex, shape [nb, nb]
        Ybus in RAW internal bus-id order
    raw_id_order : list[int]
        The RAW bus ids corresponding to Y_raw row/col order
    semantic_to_raw : dict[int, int]
    raw_to_semantic : dict[int, int]
    """
    semantic_to_raw, raw_to_semantic = parse_raw_bus_mapping(raw_path)
    text = raw_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    raw_id_order = sorted(raw_to_semantic.keys())
    raw_id_to_idx = {rid: k for k, rid in enumerate(raw_id_order)}
    nb = len(raw_id_order)
    Y = np.zeros((nb, nb), dtype=complex)

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
            # Skip pathological zero-impedance entries
            continue

        y_series = 1.0 / complex(r, x)
        y_shunt_half = 1j * b / 2.0

        a = tap * np.exp(1j * np.deg2rad(shift_deg))
        ii = raw_id_to_idx[raw_i]
        jj = raw_id_to_idx[raw_j]

        Y[ii, ii] += (y_series + y_shunt_half) / (a * np.conj(a))
        Y[jj, jj] += y_series + y_shunt_half
        Y[ii, jj] += -y_series / np.conj(a)
        Y[jj, ii] += -y_series / a

    return Y, raw_id_order, semantic_to_raw, raw_to_semantic


def align_raw_ybus_to_andes_order(ss, raw_path: Path):
    """
    Reorder RAW-based Ybus so it matches ANDES bus order.
    Assumes ANDES bus indices are the semantic bus numbers 1..39.
    """
    Y_raw, raw_id_order, semantic_to_raw, raw_to_semantic = build_ybus_from_raw(raw_path)
    raw_id_to_pos = {rid: pos for pos, rid in enumerate(raw_id_order)}

    andes_bus_order = [normalize_idx(x) for x in ss.Bus.idx.v]
    missing = [b for b in andes_bus_order if b not in semantic_to_raw]
    if missing:
        raise RuntimeError(f"These ANDES buses were not found in RAW mapping: {missing}")

    perm = [raw_id_to_pos[semantic_to_raw[b]] for b in andes_bus_order]
    Y_aligned = Y_raw[np.ix_(perm, perm)]

    return Y_aligned, semantic_to_raw, raw_to_semantic


# ============================================================
# LOAD, ADD FAULT, RUN TDS
# ============================================================
if not RAW_PATH.exists():
    raise FileNotFoundError(f"RAW file not found: {RAW_PATH.resolve()}")

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


# ============================================================
# EXTRACT RAW TIME SERIES
# ============================================================
t_raw = np.asarray(ss.dae.ts.t, dtype=float)
if t_raw.size == 0:
    raise RuntimeError("No time samples were saved by ANDES.")

actual_end = float(t_raw[-1])
if actual_end < SIM_END - 1e-6:
    warnings.warn(
        f"Simulation stopped early at t={actual_end:.4f} s. "
        "The CSV will be padded with NaNs until 6000 s."
    )

Vmag_all = np.asarray(ss.dae.ts.y[:, ss.Bus.v.a], dtype=float)
Vang_all = np.asarray(ss.dae.ts.y[:, ss.Bus.a.a], dtype=float)

bus_col = get_bus_col(ss, BUS)
Vpu_bus_raw = complex_from_mag_ang(Vmag_all[:, bus_col], Vang_all[:, bus_col])

freq_raw = np.full_like(t_raw, NOM_FREQ_HZ, dtype=float)
if hasattr(ss, "GENROU") and hasattr(ss.GENROU, "omega") and len(ss.GENROU.omega.a) > 0:
    omega_raw = np.asarray(ss.dae.ts.x[:, ss.GENROU.omega.a], dtype=float)
    freq_raw = NOM_FREQ_HZ * omega_raw.mean(axis=1)

rocof_raw = np.gradient(freq_raw, t_raw) if len(t_raw) >= 2 else np.full_like(freq_raw, np.nan)


# ============================================================
# CURRENT PROXY FROM RAW-BASED YBUS
# ============================================================
try:
    Ybus_aligned, semantic_to_raw, raw_to_semantic = align_raw_ybus_to_andes_order(ss, RAW_PATH)
    Vpu_all = complex_from_mag_ang(Vmag_all, Vang_all)   # [nt, nb] in ANDES semantic order
    Ipu_all = Vpu_all @ Ybus_aligned.T                   # nodal current injection proxy
    Ipu_bus_raw = Ipu_all[:, bus_col]
    current_available = True

    raw_internal_for_bus = semantic_to_raw[BUS]
    print(f"RAW mapping: semantic BUS{BUS} -> raw internal id {raw_internal_for_bus}")
except Exception as e:
    warnings.warn(
        "Could not compute current proxy from RAW-based Ybus. "
        f"Current columns will be NaN. Original error: {e}"
    )
    Ipu_bus_raw = np.full_like(Vpu_bus_raw, np.nan + 1j * np.nan)
    current_available = False


# ============================================================
# RESAMPLE TO PMU-LIKE 30 FPS GRID
# ============================================================
t_out = np.round(np.arange(0.0, SIM_END + 1e-12, EXPORT_DT), 3)

Vpu_bus = interp_complex_with_nan(t_raw, Vpu_bus_raw, t_out)
Ipu_bus = interp_complex_with_nan(t_raw, Ipu_bus_raw, t_out)
freq = interp_real_with_nan(t_raw, freq_raw, t_out)
rocof = interp_real_with_nan(t_raw, rocof_raw, t_out)


# ============================================================
# CONVERT TO PHASE QUANTITIES
# ============================================================
Vbase_ph = BUS_BASE_KV_LL * 1e3 / np.sqrt(3.0)
Ibase = SYS_BASE_MVA * 1e6 / (np.sqrt(3.0) * BUS_BASE_KV_LL * 1e3)

Va = Vpu_bus * Vbase_ph
Vb = phase_shift(Va, -120.0)
Vc = phase_shift(Va, +120.0)

Ia = Ipu_bus * Ibase
Ib = phase_shift(Ia, -120.0)
Ic = phase_shift(Ia, +120.0)

event = np.zeros_like(t_out, dtype=int)
event[(t_out >= FAULT_START) & (t_out < FAULT_CLEAR)] = 1

data_present = np.ones_like(t_out, dtype=int)
data_present[t_out > actual_end] = 0


# ============================================================
# BUILD CSV
# ============================================================
df = pd.DataFrame({
    "TIMESTAMP": np.round(t_out, 3),

    "VA_mag": np.abs(Va),
    "VA_ang": safe_angle_deg(Va),
    "VB_mag": np.abs(Vb),
    "VB_ang": safe_angle_deg(Vb),
    "VC_mag": np.abs(Vc),
    "VC_ang": safe_angle_deg(Vc),

    "IA_mag": np.abs(Ia),
    "IA_ang": safe_angle_deg(Ia),
    "IB_mag": np.abs(Ib),
    "IB_ang": safe_angle_deg(Ib),
    "IC_mag": np.abs(Ic),
    "IC_ang": safe_angle_deg(Ic),

    "Frequency": freq,
    "ROCOF": rocof,
    "DATA_PRESENT": data_present,
    "Event": event,
})

df.to_csv(OUT_CSV, index=False)


# ============================================================
# PLOTS
# ============================================================
# Voltage
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

axes[0].plot(df["TIMESTAMP"], df["VA_mag"], label="VA_mag")
axes[0].plot(df["TIMESTAMP"], df["VB_mag"], label="VB_mag")
axes[0].plot(df["TIMESTAMP"], df["VC_mag"], label="VC_mag")
axes[0].set_ylabel("Voltage magnitude [V]")
axes[0].set_title("Bus 39 voltage phasors vs time")
add_fault_markers(axes[0])
axes[0].legend()

axes[1].plot(df["TIMESTAMP"], df["VA_ang"], label="VA_ang")
axes[1].plot(df["TIMESTAMP"], df["VB_ang"], label="VB_ang")
axes[1].plot(df["TIMESTAMP"], df["VC_ang"], label="VC_ang")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Voltage angle [deg]")
add_fault_markers(axes[1])
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_V_PNG, dpi=200, bbox_inches="tight")
plt.close(fig)

# Current
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

axes[0].plot(df["TIMESTAMP"], df["IA_mag"], label="IA_mag")
axes[0].plot(df["TIMESTAMP"], df["IB_mag"], label="IB_mag")
axes[0].plot(df["TIMESTAMP"], df["IC_mag"], label="IC_mag")
axes[0].set_ylabel("Current magnitude [A]")
title_current = "Bus 39 current phasors vs time"
if not current_available:
    title_current += " (RAW-Ybus current unavailable -> NaN fallback)"
axes[0].set_title(title_current)
add_fault_markers(axes[0])
axes[0].legend()

axes[1].plot(df["TIMESTAMP"], df["IA_ang"], label="IA_ang")
axes[1].plot(df["TIMESTAMP"], df["IB_ang"], label="IB_ang")
axes[1].plot(df["TIMESTAMP"], df["IC_ang"], label="IC_ang")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("Current angle [deg]")
add_fault_markers(axes[1])
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_I_PNG, dpi=200, bbox_inches="tight")
plt.close(fig)

# Frequency / ROCOF
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

axes[0].plot(df["TIMESTAMP"], df["Frequency"], label="Frequency")
axes[0].set_ylabel("Frequency [Hz]")
axes[0].set_title("Bus 39 frequency / ROCOF vs time")
add_fault_markers(axes[0])
axes[0].legend()

axes[1].plot(df["TIMESTAMP"], df["ROCOF"], label="ROCOF")
axes[1].set_xlabel("Time [s]")
axes[1].set_ylabel("ROCOF [Hz/s]")
add_fault_markers(axes[1])
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_F_PNG, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Actual simulated end time: {actual_end:.4f} s")
print(f"Saved CSV: {OUT_CSV}")
print(f"Saved plot: {OUT_V_PNG}")
print(f"Saved plot: {OUT_I_PNG}")
print(f"Saved plot: {OUT_F_PNG}")
print("Done.")