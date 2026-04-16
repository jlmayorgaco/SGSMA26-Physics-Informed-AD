from __future__ import annotations

import numpy as np
import pandas as pd


def phasor_from_mag_ang(mag: np.ndarray, ang_deg: np.ndarray) -> np.ndarray:
    mag = np.asarray(mag, dtype=float)
    ang_deg = np.asarray(ang_deg, dtype=float)
    return mag * np.exp(1j * np.deg2rad(ang_deg))


def compute_three_phase_power(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    va = phasor_from_mag_ang(out["VA_mag"], out["VA_ang"])
    vb = phasor_from_mag_ang(out["VB_mag"], out["VB_ang"])
    vc = phasor_from_mag_ang(out["VC_mag"], out["VC_ang"])

    ia = phasor_from_mag_ang(out["IA_mag"], out["IA_ang"])
    ib = phasor_from_mag_ang(out["IB_mag"], out["IB_ang"])
    ic = phasor_from_mag_ang(out["IC_mag"], out["IC_ang"])

    sa = va * np.conj(ia)
    sb = vb * np.conj(ib)
    sc = vc * np.conj(ic)

    out["P_A"] = np.real(sa)
    out["Q_A"] = np.imag(sa)

    out["P_B"] = np.real(sb)
    out["Q_B"] = np.imag(sb)

    out["P_C"] = np.real(sc)
    out["Q_C"] = np.imag(sc)

    s_total = sa + sb + sc
    out["P_total"] = np.real(s_total)
    out["Q_total"] = np.imag(s_total)

    return out