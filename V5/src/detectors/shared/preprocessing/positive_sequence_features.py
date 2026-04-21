from __future__ import annotations

import numpy as np
import pandas as pd


def _phase_complex(mag: pd.Series, ang: pd.Series) -> np.ndarray:
    mag_values = pd.to_numeric(mag, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ang_values = pd.to_numeric(ang, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return mag_values * np.exp(1j * np.deg2rad(ang_values))


def add_positive_sequence_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    buses = sorted({c.split("_", 1)[0] for c in out.columns if c.endswith("_VA_MAG")})
    alpha = np.exp(1j * 2.0 * np.pi / 3.0)
    for bus in buses:
        req = [
            f"{bus}_VA_MAG",
            f"{bus}_VA_ANG",
            f"{bus}_VB_MAG",
            f"{bus}_VB_ANG",
            f"{bus}_VC_MAG",
            f"{bus}_VC_ANG",
        ]
        if not all(c in out.columns for c in req):
            continue
        va = _phase_complex(out[f"{bus}_VA_MAG"], out[f"{bus}_VA_ANG"])
        vb = _phase_complex(out[f"{bus}_VB_MAG"], out[f"{bus}_VB_ANG"])
        vc = _phase_complex(out[f"{bus}_VC_MAG"], out[f"{bus}_VC_ANG"])
        v1 = (va + alpha * vb + (alpha**2) * vc) / 3.0
        out[f"{bus}_V1_MAG"] = np.abs(v1)
        out[f"{bus}_V1_ANG"] = np.rad2deg(np.angle(v1))
    return out

