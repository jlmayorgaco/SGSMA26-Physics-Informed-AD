"""Generator dynamic parameters for the classical 2nd-order swing model.

Primary source: ANDES ieee39_full.xlsx GENROU sheet.
  M  = 2H (inertia coefficient, seconds)
  D  = damping (all 0 in the ANDES case)
  xd1 = x'd (d-axis transient reactance, p.u. on machine base)

Mapping between ANDES bus numbers (30-39) and our PSS/E file bus numbers:
  ANDES bus 30 ↔ PSS/E bus 24 (BUS30x1, connects to PSS/E bus 12 = BUS2)
  ANDES bus 31 ↔ PSS/E bus 25 (BUS31x1 SLACK)
  ANDES bus 32 ↔ PSS/E bus 26 (BUS32x1, connects to PSS/E bus 2  = BUS10)
  ANDES bus 33 ↔ PSS/E bus 27 (BUS33x1, connects to PSS/E bus 11 = BUS19)
  ANDES bus 34 ↔ PSS/E bus 28 (BUS34x1, connects to PSS/E bus 13 = BUS20)
  ANDES bus 35 ↔ PSS/E bus 29 (BUS35x1, connects to PSS/E bus 15 = BUS22)
  ANDES bus 36 ↔ PSS/E bus 30 (BUS36x1, connects to PSS/E bus 16 = BUS23)
  ANDES bus 37 ↔ PSS/E bus 31 (BUS37x1, connects to PSS/E bus 18 = BUS25)
  ANDES bus 38 ↔ PSS/E bus 32 (BUS38x1, connects to PSS/E bus 22 = BUS29)
  ANDES bus 39 ↔ PSS/E bus 33 (BUS39, 345 kV, directly in network)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

OMEGA_S = 2 * np.pi * 60.0   # synchronous angular velocity (rad/s)
S_BASE = 100.0                # system MVA base

# Ordered list of PSS/E bus numbers for the 10 generators (index 0..9)
GEN_PSSE_BUSES = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

# Mechanical power at base case (MW) from .raw source section, in the same PSS/E bus order
GEN_P_MW_BASE = [250.0, 520.696, 650.0, 632.0, 508.0, 650.0, 560.0, 540.0, 830.0, 1000.0]


@dataclass
class GenParams:
    """Dynamic parameters for the 10 IEEE-39 generators.

    All arrays are length-10, index matches GEN_PSSE_BUSES order.
    """
    H: np.ndarray      # inertia constants (seconds)
    D: np.ndarray      # damping coefficients (p.u.)
    xd1_sys: np.ndarray  # transient reactance on 100 MVA system base (p.u.)
    Sn: np.ndarray     # machine rated MVA
    Pm_base: np.ndarray  # base-case mechanical power (p.u. on S_BASE)
    psse_buses: list[int]  # PSS/E bus numbers, length 10


def load_params(andes_xlsx_path: str | None = None) -> GenParams:
    """Load generator parameters from ANDES ieee39_full.xlsx.

    Falls back to well-known literature values (Anderson & Fouad / Pai)
    if the ANDES file is not found.

    Args:
        andes_xlsx_path: explicit path to ieee39_full.xlsx; if None, auto-detected
                         from the andes package installation directory.
    """
    if andes_xlsx_path is None:
        andes_xlsx_path = _find_andes_xlsx()

    if andes_xlsx_path and os.path.exists(andes_xlsx_path):
        return _load_from_andes(andes_xlsx_path)
    else:
        log.warning("ANDES xlsx not found — using literature fallback parameters.")
        return _literature_fallback()


def _find_andes_xlsx() -> str | None:
    try:
        import andes
        base = os.path.dirname(andes.__file__)
        path = os.path.join(base, "cases", "ieee39", "ieee39_full.xlsx")
        return path if os.path.exists(path) else None
    except ImportError:
        return None


def _load_from_andes(xlsx_path: str) -> GenParams:
    """Parse GENROU sheet; order rows by bus number (30→39 = gen 1→10)."""
    import pandas as pd
    df = pd.read_excel(xlsx_path, sheet_name="GENROU")
    df = df.sort_values("bus").reset_index(drop=True)  # sorted by ANDES bus 30..39

    H = df["M"].to_numpy(float) / 2.0       # M = 2H
    D = df["D"].to_numpy(float)
    Sn = df["Sn"].to_numpy(float)

    # xd1 is on machine base; convert to system base
    xd1_sys = df["xd1"].to_numpy(float) * Sn / S_BASE

    Pm_base = np.array(GEN_P_MW_BASE) / S_BASE  # p.u. on system base

    log.info(
        "Loaded ANDES GENROU params: H=%s, D=%s, xd1_sys=%s",
        np.round(H, 2), D, np.round(xd1_sys, 4),
    )
    return GenParams(H=H, D=D, xd1_sys=xd1_sys, Sn=Sn, Pm_base=Pm_base,
                     psse_buses=GEN_PSSE_BUSES)


def _literature_fallback() -> GenParams:
    """Well-known IEEE 39-bus parameters from Pai / Anderson & Fouad (system base)."""
    # H values (seconds), D=0 for all
    H = np.array([4.20, 30.3, 35.8, 28.6, 26.0, 34.8, 26.4, 24.3, 34.5, 500.0])
    D = np.zeros(10)
    # x'd on machine base (from Pai)
    xd1_mach = np.array([0.0060, 0.0697, 0.0531, 0.0436, 0.1320, 0.0500,
                          0.0490, 0.0570, 0.0570, 0.0300])
    Sn = np.array([1040., 836., 843.7, 1174.8, 1080.2, 1085.7,
                   1025.2, 970.2, 1684.1, 1199.])
    xd1_sys = xd1_mach * Sn / S_BASE
    Pm_base = np.array(GEN_P_MW_BASE) / S_BASE
    return GenParams(H=H, D=D, xd1_sys=xd1_sys, Sn=Sn, Pm_base=Pm_base,
                     psse_buses=GEN_PSSE_BUSES)
