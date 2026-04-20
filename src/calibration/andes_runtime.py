"""ANDES runtime helpers for m3 calibration."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def run_andes_normal(tf=60.0, tstep=1.0 / 30.0):
    import andes

    case_path = andes.get_case("ieee39/ieee39_full.xlsx")
    system = andes.load(case_path)
    if system.PFlow.run() is False:
        raise RuntimeError("ANDES power flow failed")
    system.TDS.config.tf = tf
    system.TDS.config.tstep = tstep
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
            return float(getattr(system.config, attr))
        except Exception:
            pass
    return 100.0


def current_base_amp(system, bus_id):
    vn_kv = max(get_bus_vn_kv(system, bus_id), 1e-6)
    return get_system_mva(system) * 1e6 / (math.sqrt(3.0) * vn_kv * 1e3)
