"""ANDES trajectory and candidate-current construction helpers."""

from __future__ import annotations

import numpy as np

from src.calibration.andes_runtime import current_base_amp, get_timeseries
from src.calibration.current_mapping import build_current_candidates_for_bus


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
    _ = t
    return out


def compute_incident_branch_currents(system, t, v_df, a_df):
    lines = system.Line
    out = {}

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
        out.setdefault(fb, []).append((f"branch_{k}_from_{fb}_to_{tb}", np.abs(i_from) * current_base_amp(system, fb)))
        out.setdefault(tb, []).append((f"branch_{k}_to_{tb}_from_{fb}", np.abs(i_to) * current_base_amp(system, tb)))
    _ = t
    return out


def compute_generator_current_candidates(system, v_df):
    out = {}
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
            out.setdefault(bus_id, []).append((f"generator_{model_name}_{k}", i_pu * current_base_amp(system, bus_id)))
    return out


def build_all_trajectories(system, pmu_buses):
    t, v_df, a_df = get_timeseries(system)
    bus_injection = compute_bus_injection_current(system, t, v_df, a_df)
    branch_currents = compute_incident_branch_currents(system, t, v_df, a_df)
    generator_currents = compute_generator_current_candidates(system, v_df)

    trajectories = {}
    for bus_id in pmu_buses:
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
