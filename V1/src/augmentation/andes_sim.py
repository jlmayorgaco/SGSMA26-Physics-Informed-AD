"""Synthetic event generator using ANDES (primary) with physics fallback.

Generates labeled synthetic PMU data in competition CSV format for M7.

Event types generated
---------------------
1  - Three-phase fault (Fault): voltage dip + current surge + oscillations
2  - Line outage (Toggle): power flow redistribution + oscillations
3  - Generation step change: swing equation transient, new steady-state freq
4  - Load step change: similar to gen change, symmetric
5  - PMU data dropout: DATA_PRESENT=0, all NaN

Output format (per bus CSV)
---------------------------
TIMESTAMP, BUSk_VA_ANG, BUSk_VA_MAG, ..., BUSk_ROCOF, DATA_PRESENT, Event

Each synthetic run generates a single event within a 6-second window:
  [0, 2 s)   — normal pre-event operation (Event=0)
  [2, 2+dur) — event in progress (Event=label)
  [2+dur, 6) — post-event (Event=0 or label if event persists)
PMU-local data-quality events keep that label only in the affected bus CSV;
the merged training view reconstructs the scenario label from all PMU streams.

The "normal" baseline is sampled from the calibration (first 60 s of real data).
Measurements are given in the same physical units as the competition CSVs:
  VA_MAG in volts (line-to-neutral RMS, ~200 kV)
  IA_MAG in amperes (~500 A)
  Freq in Hz (nominal 60)
  ROCOF in Hz/s
  angles in degrees
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.io.load_csv import PMU_BUSES, load_all, get_measurement_cols

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
FPS = 30.0
DT  = 1.0 / FPS
NOMINAL_FREQ = 60.0      # Hz

# Competition channel suffixes and their order per bus (14 channels)
_SUFFIXES = [
    "VA_ANG", "VA_MAG", "VB_ANG", "VB_MAG", "VC_ANG", "VC_MAG",
    "IA_ANG", "IA_MAG", "IB_ANG", "IB_MAG", "IC_ANG", "IC_MAG",
    "Freq", "ROCOF",
]


# ── baseline extraction ────────────────────────────────────────────────────────

def extract_normal_baseline(
    df: pd.DataFrame,
    fps: float = FPS,
    n_seconds: float = 60.0,
) -> dict[str, np.ndarray]:
    """Extract per-channel statistics from the first n_seconds of normal data.

    Returns {col_name: (mean, std)} for each measurement column.
    """
    calib_mask = (df["TIMESTAMP"] <= n_seconds) & (df["Event"] == 0)
    calib = df[calib_mask]
    stats: dict[str, tuple[float, float]] = {}
    for bus in PMU_BUSES:
        for suf in _SUFFIXES:
            col = f"BUS{bus}_{suf}"
            if col in calib.columns:
                vals = calib[col].dropna().to_numpy(float)
                if len(vals) > 1:
                    stats[col] = (float(vals.mean()), float(vals.std()))
                else:
                    stats[col] = (0.0, 0.0)
    return stats


# ── measurement noise injection ────────────────────────────────────────────────

def _make_noise(rng: np.random.Generator, col: str, T: int,
                stats: dict) -> np.ndarray:
    """Generate realistic measurement noise for a column from calibration stats."""
    mean, std = stats.get(col, (0.0, 0.0))
    if std < 1e-12:
        return np.full(T, mean)
    return mean + rng.standard_normal(T) * std


# ── base signal builder ────────────────────────────────────────────────────────

class SyntheticWindow:
    """Holds the (T, 14) measurement array for one bus during one event window."""

    def __init__(
        self,
        bus: int,
        T: int,
        stats: dict,
        rng: np.random.Generator,
    ):
        self.bus = bus
        self.T = T
        self.rng = rng
        self.suffixes = _SUFFIXES
        self.data: dict[str, np.ndarray] = {}
        # Fill with calibration noise for all channels
        for suf in self.suffixes:
            col = f"BUS{bus}_{suf}"
            self.data[col] = _make_noise(rng, col, T, stats)

    def col(self, suf: str) -> np.ndarray:
        return self.data[f"BUS{self.bus}_{suf}"]

    def set(self, suf: str, arr: np.ndarray) -> None:
        self.data[f"BUS{self.bus}_{suf}"] = arr

    def to_series_dict(self) -> dict[str, np.ndarray]:
        return dict(self.data)


# ── per-event generators ───────────────────────────────────────────────────────

def _pmu_bus_idx(bus: int) -> int:
    return PMU_BUSES.index(bus) if bus in PMU_BUSES else -1


_IEEE39_BRANCHES = [
    (1, 2), (1, 39), (2, 3), (2, 25), (3, 4), (3, 18), (4, 5), (4, 14),
    (5, 6), (5, 8), (6, 7), (6, 11), (7, 8), (8, 9), (9, 39), (10, 11),
    (10, 13), (13, 14), (14, 15), (15, 16), (16, 17), (16, 19), (16, 21),
    (16, 24), (17, 18), (17, 27), (21, 22), (22, 23), (23, 24), (25, 26),
    (26, 27), (26, 28), (26, 29), (28, 29), (12, 11), (12, 13),
    (30, 2), (31, 6), (32, 10), (33, 19), (34, 20), (35, 22), (36, 23),
    (37, 25), (38, 29), (39, 9),
]


def _topological_distance(src: int, dst: int) -> int:
    """Shortest-path distance on the IEEE-39 branch graph."""
    if src == dst:
        return 0
    graph: dict[int, list[int]] = {}
    for a, b in _IEEE39_BRANCHES:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    queue: list[tuple[int, int]] = [(src, 0)]
    seen = {src}
    while queue:
        node, dist = queue.pop(0)
        for nb in graph.get(node, []):
            if nb == dst:
                return dist + 1
            if nb not in seen:
                seen.add(nb)
                queue.append((nb, dist + 1))
    return 99


def _topology_weight(event_bus: int, pmu_bus: int, tau: float = 2.5) -> float:
    """Smooth spatial attenuation from an event bus to a PMU bus."""
    d = _topological_distance(event_bus, pmu_bus)
    return float(np.exp(-d / tau))


def _voltage_drop_profile(
    t: np.ndarray,
    t_fault: float,
    t_clear: float,
    drop_frac: float,
    tau_osc: float = 0.2,
    f_osc: float = 1.2,
) -> np.ndarray:
    """Generate voltage profile with fault dip and post-fault oscillation."""
    v = np.ones_like(t)
    during = (t >= t_fault) & (t < t_clear)
    after  = t >= t_clear
    v[during] = 1.0 - drop_frac
    # Post-fault oscillation (decaying sinusoid)
    v[after] = 1.0 + (drop_frac * 0.3) * np.exp(-(t[after] - t_clear) / tau_osc) * \
               np.sin(2 * np.pi * f_osc * (t[after] - t_clear))
    return v


def _freq_transient(
    t: np.ndarray,
    t_event: float,
    delta_p_pu: float,
    H_total: float = 120.0,
    D: float = 5.0,
    f0: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Freq and ROCOF for a power step ΔP_pu (p.u. on system base).

    Simple first-order inertia model:
        df/dt = -(f0 / (2H)) * ΔP - (D * f0 / (2H)) * Δf
    with RK4 integration.
    """
    freq = np.full_like(t, f0)
    rocof = np.zeros_like(t)

    dt = t[1] - t[0] if len(t) > 1 else DT
    f_state = f0    # current frequency
    Df = 0.0       # frequency deviation

    for i, ti in enumerate(t):
        if ti < t_event:
            freq[i]  = f0
            rocof[i] = 0.0
            continue
        # RK4 for df/dt = -(f0/(2H)) * ΔP - (D*f0/(2H)) * Δf
        def deriv(df_in: float) -> float:
            return -(f0 / (2 * H_total)) * delta_p_pu - (D * f0 / (2 * H_total)) * df_in

        k1 = deriv(Df)
        k2 = deriv(Df + dt/2 * k1)
        k3 = deriv(Df + dt/2 * k2)
        k4 = deriv(Df + dt * k3)
        dDf = dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        Df += dDf
        freq[i]  = f0 + Df
        rocof[i] = deriv(Df)   # Hz/s = -f0/(2H)*ΔP - D*f0/(2H)*Δf

    return freq, rocof


def generate_fault(
    event_bus: int,
    stats: dict,
    rng: np.random.Generator,
    window_sec: float = 6.0,
    t_event: float = 2.0,
    fault_impedance: float = 0.0,
    clear_cycles: int = 5,
    delta_p_pu: float | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Generate a 3LG fault at event_bus.

    Returns (bus_data_dict, timestamps, event_labels).
    """
    T = int(round(window_sec * FPS))
    t = np.arange(T) / FPS
    t_clear = t_event + clear_cycles / 60.0  # clear time in seconds

    # Voltage drop proportional to fault type (0=bolted, higher=high-impedance)
    drop_base = max(0.4, 1.0 - 1.0 / (1.0 + fault_impedance * 10.0))

    event_labels = np.zeros(T, dtype=int)
    event_labels[(t >= t_event) & (t < t_clear)] = 1  # fault label

    result: dict[str, np.ndarray] = {}
    for bus in PMU_BUSES:
        win = SyntheticWindow(bus, T, stats, rng)

        # Electrical distance factor: faulted bus drops most
        if bus == event_bus:
            drop = drop_base
            i_factor = 3.0    # current surge at faulted bus
        else:
            # Approximate voltage drop via 1/(1 + distance)
            # Use a simple heuristic: closer PMU buses see larger dip
            bus_dists = {
                2: [10, 5, 8, 7, 14, 15, 16, 2],    # rough electrical distances from PMU buses
                5: [5, 2, 4, 3, 10, 11, 12, 5],
                6: [6, 3, 2, 4, 11, 12, 13, 6],
                10: [7, 4, 5, 2, 9, 10, 11, 5],
                19: [14, 10, 11, 9, 2, 5, 6, 12],
                22: [15, 11, 12, 10, 5, 2, 4, 13],
                29: [16, 12, 13, 11, 6, 4, 2, 14],
                39: [2, 5, 6, 5, 12, 13, 14, 1],
            }
            if event_bus in bus_dists:
                pmu_idx = PMU_BUSES.index(bus)
                dist = bus_dists[event_bus][pmu_idx]
            else:
                dist = 8  # default medium distance
            drop = drop_base / (1 + 0.2 * dist)
            i_factor = max(1.0, 2.0 / (1 + 0.1 * dist))

        v_profile = _voltage_drop_profile(t, t_event, t_clear, drop)
        # Apply to VA_MAG (and approximately VB, VC for balanced three-phase)
        base_vmag = stats.get(f"BUS{bus}_VA_MAG", (200_000.0, 500.0))[0]
        for vphase in ("VA_MAG", "VB_MAG", "VC_MAG"):
            win.set(vphase, win.col(vphase) * v_profile * base_vmag /
                    max(stats.get(f"BUS{bus}_{vphase}", (base_vmag, 1.0))[0], 1.0))

        # Current surge during fault
        base_imag = stats.get(f"BUS{bus}_IA_MAG", (500.0, 5.0))[0]
        i_profile = np.where(
            (t >= t_event) & (t < t_clear),
            i_factor,
            1.0 + 0.05 * np.exp(-(t - t_clear) / 0.3) * np.where(t >= t_clear, 1, 0)
        )
        for iphase in ("IA_MAG", "IB_MAG", "IC_MAG"):
            win.set(iphase, win.col(iphase) * i_profile)

        # Frequency and ROCOF transient
        dp = delta_p_pu if delta_p_pu is not None else rng.uniform(0.1, 0.3)
        freq, rocof = _freq_transient(t, t_event, dp)
        win.set("Freq", freq + rng.standard_normal(T) * 0.002)
        win.set("ROCOF", rocof + rng.standard_normal(T) * 0.005)

        result.update(win.to_series_dict())

    return result, t, event_labels


def generate_line_outage(
    line_from: int,
    line_to: int,
    stats: dict,
    rng: np.random.Generator,
    window_sec: float = 6.0,
    t_event: float = 2.0,
    delta_p_pu: float | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Generate a line outage between line_from and line_to buses."""
    T = int(round(window_sec * FPS))
    t = np.arange(T) / FPS

    event_labels = np.zeros(T, dtype=int)
    event_labels[t >= t_event] = 2  # line outage label

    if delta_p_pu is None:
        delta_p_pu = rng.choice([-1, 1]) * rng.uniform(0.05, 0.2)

    result: dict[str, np.ndarray] = {}
    for bus in PMU_BUSES:
        win = SyntheticWindow(bus, T, stats, rng)

        # Post-outage voltage/angle perturbation. Attenuation follows shortest
        # path distance to either endpoint, which makes non-PMU line 24-23
        # produce strong but indirect signatures at Bus22/19/29.
        w = max(_topology_weight(line_from, bus), _topology_weight(line_to, bus))
        vmag_delta_frac = rng.choice([-1, 1]) * rng.uniform(0.002, 0.010) * w

        step = np.where(t >= t_event, 1.0, 0.0)
        for vphase in ("VA_MAG", "VB_MAG", "VC_MAG"):
            base = stats.get(f"BUS{bus}_{vphase}", (200_000.0, 500.0))[0]
            win.set(vphase, win.col(vphase) + vmag_delta_frac * base * step)
        angle_shift = rng.choice([-1, 1]) * rng.uniform(0.03, 0.25) * w * step
        for aphase in ("VA_ANG", "VB_ANG", "VC_ANG"):
            win.set(aphase, win.col(aphase) + angle_shift)

        # Frequency and ROCOF
        freq, rocof = _freq_transient(t, t_event, delta_p_pu)
        win.set("Freq", freq + rng.standard_normal(T) * 0.002)
        win.set("ROCOF", rocof + rng.standard_normal(T) * 0.005)

        result.update(win.to_series_dict())

    return result, t, event_labels


def _bus_neighbors(bus: int) -> list[int]:
    """Return PMU buses adjacent to the given bus (rough topology heuristic)."""
    # Approximate adjacency for the 8 PMU buses in IEEE 39
    neighbors = {
        2:  [39, 3, 25],
        5:  [4, 6, 8],
        6:  [5, 7, 11],
        10: [11, 13, 32],
        19: [16, 20, 33],
        22: [21, 23],
        29: [28],
        39: [1, 9],
    }
    return neighbors.get(bus, [])


def generate_gen_change(
    gen_bus: int,
    stats: dict,
    rng: np.random.Generator,
    window_sec: float = 6.0,
    t_event: float = 2.0,
    delta_mw: float | None = None,
    base_mva: float = 100.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Generate a generation step change at gen_bus."""
    T = int(round(window_sec * FPS))
    t = np.arange(T) / FPS

    event_labels = np.zeros(T, dtype=int)
    event_labels[t >= t_event] = 3

    if delta_mw is None:
        delta_mw = rng.choice([-1, 1]) * rng.uniform(5.0, 50.0)
    delta_pu = delta_mw / base_mva

    result: dict[str, np.ndarray] = {}
    for bus in PMU_BUSES:
        win = SyntheticWindow(bus, T, stats, rng)

        # IA_MAG at gen_bus changes proportionally
        if bus == gen_bus:
            base_i = stats.get(f"BUS{bus}_IA_MAG", (500.0, 5.0))[0]
            di_frac = delta_pu * 0.3  # rough proportionality
            step = np.where(t >= t_event, 1.0, 0.0)
            for iphase in ("IA_MAG", "IB_MAG", "IC_MAG"):
                win.set(iphase, win.col(iphase) + di_frac * base_i * step)

        freq, rocof = _freq_transient(t, t_event, -delta_pu)
        win.set("Freq", freq + rng.standard_normal(T) * 0.002)
        win.set("ROCOF", rocof + rng.standard_normal(T) * 0.005)

        result.update(win.to_series_dict())

    return result, t, event_labels


def generate_load_change(
    load_bus: int,
    stats: dict,
    rng: np.random.Generator,
    window_sec: float = 6.0,
    t_event: float = 2.0,
    delta_mw: float | None = None,
    base_mva: float = 100.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Generate a load step change at load_bus."""
    T = int(round(window_sec * FPS))
    t = np.arange(T) / FPS

    event_labels = np.zeros(T, dtype=int)
    event_labels[t >= t_event] = 4

    if delta_mw is None:
        delta_mw = rng.choice([-1, 1]) * rng.uniform(5.0, 50.0)
    delta_pu = delta_mw / base_mva

    result: dict[str, np.ndarray] = {}
    for bus in PMU_BUSES:
        win = SyntheticWindow(bus, T, stats, rng)

        # Voltage/current sag or surge attenuates over the network. This makes
        # Bus7 load changes visible at nearby PMUs 5/6/10 without pretending a
        # PMU exists at Bus7.
        w = _topology_weight(load_bus, bus)
        vmag_delta = rng.uniform(0.001, 0.007) * np.sign(-delta_pu) * w
        imag_delta = rng.uniform(0.01, 0.08) * np.sign(delta_pu) * w
        step = np.where(t >= t_event, 1.0, 0.0)
        for vphase in ("VA_MAG", "VB_MAG", "VC_MAG"):
            base = stats.get(f"BUS{bus}_{vphase}", (200_000.0, 500.0))[0]
            win.set(vphase, win.col(vphase) + vmag_delta * base * step)
        for iphase in ("IA_MAG", "IB_MAG", "IC_MAG"):
            base = stats.get(f"BUS{bus}_{iphase}", (585.0, 5.0))[0]
            win.set(iphase, win.col(iphase) + imag_delta * base * step)

        # Load increase → frequency drops (positive delta_pu means load increase)
        freq, rocof = _freq_transient(t, t_event, delta_pu)
        win.set("Freq", freq + rng.standard_normal(T) * 0.002)
        win.set("ROCOF", rocof + rng.standard_normal(T) * 0.005)

        result.update(win.to_series_dict())

    return result, t, event_labels


def generate_pmu_dropout(
    dropout_bus: int,
    stats: dict,
    rng: np.random.Generator,
    window_sec: float = 6.0,
    t_event: float = 2.0,
    dropout_sec: float | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Generate PMU data dropout at dropout_bus (DATA_PRESENT=0, all NaN)."""
    T = int(round(window_sec * FPS))
    t = np.arange(T) / FPS

    if dropout_sec is None:
        dropout_sec = rng.uniform(0.5, 3.0)
    t_recover = min(t_event + dropout_sec, window_sec - 0.5)

    event_labels = np.zeros(T, dtype=int)
    event_labels[(t >= t_event) & (t < t_recover)] = 5

    result: dict[str, np.ndarray] = {}
    for bus in PMU_BUSES:
        win = SyntheticWindow(bus, T, stats, rng)

        # Only dropout_bus goes missing
        if bus == dropout_bus:
            missing_mask = (t >= t_event) & (t < t_recover)
            for suf in _SUFFIXES:
                arr = win.col(suf).copy()
                arr[missing_mask] = np.nan
                win.set(suf, arr)

        result.update(win.to_series_dict())

    # DATA_PRESENT: separate array per bus (stored as BUSk_DATA_PRESENT in result)
    for bus in PMU_BUSES:
        dp = np.ones(T, dtype=float)
        bus_labels = np.zeros(T, dtype=int)
        if bus == dropout_bus:
            dp[(t >= t_event) & (t < t_recover)] = 0.0
            bus_labels = event_labels.copy()
        result[f"BUS{bus}_DATA_PRESENT"] = dp
        result[f"BUS{bus}_Event"] = bus_labels

    return result, t, event_labels


def generate_bad_data(
    bad_bus: int,
    stats: dict,
    rng: np.random.Generator,
    window_sec: float = 6.0,
    t_event: float = 2.0,
    duration_sec: float | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Generate a label-7 bad-data burst with DATA_PRESENT still true."""
    T = int(round(window_sec * FPS))
    t = np.arange(T) / FPS
    if duration_sec is None:
        duration_sec = float(rng.uniform(0.4, 1.1))
    t_end = min(window_sec - 0.5, t_event + duration_sec)
    bad_mask = (t >= t_event) & (t < t_end)

    event_labels = np.zeros(T, dtype=int)
    event_labels[bad_mask] = 7

    result: dict[str, np.ndarray] = {}
    corrupt_phase = str(rng.choice(["VA_MAG", "VB_MAG", "VC_MAG"]))
    scale = float(rng.choice([0.90, 0.92, 1.08, 1.10, 1.12]))
    for bus in PMU_BUSES:
        win = SyntheticWindow(bus, T, stats, rng)
        if bus == bad_bus:
            arr = win.col(corrupt_phase).copy()
            arr[bad_mask] = arr[bad_mask] * scale
            win.set(corrupt_phase, arr)
        result.update(win.to_series_dict())
        result[f"BUS{bus}_DATA_PRESENT"] = np.ones(T, dtype=float)
        result[f"BUS{bus}_Event"] = np.where(bus == bad_bus, event_labels, 0).astype(int)

    return result, t, event_labels


def generate_post_cyber_physical(
    stats: dict,
    rng: np.random.Generator,
    physical: str = "gen",
    dropout_bus: int = 29,
    window_sec: float = 8.0,
    t_dropout: float = 1.0,
    t_physical: float = 2.5,
    t_recover: float = 4.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Generate a dropout followed by/overlapping a physical event.

    Labels mirror the difficult real sequence:
      label 5: missing PMU data only,
      label 6: missing PMU data concurrent with a physical event,
      label 3/4/2: physical event after PMU recovery.
    """
    if physical == "gen":
        bus_data, t, _ = generate_gen_change(
            gen_bus=2, stats=stats, rng=rng, window_sec=window_sec,
            t_event=t_physical, delta_mw=float(rng.choice([-1, 1])) * rng.uniform(10.0, 60.0),
        )
        physical_label = 3
    elif physical == "load":
        bus_data, t, _ = generate_load_change(
            load_bus=7, stats=stats, rng=rng, window_sec=window_sec,
            t_event=t_physical, delta_mw=float(rng.choice([-1, 1])) * rng.uniform(10.0, 60.0),
        )
        physical_label = 4
    elif physical == "line":
        bus_data, t, _ = generate_line_outage(
            line_from=24, line_to=23, stats=stats, rng=rng,
            window_sec=window_sec, t_event=t_physical,
            delta_p_pu=float(rng.choice([-1, 1])) * rng.uniform(0.05, 0.25),
        )
        physical_label = 2
    else:
        raise ValueError("physical must be one of: 'gen', 'load', 'line'")

    labels = np.zeros(len(t), dtype=int)
    dropout_mask = (t >= t_dropout) & (t < t_recover)
    physical_mask = t >= t_physical
    labels[dropout_mask] = 5
    labels[physical_mask] = physical_label
    labels[dropout_mask & physical_mask] = 6

    for bus in PMU_BUSES:
        dp = np.ones(len(t), dtype=float)
        if bus == dropout_bus:
            dp[dropout_mask] = 0.0
            for suf in _SUFFIXES:
                col = f"BUS{bus}_{suf}"
                arr = bus_data[col].copy()
                arr[dropout_mask] = np.nan
                bus_data[col] = arr
        bus_data[f"BUS{bus}_DATA_PRESENT"] = dp
        if bus == dropout_bus:
            bus_data[f"BUS{bus}_Event"] = labels.copy()
        else:
            bus_labels = np.zeros(len(t), dtype=int)
            bus_labels[physical_mask] = physical_label
            bus_data[f"BUS{bus}_Event"] = bus_labels

    return bus_data, t, labels


# ── CSV writer ─────────────────────────────────────────────────────────────────

def _write_synthetic_csvs(
    run_id: int,
    bus_data: dict[str, np.ndarray],
    timestamps: np.ndarray,
    event_labels: np.ndarray,
    out_dir: Path,
) -> list[Path]:
    """Write one synthetic event as 8 bus CSVs in competition format."""
    paths = []
    for bus in PMU_BUSES:
        rows = {"TIMESTAMP": np.round(timestamps, 10)}
        for suf in _SUFFIXES:
            col = f"BUS{bus}_{suf}"
            rows[col] = bus_data.get(col, np.zeros(len(timestamps)))
        rows["DATA_PRESENT"] = bus_data.get(f"BUS{bus}_DATA_PRESENT",
                                             np.ones(len(timestamps))).astype(int)
        rows["Event"] = bus_data.get(f"BUS{bus}_Event", event_labels).astype(int)

        df = pd.DataFrame(rows)
        fname = out_dir / f"syn{run_id:04d}_Bus{bus}_Competition_Data_nanmask.csv"
        df.to_csv(fname, index=False)
        paths.append(fname)
    return paths


# ── main generator ─────────────────────────────────────────────────────────────

# IEEE 39-bus load buses (PQ buses) and generator buses
_ALL_BUSES = list(range(1, 40))
_GEN_BUSES = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
_LOAD_BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31, 39]
# Branch list from IEEE 39-bus topology
_BRANCHES = _IEEE39_BRANCHES
DEFAULT_SCENARIO_MIX: dict[int, int] = {
    1: 850,
    2: 700,
    3: 650,
    4: 700,
    5: 550,
    6: 500,
    7: 650,
    8: 400,
}
LABEL_NAMES: dict[int, str] = {
    0: "normal",
    1: "fault",
    2: "line_outage",
    3: "generation_change",
    4: "load_change",
    5: "pmu_dropout",
    6: "pmu_dropout_plus_physical",
    7: "bad_data",
    8: "unknown_multi_event",
}


def generate_all(
    n_faults: int = 40,
    n_line_outages: int = 40,
    n_gen_changes: int = 40,
    n_load_changes: int = 40,
    n_dropouts: int = 20,
    n_bus7_load_changes: int = 30,
    n_line_2423_outages: int = 30,
    n_post_cyber_physical: int = 30,
    n_bad_data: int = 0,
    n_unknown: int = 0,
    data_dir: Path | str = "data/raw",
    out_dir: Path | str = "data/synthetic",
    seed: int = 42,
) -> dict:
    """Generate all synthetic events and write CSVs.

    Args:
        n_faults:       Number of fault events to generate.
        n_line_outages: Number of line outage events.
        n_gen_changes:  Number of generation step change events.
        n_load_changes: Number of load step change events.
        n_dropouts:     Number of PMU dropout events.
        n_bus7_load_changes: targeted Bus 7 load-change scenarios.
        n_line_2423_outages: targeted Bus 24-23 line-outage scenarios.
        n_post_cyber_physical: targeted dropout+physical recovery scenarios.
        n_bad_data:     Number of label-7 bad-data phase-corruption events.
        n_unknown:      Number of label-8 multi-event/unknown scenarios.
        data_dir:       Path to competition raw CSVs (for baseline statistics).
        out_dir:        Output directory for synthetic CSVs.
        seed:           Random seed for reproducibility.

    Returns:
        dict with keys 'n_generated', 'out_dir', 'event_counts' by label.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load real data to get calibration statistics
    log.info("Loading real data for baseline statistics...")
    df = load_all(data_dir)
    stats = extract_normal_baseline(df)
    log.info("Calibration stats computed from %d normal frames", int(
        ((df["TIMESTAMP"] <= 60.0) & (df["Event"] == 0)).sum()
    ))

    n_generated = 0
    event_counts: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    # ── 1. Three-phase faults ─────────────────────────────────────────────────
    fault_buses = rng.choice(_ALL_BUSES, size=n_faults, replace=True)
    for i, bus in enumerate(fault_buses):
        try:
            fault_imp = float(rng.choice([0.0, 0.01, 0.05, 0.1, 0.2]))
            clear_cyc = int(rng.integers(3, 12))
            t_event   = float(rng.uniform(1.0, 3.0))
            dp = float(rng.uniform(0.05, 0.4))
            bus_data, ts, labels = generate_fault(
                event_bus=bus, stats=stats, rng=rng,
                t_event=t_event, fault_impedance=fault_imp,
                clear_cycles=clear_cyc, delta_p_pu=dp,
            )
            _write_synthetic_csvs(n_generated, bus_data, ts, labels, out_dir)
            n_generated += 1
            event_counts[1] += 1
        except Exception as e:
            log.warning("Fault %d failed: %s", i, e)

    # ── 2. Line outages ────────────────────────────────────────────────────────
    branch_subset = [_BRANCHES[i] for i in rng.choice(len(_BRANCHES), size=n_line_outages, replace=True)]
    for i, (b1, b2) in enumerate(branch_subset):
        try:
            t_event = float(rng.uniform(1.0, 3.0))
            dp = float(rng.choice([-1, 1])) * float(rng.uniform(0.03, 0.2))
            bus_data, ts, labels = generate_line_outage(
                line_from=b1, line_to=b2, stats=stats, rng=rng,
                t_event=t_event, delta_p_pu=dp,
            )
            _write_synthetic_csvs(n_generated, bus_data, ts, labels, out_dir)
            n_generated += 1
            event_counts[2] += 1
        except Exception as e:
            log.warning("Line outage %d failed: %s", i, e)

    # ── 3. Generation step changes ─────────────────────────────────────────────
    gen_subset = rng.choice(_GEN_BUSES, size=n_gen_changes, replace=True)
    for i, bus in enumerate(gen_subset):
        try:
            t_event  = float(rng.uniform(1.0, 3.0))
            delta_mw = float(rng.choice([-1, 1])) * float(rng.uniform(5.0, 50.0))
            bus_data, ts, labels = generate_gen_change(
                gen_bus=bus, stats=stats, rng=rng,
                t_event=t_event, delta_mw=delta_mw,
            )
            _write_synthetic_csvs(n_generated, bus_data, ts, labels, out_dir)
            n_generated += 1
            event_counts[3] += 1
        except Exception as e:
            log.warning("Gen change %d failed: %s", i, e)

    # ── 4. Load step changes ───────────────────────────────────────────────────
    load_subset = rng.choice(_LOAD_BUSES, size=n_load_changes, replace=True)
    for i, bus in enumerate(load_subset):
        try:
            t_event  = float(rng.uniform(1.0, 3.0))
            delta_mw = float(rng.choice([-1, 1])) * float(rng.uniform(5.0, 50.0))
            bus_data, ts, labels = generate_load_change(
                load_bus=bus, stats=stats, rng=rng,
                t_event=t_event, delta_mw=delta_mw,
            )
            _write_synthetic_csvs(n_generated, bus_data, ts, labels, out_dir)
            n_generated += 1
            event_counts[4] += 1
        except Exception as e:
            log.warning("Load change %d failed: %s", i, e)

    # ── 5. PMU dropouts ────────────────────────────────────────────────────────
    pmu_subset = rng.choice(PMU_BUSES, size=n_dropouts, replace=True)
    for i, bus in enumerate(pmu_subset):
        try:
            t_event     = float(rng.uniform(1.0, 3.0))
            dropout_sec = float(rng.uniform(0.5, 3.0))
            bus_data, ts, labels = generate_pmu_dropout(
                dropout_bus=bus, stats=stats, rng=rng,
                t_event=t_event, dropout_sec=dropout_sec,
            )
            _write_synthetic_csvs(n_generated, bus_data, ts, labels, out_dir)
            n_generated += 1
            event_counts[5] += 1
        except Exception as e:
            log.warning("PMU dropout %d failed: %s", i, e)

    bad_subset = rng.choice(PMU_BUSES, size=n_bad_data, replace=True)
    for i, bus in enumerate(bad_subset):
        try:
            t_event = float(rng.uniform(1.0, 3.0))
            bus_data, ts, labels = generate_bad_data(
                bad_bus=bus, stats=stats, rng=rng, t_event=t_event,
            )
            _write_synthetic_csvs(n_generated, bus_data, ts, labels, out_dir)
            n_generated += 1
            event_counts[7] += 1
        except Exception as e:
            log.warning("Bad data %d failed: %s", i, e)

    # ── 6. Targeted hard cases: Bus7, line 24-23, and post-cyber physical ────
    for i in range(n_bus7_load_changes):
        try:
            bus_data, ts, labels = generate_load_change(
                load_bus=7, stats=stats, rng=rng,
                t_event=float(rng.uniform(1.0, 3.0)),
                delta_mw=float(rng.choice([-1, 1])) * float(rng.uniform(10.0, 70.0)),
            )
            _write_synthetic_csvs(n_generated, bus_data, ts, labels, out_dir)
            n_generated += 1
            event_counts[4] += 1
        except Exception as e:
            log.warning("Target Bus7 load %d failed: %s", i, e)

    for i in range(n_line_2423_outages):
        try:
            bus_data, ts, labels = generate_line_outage(
                line_from=24, line_to=23, stats=stats, rng=rng,
                t_event=float(rng.uniform(1.0, 3.0)),
                delta_p_pu=float(rng.choice([-1, 1])) * float(rng.uniform(0.05, 0.25)),
            )
            _write_synthetic_csvs(n_generated, bus_data, ts, labels, out_dir)
            n_generated += 1
            event_counts[2] += 1
        except Exception as e:
            log.warning("Target line 24-23 %d failed: %s", i, e)

    physical_cycle = ["gen", "load", "line"]
    for i in range(n_post_cyber_physical):
        try:
            physical = physical_cycle[i % len(physical_cycle)]
            bus_data, ts, labels = generate_post_cyber_physical(
                stats=stats, rng=rng, physical=physical, dropout_bus=29,
                t_dropout=float(rng.uniform(0.8, 1.4)),
                t_physical=float(rng.uniform(2.0, 3.0)),
                t_recover=float(rng.uniform(3.5, 4.8)),
            )
            _write_synthetic_csvs(n_generated, bus_data, ts, labels, out_dir)
            n_generated += 1
            event_counts[6] += 1
        except Exception as e:
            log.warning("Target post-cyber physical %d failed: %s", i, e)

    for i in range(n_unknown):
        try:
            bus_data, ts, labels = generate_unknown_event(stats=stats, rng=rng)
            _write_synthetic_csvs(n_generated, bus_data, ts, labels, out_dir)
            n_generated += 1
            event_counts[8] += 1
        except Exception as e:
            log.warning("Unknown/multi-event %d failed: %s", i, e)

    log.info(
        "Generated %d synthetic events: %s",
        n_generated, event_counts,
    )
    return {
        "n_generated": n_generated,
        "out_dir": str(out_dir),
        "event_counts": event_counts,
    }


def generate_unknown_event(
    stats: dict,
    rng: np.random.Generator,
    window_sec: float = 8.0,
    t_event: float = 2.0,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Generate a label-8 multi-event/unknown signature.

    The competition schema reserves label 8 for events that do not cleanly match
    one of the visible classes.  We synthesize these as overlapping weak
    physical effects plus one localized telemetry distortion, so the classifier
    sees label 8 as "explainable but out-of-family" rather than random noise.
    """
    physical = str(rng.choice(["fault", "load", "line", "gen"]))
    if physical == "fault":
        bus_data, t, _ = generate_fault(
            event_bus=int(rng.choice(_ALL_BUSES)),
            stats=stats,
            rng=rng,
            window_sec=window_sec,
            t_event=t_event,
            fault_impedance=float(rng.uniform(0.05, 0.3)),
            clear_cycles=int(rng.integers(3, 8)),
            delta_p_pu=float(rng.uniform(0.02, 0.12)),
        )
    elif physical == "line":
        line = _BRANCHES[int(rng.integers(0, len(_BRANCHES)))]
        bus_data, t, _ = generate_line_outage(
            line_from=line[0],
            line_to=line[1],
            stats=stats,
            rng=rng,
            window_sec=window_sec,
            t_event=t_event,
            delta_p_pu=float(rng.choice([-1, 1])) * float(rng.uniform(0.02, 0.12)),
        )
    elif physical == "gen":
        bus_data, t, _ = generate_gen_change(
            gen_bus=int(rng.choice(_GEN_BUSES)),
            stats=stats,
            rng=rng,
            window_sec=window_sec,
            t_event=t_event,
            delta_mw=float(rng.choice([-1, 1])) * float(rng.uniform(5.0, 35.0)),
        )
    else:
        bus_data, t, _ = generate_load_change(
            load_bus=int(rng.choice(_LOAD_BUSES)),
            stats=stats,
            rng=rng,
            window_sec=window_sec,
            t_event=t_event,
            delta_mw=float(rng.choice([-1, 1])) * float(rng.uniform(5.0, 35.0)),
        )

    labels = np.zeros(len(t), dtype=int)
    labels[(t >= t_event) & (t < window_sec - 0.5)] = 8

    # Add a mild phase or frequency distortion at a PMU while keeping data
    # present, which makes this distinguishable from pure cyber dropout.
    bad_bus = int(rng.choice(PMU_BUSES))
    bad_mask = (t >= t_event + float(rng.uniform(0.2, 0.8))) & (t < window_sec - 0.8)
    distortion = str(rng.choice(["angle", "frequency", "current"]))
    if distortion == "angle":
        for suffix in ("VA_ANG", "VB_ANG", "VC_ANG"):
            col = f"BUS{bad_bus}_{suffix}"
            bus_data[col] = bus_data[col] + bad_mask.astype(float) * float(rng.uniform(0.4, 1.5))
    elif distortion == "frequency":
        col = f"BUS{bad_bus}_Freq"
        bus_data[col] = bus_data[col] + bad_mask.astype(float) * float(rng.uniform(0.02, 0.08))
    else:
        for suffix in ("IA_MAG", "IB_MAG", "IC_MAG"):
            col = f"BUS{bad_bus}_{suffix}"
            bus_data[col] = bus_data[col] * (1.0 + bad_mask.astype(float) * float(rng.uniform(0.04, 0.12)))

    for bus in PMU_BUSES:
        bus_data.setdefault(f"BUS{bus}_DATA_PRESENT", np.ones(len(t), dtype=float))
    return bus_data, t, labels


def scenario_counts_for_total(n_scenarios: int) -> dict[int, int]:
    """Scale the default 5,000-scenario mix to any requested size."""
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    labels = sorted(DEFAULT_SCENARIO_MIX)
    if n_scenarios <= len(labels):
        return {label: 1 if i < n_scenarios else 0 for i, label in enumerate(labels)}

    total_default = sum(DEFAULT_SCENARIO_MIX.values())
    raw = {
        label: DEFAULT_SCENARIO_MIX[label] * n_scenarios / total_default
        for label in labels
    }
    counts = {label: int(np.floor(value)) for label, value in raw.items()}
    for label in labels:
        if counts[label] == 0:
            counts[label] = 1
    remainder = n_scenarios - sum(counts.values())
    order = sorted(labels, key=lambda label: raw[label] - np.floor(raw[label]), reverse=True)
    if remainder > 0:
        for i in range(remainder):
            counts[order[i % len(order)]] += 1
    elif remainder < 0:
        for label in reversed(order):
            while remainder < 0 and counts[label] > 1:
                counts[label] -= 1
                remainder += 1
            if remainder == 0:
                break
    return counts


def _feature_cache_grid(data_dir: Path) -> tuple[dict[int, np.ndarray] | None, object | None]:
    """Best-effort grid setup for synthetic feature caches.

    Feature caching should never make augmentation fail.  If the RAW file or
    topology estimator is unavailable, the cache is still written with the core
    residual features and zeros for topology-only features.
    """
    try:
        from src.grid.jacobians import bus_sensitivity_columns, compute_jacobians
        from src.grid.load_case import load_case
        from src.estimator.topology_state import TopologyStateEstimator

        raw_path = data_dir.parent / "metadata" / "IEEE 39 Bus Power System.raw"
        if not raw_path.exists():
            return None, None
        grid = load_case(raw_path)
        J_cols = bus_sensitivity_columns(compute_jacobians(grid), grid)
        return J_cols, TopologyStateEstimator(grid)
    except Exception as exc:
        log.warning("Synthetic feature cache will omit grid features: %s", exc)
        return None, None


def scenario_label_and_onset(labels: np.ndarray) -> tuple[int, int]:
    """Return the scenario-level training label and representative event frame."""
    labels = np.asarray(labels, dtype=int)
    present = sorted({int(v) for v in labels.tolist()} - {0})
    if not present:
        return 0, min(len(labels) - 1, int(round(2.0 * FPS))) if len(labels) else 0

    # Mixed cyber+physical windows can include 5, 6, and a physical tail label.
    # The cache should represent the scenario family, not whichever label lands
    # at the midpoint of all nonzero frames.
    priority = (8, 6, 7, 5, 1, 2, 3, 4)
    label = next((candidate for candidate in priority if candidate in present), present[0])
    frames = np.where(labels == label)[0]
    if len(frames) == 0:
        frames = np.where(labels != 0)[0]
    return int(label), int(frames[len(frames) // 2])


def write_feature_cache(
    syn_dir: Path | str,
    feature_cache: Path | str,
    *,
    data_dir: Path | str = "data/raw",
    fps: float = FPS,
    window_sec: float = 3.0,
) -> dict:
    """Extract and cache synthetic feature windows as an ``.npz`` file."""
    from src.classifier.features import N_FEATURES, extract_features

    syn_dir = Path(syn_dir)
    feature_cache = Path(feature_cache)
    run_ids: list[int] = []
    for p in syn_dir.glob("syn*_Bus2_Competition_Data_nanmask.csv"):
        try:
            run_ids.append(int(p.name[3:7]))
        except ValueError:
            continue
    run_ids = sorted(set(run_ids))
    J_cols, state_estimator = _feature_cache_grid(Path(data_dir))

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    kept_ids: list[int] = []
    for run_id in run_ids:
        try:
            df_syn = load_synthetic(syn_dir, run_id)
            ev = df_syn["Event"].to_numpy(dtype=int)
            label, onset = scenario_label_and_onset(ev)
            if label == 0:
                continue
            y_rows.append(int(label))
            X_rows.append(
                extract_features(
                    df_syn,
                    onset,
                    J_cols=J_cols,
                    state_estimator=state_estimator,
                    fps=fps,
                    window_sec=window_sec,
                )
            )
            kept_ids.append(run_id)
        except Exception as exc:
            log.debug("Skipping synthetic feature cache run %d: %s", run_id, exc)

    X = np.stack(X_rows, axis=0) if X_rows else np.zeros((0, N_FEATURES), dtype=float)
    y = np.asarray(y_rows, dtype=int)
    feature_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        feature_cache,
        X=X,
        y=y,
        run_ids=np.asarray(kept_ids, dtype=int),
        feature_count=np.asarray([N_FEATURES], dtype=int),
    )
    return {
        "feature_cache": str(feature_cache),
        "n_windows": int(len(y)),
        "label_counts": {str(k): int(v) for k, v in Counter(y.tolist()).items()},
    }


def generate_scenario_dataset(
    *,
    n_scenarios: int = 5000,
    data_dir: Path | str = "data/raw",
    out_dir: Path | str = "data/synthetic",
    manifest: Path | str | None = None,
    feature_cache: Path | str | None = None,
    seed: int = 42,
) -> dict:
    """Generate the plan's scenario-driven synthetic training set."""
    counts = scenario_counts_for_total(n_scenarios)
    # Reserve the explicit hard-case generators inside labels 2/4/6.  They are
    # deducted from the generic class pools so the total remains exact.
    bus7_load = max(1, min(counts[4] // 6, counts[4]))
    line_2423 = max(1, min(counts[2] // 6, counts[2]))
    post_cyber = counts[6]
    generic_load = max(0, counts[4] - bus7_load)
    generic_line = max(0, counts[2] - line_2423)

    result = generate_all(
        n_faults=counts[1],
        n_line_outages=generic_line,
        n_gen_changes=counts[3],
        n_load_changes=generic_load,
        n_dropouts=counts[5],
        n_bus7_load_changes=bus7_load,
        n_line_2423_outages=line_2423,
        n_post_cyber_physical=post_cyber,
        n_bad_data=counts[7],
        n_unknown=counts[8],
        data_dir=data_dir,
        out_dir=out_dir,
        seed=seed,
    )
    result["requested_scenarios"] = int(n_scenarios)
    result["requested_label_mix"] = {str(k): int(v) for k, v in counts.items()}
    result["label_names"] = {str(k): v for k, v in LABEL_NAMES.items()}
    result["targeted_coverage"] = {
        "bus7_load_changes": int(bus7_load),
        "line_24_23_outages": int(line_2423),
        "post_cyber_physical": int(post_cyber),
        "pmu_dropout_buses": list(PMU_BUSES),
        "bad_data_buses": list(PMU_BUSES),
        "fault_impedances": [0.0, 0.01, 0.05, 0.1, 0.2],
        "unknown_multi_event": int(counts[8]),
    }

    manifest_path = Path(manifest) if manifest is not None else Path(out_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["manifest"] = str(manifest_path)

    if feature_cache is not None:
        result["feature_cache_summary"] = write_feature_cache(
            out_dir,
            feature_cache,
            data_dir=data_dir,
        )
        manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return result


def load_synthetic(
    syn_dir: Path | str,
    run_id: int,
) -> pd.DataFrame:
    """Load one synthetic event (8 CSVs) as a merged DataFrame."""
    syn_dir = Path(syn_dir)
    frames = []
    ts_ref = None
    for bus in PMU_BUSES:
        fname = syn_dir / f"syn{run_id:04d}_Bus{bus}_Competition_Data_nanmask.csv"
        if not fname.exists():
            continue
        df = pd.read_csv(fname)
        if ts_ref is None:
            ts_ref = df[["TIMESTAMP"]]
        df = df.rename(columns={
            "DATA_PRESENT": f"BUS{bus}_DATA_PRESENT",
            "Event": f"BUS{bus}_Event",
        })
        frames.append(df.drop(columns=["TIMESTAMP"]))

    if not frames:
        raise FileNotFoundError(f"No synthetic files for run {run_id} in {syn_dir}")

    merged = pd.concat([ts_ref] + frames, axis=1)
    # Global event
    ecols = [f"BUS{b}_Event" for b in PMU_BUSES if f"BUS{b}_Event" in merged]
    merged["Event"] = merged[ecols].fillna(0).astype(int).max(axis=1)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate reproducible physics-informed synthetic PMU scenarios."
    )
    parser.add_argument("--n-scenarios", type=int, default=5000)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    result = generate_scenario_dataset(
        n_scenarios=args.n_scenarios,
        data_dir=args.data,
        out_dir=args.out,
        manifest=args.manifest,
        feature_cache=args.feature_cache,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
