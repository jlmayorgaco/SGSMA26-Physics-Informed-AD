"""Physical truth generation for M9 scenarios.

The module can seed trajectories from a normal ANDES IEEE-39 run when requested.
Physical event signatures are then applied deterministically so the full scenario
matrix remains fast and reproducible in tests and CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math

import numpy as np
import pandas as pd

from src.domain.topology import bus_sort_key, canonical_bus_name
from src.infrastructure.io.metadata_loader import load_pmu_metadata
from src.infrastructure.io.raw_network_loader import load_network_model_from_raw
from src.metadata.raw_parser import parse_raw_file
from src.simulation.m9.constants import DEFAULT_RATED_FREQUENCY_HZ, PHYSICAL_EVENT_LABELS


@dataclass(slots=True)
class PhysicalTruth:
    timestamps: np.ndarray
    bus_order: list[str]
    voltage_complex_pu: np.ndarray
    frequency_hz: np.ndarray
    rocof_hz_per_s: np.ndarray
    event_by_frame: np.ndarray
    physical_type_by_frame: list[str]
    origin_bus_by_frame: list[str]
    origin_line_by_frame: list[str]
    window_type_by_frame: list[str]
    metadata: dict[str, Any]


def _bus_num(bus: str | None) -> int | None:
    if not bus:
        return None
    digits = "".join(ch for ch in canonical_bus_name(bus) if ch.isdigit())
    return int(digits) if digits else None


def _distance_weight(bus: str, target_bus: str | None, target_line: str | None = None) -> float:
    n = _bus_num(bus)
    targets: list[int] = []
    if target_bus:
        tb = _bus_num(target_bus)
        if tb is not None:
            targets.append(tb)
    if target_line:
        for piece in str(target_line).replace("_", "-").split("-"):
            tb = _bus_num(piece)
            if tb is not None:
                targets.append(tb)
    if n is None or not targets:
        return 0.45
    d = min(abs(n - t) for t in targets)
    return float(0.18 + 0.82 * math.exp(-d / 8.0))


def _smooth_step(x: np.ndarray) -> np.ndarray:
    return np.clip(3 * x**2 - 2 * x**3, 0.0, 1.0)


def _event_masks(t: np.ndarray, event: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start = float(event.get("start_time_s", 0.0))
    end = float(event.get("end_time_s", start + float(event.get("duration_s", 0.0))))
    active = (t >= start) & (t <= end)
    after = t > end
    x = np.maximum(t - start, 0.0)
    dur = max(end - start, 1e-6)
    phase = np.clip(x / dur, 0.0, 1.0)
    _ = after
    return active, x, phase


def _load_base_profile(raw_path: Path, pmu_location_path: Path) -> tuple[list[str], dict[str, float], dict[str, float], dict[str, float], float]:
    pmu_meta = load_pmu_metadata(pmu_location_path)
    raw = parse_raw_file(raw_path) if raw_path.exists() else {"buses": [], "base_mva": 100.0}
    bus_records = raw.get("buses") or pmu_meta.get("buses", [])
    bus_order = sorted({canonical_bus_name(b["bus_label_canonical"]) for b in bus_records}, key=bus_sort_key)
    v0: dict[str, float] = {}
    a0: dict[str, float] = {}
    kv: dict[str, float] = {}
    for rec in pmu_meta.get("buses", []):
        bus = canonical_bus_name(rec["bus_label_canonical"])
        v0[bus] = float(rec.get("v_pu", 1.0))
        a0[bus] = float(rec.get("theta_deg", 0.0))
        kv[bus] = float(rec.get("kv_ll", 345.0))
    for rec in raw.get("buses", []):
        bus = canonical_bus_name(rec["bus_label_canonical"])
        v0.setdefault(bus, float(rec.get("v_pu", 1.0)))
        a0.setdefault(bus, float(rec.get("theta_deg", 0.0)))
        kv.setdefault(bus, float(rec.get("kv_ll", 345.0)))
    if not bus_order:
        bus_order = [f"BUS{i}" for i in range(1, 40)]
    for bus in bus_order:
        v0.setdefault(bus, 1.0)
        a0.setdefault(bus, 0.0)
        kv.setdefault(bus, 345.0)
    return bus_order, v0, a0, kv, float(raw.get("base_mva", pmu_meta.get("base_mva", 100.0)))


def _try_andes_normal(duration_s: float, fps: float) -> tuple[np.ndarray, list[str], np.ndarray] | None:
    try:
        from src.simulation.andes_ieee39_runner import run_andes_ieee39_truth

        truth = run_andes_ieee39_truth(tf=float(duration_s), tstep=1.0 / float(fps), stride=1)
        t = np.asarray(truth.timestamps, dtype=float)
        v = np.asarray(truth.voltage_complex_pu, dtype=complex)
        buses = [canonical_bus_name(b) for b in truth.bus_ids]
        if t.size >= 4 and v.shape[0] == t.size and v.shape[1] == len(buses):
            return t, buses, v
    except Exception:
        return None
    return None


def _align_andes(t_target: np.ndarray, bus_order: list[str], andes_payload: tuple[np.ndarray, list[str], np.ndarray]) -> np.ndarray:
    t_src, buses_src, v_src = andes_payload
    src_pos = {canonical_bus_name(b): i for i, b in enumerate(buses_src)}
    src_num: dict[str, int] = {}
    for bus, pos in src_pos.items():
        digits = "".join(ch for ch in bus if ch.isdigit())
        if digits and digits not in src_num:
            src_num[digits] = pos
    out = np.zeros((len(t_target), len(bus_order)), dtype=complex)
    for j, bus in enumerate(bus_order):
        pos = src_pos.get(bus)
        if pos is None:
            digits = "".join(ch for ch in bus if ch.isdigit())
            pos = src_num.get(digits)
        if pos is None:
            out[:, j] = 1.0 + 0.0j
            continue
        real = np.interp(t_target, t_src, np.real(v_src[:, pos]))
        imag = np.interp(t_target, t_src, np.imag(v_src[:, pos]))
        out[:, j] = real + 1j * imag
    return out


def _base_trajectory(t: np.ndarray, bus_order: list[str], v0: dict[str, float], a0: dict[str, float], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((len(t), len(bus_order)), dtype=complex)
    for j, bus in enumerate(bus_order):
        n = _bus_num(bus) or (j + 1)
        mag = v0[bus] + 0.0015 * np.sin(2 * np.pi * (0.13 + (n % 5) * 0.017) * t + n * 0.21)
        mag += 0.0006 * np.sin(2 * np.pi * 0.031 * t + n * 0.11)
        mag += rng.normal(0.0, 0.00008, size=len(t))
        ang = a0[bus] + 0.035 * np.sin(2 * np.pi * (0.10 + (n % 7) * 0.011) * t + n * 0.17)
        ang += 0.015 * np.sin(2 * np.pi * 0.023 * t + n * 0.07)
        out[:, j] = mag * np.exp(1j * np.deg2rad(ang))
    return out


def _apply_physical_events(
    t: np.ndarray,
    bus_order: list[str],
    v: np.ndarray,
    physical_events: list[dict[str, Any]],
    rated_frequency_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], list[str], list[str]]:
    v_out = np.array(v, dtype=complex, copy=True)
    freq = np.full(len(t), float(rated_frequency_hz), dtype=float)
    event_label = np.zeros(len(t), dtype=int)
    physical_type = ["normal"] * len(t)
    origin_bus = [""] * len(t)
    origin_line = [""] * len(t)
    window = ["quiet"] * len(t)

    for event in physical_events:
        etype = str(event.get("event_type", "")).lower()
        subtype = str(event.get("subtype", ""))
        label = int(event.get("event_label", PHYSICAL_EVENT_LABELS.get(etype, 8)))
        target_bus = event.get("target_bus")
        target_line = event.get("target_line")
        severity = float(event.get("severity", 1.0))
        active, x, phase = _event_masks(t, event)
        start = float(event.get("start_time_s", 0.0))
        end = float(event.get("end_time_s", start + float(event.get("duration_s", 0.0))))
        tail = t > end
        affected = active | (tail & (t <= end + 3.0))
        idx = np.where(affected)[0]
        if idx.size:
            event_label[idx] = label
            for i in idx:
                physical_type[i] = etype or subtype or "physical"
                origin_bus[i] = str(target_bus or "")
                origin_line[i] = str(target_line or "")
                window[i] = "event"

        if etype == "fault":
            fault_pulse = active.astype(float)
            recovery = np.where(t > end, np.exp(-(t - end) / 1.6), 0.0)
            osc = np.sin(2 * np.pi * 1.35 * np.maximum(t - start, 0.0)) * np.exp(-np.maximum(t - start, 0.0) / 1.8)
            for j, bus in enumerate(bus_order):
                w = _distance_weight(bus, target_bus, target_line)
                mag_factor = 1.0 - (0.22 * severity * w * fault_pulse) - (0.045 * severity * w * recovery)
                ang_shift = (5.0 * severity * w * fault_pulse + 2.2 * severity * w * osc)
                v_out[:, j] *= np.maximum(mag_factor, 0.55) * np.exp(1j * np.deg2rad(ang_shift))
            freq += -0.035 * severity * active.astype(float) + 0.010 * severity * osc
        elif etype == "line_outage":
            step = np.where(t >= start, _smooth_step(np.clip((t - start) / 0.4, 0.0, 1.0)), 0.0)
            decay = np.where(t >= start, np.exp(-np.maximum(t - start, 0.0) / 4.0), 0.0)
            osc = np.sin(2 * np.pi * 0.75 * np.maximum(t - start, 0.0)) * decay
            for j, bus in enumerate(bus_order):
                w = _distance_weight(bus, target_bus, target_line)
                mag_factor = 1.0 - 0.018 * severity * w * step + 0.006 * severity * w * osc
                ang_shift = 1.6 * severity * w * step + 0.8 * severity * w * osc
                v_out[:, j] *= mag_factor * np.exp(1j * np.deg2rad(ang_shift))
            freq += -0.012 * severity * decay
        elif etype in {"generation_change", "generation_outage"}:
            ramp = np.where(t >= start, _smooth_step(phase), 0.0)
            decay = np.where(t >= start, np.exp(-np.maximum(t - start, 0.0) / 5.0), 0.0)
            osc = np.sin(2 * np.pi * 0.55 * np.maximum(t - start, 0.0)) * decay
            for j, bus in enumerate(bus_order):
                w = _distance_weight(bus, target_bus, target_line)
                mag_factor = 1.0 - 0.012 * severity * w * ramp + 0.004 * severity * w * osc
                ang_shift = -2.0 * severity * w * ramp + 0.9 * severity * w * osc
                v_out[:, j] *= mag_factor * np.exp(1j * np.deg2rad(ang_shift))
            freq += -0.085 * severity * ramp + 0.030 * severity * osc
        elif etype in {"load_change", "load_drop"}:
            ramp = np.where(t >= start, _smooth_step(phase), 0.0)
            slow = np.where(t >= start, 1.0 - np.exp(-np.maximum(t - start, 0.0) / 3.0), 0.0)
            for j, bus in enumerate(bus_order):
                w = _distance_weight(bus, target_bus, target_line)
                mag_factor = 1.0 + 0.010 * severity * w * ramp - 0.006 * severity * slow
                ang_shift = 0.9 * severity * w * ramp
                v_out[:, j] *= mag_factor * np.exp(1j * np.deg2rad(ang_shift))
            freq += 0.030 * severity * ramp * np.exp(-np.maximum(t - start, 0.0) / 6.0)
        else:
            bump = active.astype(float) * severity
            for j, bus in enumerate(bus_order):
                w = _distance_weight(bus, target_bus, target_line)
                v_out[:, j] *= (1.0 + 0.004 * w * bump) * np.exp(1j * np.deg2rad(0.6 * w * bump))
            freq += 0.008 * bump

    if len(t) > 1:
        rocof = np.gradient(freq, t)
    else:
        rocof = np.zeros_like(freq)
    return v_out, freq, rocof, event_label, physical_type, origin_bus, origin_line, window


def generate_physical_truth(
    raw_path: str | Path,
    pmu_location_path: str | Path,
    template: dict[str, Any],
    seed: int = 12345,
    fps: float = 30.0,
    duration_s: float | None = None,
    use_andes: bool = False,
) -> PhysicalTruth:
    raw_p = Path(raw_path)
    pmu_p = Path(pmu_location_path)
    duration = float(duration_s if duration_s is not None else template.get("duration_s", 12.0))
    frame_count = int(round(duration * float(fps))) + 1
    t = np.round(np.arange(frame_count, dtype=float) / float(fps), 3)

    bus_order, v0, a0, kv, base_mva = _load_base_profile(raw_p, pmu_p)
    base_v = _base_trajectory(t, bus_order, v0, a0, seed=seed)
    truth_source = "metadata_synthetic_dynamic"
    andes_payload = _try_andes_normal(duration, fps) if use_andes else None
    if andes_payload is not None:
        base_v = _align_andes(t, bus_order, andes_payload)
        truth_source = "andes_normal_seed_plus_m9_event_overlays"

    v_event, freq, rocof, labels, ptypes, obus, oline, windows = _apply_physical_events(
        t,
        bus_order,
        base_v,
        list(template.get("physical_events", [])),
        DEFAULT_RATED_FREQUENCY_HZ,
    )
    metadata = {
        "truth_source": truth_source,
        "base_mva": base_mva,
        "rated_frequency_hz": DEFAULT_RATED_FREQUENCY_HZ,
        "bus_kv_ll": kv,
        "andes_requested": bool(use_andes),
        "andes_used": bool(andes_payload is not None),
        "physical_event_count": len(template.get("physical_events", [])),
    }
    return PhysicalTruth(
        timestamps=t,
        bus_order=bus_order,
        voltage_complex_pu=v_event,
        frequency_hz=freq,
        rocof_hz_per_s=rocof,
        event_by_frame=labels,
        physical_type_by_frame=ptypes,
        origin_bus_by_frame=obus,
        origin_line_by_frame=oline,
        window_type_by_frame=windows,
        metadata=metadata,
    )


def compute_injection_currents_pu(raw_path: str | Path, bus_order: list[str], voltage_complex_pu: np.ndarray) -> np.ndarray:
    try:
        network = load_network_model_from_raw(raw_path)
        pos = {canonical_bus_name(b): i for i, b in enumerate(network.bus_order)}
        idx = [pos.get(canonical_bus_name(b), None) for b in bus_order]
        y = np.asarray(network.ybus, dtype=complex)
        v_aligned = np.zeros((voltage_complex_pu.shape[0], len(network.bus_order)), dtype=complex)
        for j, src in enumerate(idx):
            if src is not None:
                v_aligned[:, src] = voltage_complex_pu[:, j]
        i_aligned = (y @ v_aligned.T).T
        out = np.zeros_like(voltage_complex_pu, dtype=complex)
        for j, src in enumerate(idx):
            out[:, j] = i_aligned[:, src] if src is not None else 0.0
        return out
    except Exception:
        return 0.18 * voltage_complex_pu * np.exp(-1j * np.deg2rad(20.0))


def truth_to_dataframe(truth: PhysicalTruth, pmu_buses: set[str], cyber_type_by_frame: list[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cyber_type_by_frame = cyber_type_by_frame or [""] * len(truth.timestamps)
    for i, ts in enumerate(truth.timestamps):
        for j, bus in enumerate(truth.bus_order):
            z = complex(truth.voltage_complex_pu[i, j])
            rows.append(
                {
                    "TIMESTAMP": float(ts),
                    "BUS": bus,
                    "IS_PMU_BUS": bool(bus in pmu_buses),
                    "IS_NON_PMU_BUS": bool(bus not in pmu_buses),
                    "V_TRUE_REAL_PU": float(np.real(z)),
                    "V_TRUE_IMAG_PU": float(np.imag(z)),
                    "V_TRUE_MAG_PU": float(np.abs(z)),
                    "V_TRUE_ANG_DEG": float(np.rad2deg(np.angle(z))),
                    "EVENT": int(truth.event_by_frame[i]),
                    "EVENT_ORIGIN_BUS": truth.origin_bus_by_frame[i],
                    "EVENT_ORIGIN_LINE": truth.origin_line_by_frame[i],
                    "PHYSICAL_EVENT_TYPE": truth.physical_type_by_frame[i],
                    "CYBER_EVENT_TYPE": cyber_type_by_frame[i],
                    "WINDOW_TYPE": truth.window_type_by_frame[i],
                }
            )
    return pd.DataFrame(rows)
