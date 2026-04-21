"""Ybus builder from parsed RAW model."""

from __future__ import annotations

import numpy as np

from src.metadata.bus_mapping import numeric_bus_sort_key


def _complex_polar_tap(tap_ratio: float, phase_shift_deg: float) -> complex:
    tap = 1.0 if abs(tap_ratio) < 1e-12 else float(tap_ratio)
    phase = np.deg2rad(float(phase_shift_deg))
    return tap * np.exp(1j * phase)


def build_ybus(raw_model: dict) -> tuple[np.ndarray, list[str], dict]:
    """Build complex Ybus matrix and explicit bus order from RAW model."""
    bus_order = sorted([b["bus_label_canonical"] for b in raw_model.get("buses", [])], key=numeric_bus_sort_key)
    n = len(bus_order)
    if n == 0:
        raise ValueError("Cannot build Ybus: RAW model has zero buses.")

    pos = {b: i for i, b in enumerate(bus_order)}
    ybus = np.zeros((n, n), dtype=complex)

    active_branch_count = 0
    for br in raw_model.get("branches", []):
        if int(br.get("status", 1)) == 0:
            continue
        i_bus = br["from_bus"]
        j_bus = br["to_bus"]
        if i_bus not in pos or j_bus not in pos:
            continue
        r = float(br.get("r_pu", 0.0))
        x = float(br.get("x_pu", 0.0))
        b = float(br.get("b_pu", 0.0))
        if abs(r) + abs(x) < 1e-12:
            continue
        y = 1.0 / complex(r, x)
        ysh = 0.5j * b
        tap = _complex_polar_tap(float(br.get("tap_ratio", 0.0)), float(br.get("phase_shift_deg", 0.0)))

        i = pos[i_bus]
        j = pos[j_bus]
        ybus[i, i] += (y + ysh) / (tap * np.conj(tap))
        ybus[j, j] += y + ysh
        ybus[i, j] -= y / np.conj(tap)
        ybus[j, i] -= y / tap
        active_branch_count += 1

    for sh in raw_model.get("fixed_shunts", []):
        bus = sh.get("bus_label_canonical")
        if bus not in pos:
            continue
        g = float(sh.get("g_pu", 0.0))
        b = float(sh.get("b_pu", 0.0))
        ybus[pos[bus], pos[bus]] += complex(g, b)

    meta = {
        "builder": "project_raw_parser",
        "active_branch_count": active_branch_count,
        "fixed_shunt_count": len(raw_model.get("fixed_shunts", [])),
        "shape": [n, n],
        "bus_ordering": "canonical_numeric",
    }
    return ybus, bus_order, meta
