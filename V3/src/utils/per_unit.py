# src/utils/per_unit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np


# --- Nominal bus base map (from network model) ---
# Bus-side nominal kV line-to-line values as defined by the IEEE-39 network model.
BUS_BASE_KV_LL: dict[int, float] = {
    1: 345.0, 2: 345.0, 3: 345.0, 4: 345.0, 5: 345.0, 6: 345.0, 7: 345.0,
    8: 345.0, 9: 345.0, 10: 345.0, 11: 345.0, 12: 345.0, 13: 345.0, 14: 345.0,
    15: 345.0, 16: 345.0, 17: 345.0, 18: 345.0, 19: 345.0, 20: 110.0, 21: 345.0,
    22: 345.0, 23: 345.0, 24: 345.0, 25: 345.0, 26: 345.0, 27: 345.0, 28: 345.0,
    29: 345.0, 30: 13.8, 31: 13.8, 32: 13.8, 33: 13.8, 34: 13.8, 35: 13.8,
    36: 13.8, 37: 13.8, 38: 13.8, 39: 345.0,
}


# --- Measurement-side override map (Hypothesis B) ---
# ANDES reports all bus voltages at the 345 kV backbone scale, regardless of
# terminal nominal kV. Bus 20 (110 kV nominal) and buses 30-38 (13.8 kV nominal,
# generator terminals) exhibit ~3.2x and ~25x p.u. values under the nominal base,
# which is the symptom being tested here.
BUS_MEASUREMENT_BASE_KV_LL_OVERRIDE_345: dict[int, float] = {
    **BUS_BASE_KV_LL,
    20: 345.0,   # nominal 110 kV — ANDES reports at 345 kV backbone scale
    30: 345.0,   # nominal 13.8 kV — generator terminal, reported at backbone scale
    31: 345.0,
    32: 345.0,
    33: 345.0,
    34: 345.0,
    35: 345.0,
    36: 345.0,
    37: 345.0,
    38: 345.0,
}

MeasurementBaseMode = Literal["nominal", "override_345"]

MEASUREMENT_BASE_MAPS: dict[str, dict[int, float]] = {
    "nominal": BUS_BASE_KV_LL,
    "override_345": BUS_MEASUREMENT_BASE_KV_LL_OVERRIDE_345,
}


@dataclass(frozen=True)
class PerUnitSystem:
    s_base_mva: float = 100.0


# --- Nominal base helpers (network model, no mode argument) ---

def get_bus_base_kv_ll(bus_id: int) -> float:
    if bus_id not in BUS_BASE_KV_LL:
        raise KeyError(f"Missing BUS_BASE_KV_LL for bus {bus_id}")
    return BUS_BASE_KV_LL[bus_id]


def voltage_base_ll_volts(bus_id: int) -> float:
    return get_bus_base_kv_ll(bus_id) * 1e3


def voltage_base_ln_volts(bus_id: int) -> float:
    """Phase-to-neutral nominal base voltage in volts (from network model)."""
    return voltage_base_ll_volts(bus_id) / np.sqrt(3.0)


def current_base_amperes(bus_id: int, s_base_mva: float = 100.0) -> float:
    v_base_ll = voltage_base_ll_volts(bus_id)
    return (s_base_mva * 1e6) / (np.sqrt(3.0) * v_base_ll)


def impedance_base_ohms(bus_id: int, s_base_mva: float = 100.0) -> float:
    v_base_ll = voltage_base_ll_volts(bus_id)
    return (v_base_ll ** 2) / (s_base_mva * 1e6)


# --- Measurement base helper ---

def voltage_measurement_base_ln_volts(
    bus_id: int,
    mode: MeasurementBaseMode = "nominal",
) -> float:
    """Phase-to-neutral measurement base in volts for the given mode.

    "nominal"      — uses the network-model nominal base (BUS_BASE_KV_LL).
    "override_345" — uses 345 kV for buses 20 and 30-38 instead of their
                     nominal values, testing whether ANDES reports those buses
                     at the 345 kV backbone scale.
    """
    base_map = MEASUREMENT_BASE_MAPS.get(mode)
    if base_map is None:
        raise ValueError(
            f"Unknown measurement base mode: {mode!r}. Valid: {sorted(MEASUREMENT_BASE_MAPS)}"
        )
    if bus_id not in base_map:
        raise KeyError(f"Missing measurement base for bus {bus_id} in mode {mode!r}")
    return base_map[bus_id] * 1e3 / np.sqrt(3.0)


# --- Conversion functions (measurement-base aware) ---
# Default mode="nominal" keeps all existing call sites working unchanged.

def complex_voltage_to_pu(
    v_complex: np.ndarray,
    bus_id: int,
    mode: MeasurementBaseMode = "nominal",
) -> np.ndarray:
    return v_complex / voltage_measurement_base_ln_volts(bus_id, mode)


def complex_voltage_from_pu(
    v_pu: np.ndarray,
    bus_id: int,
    mode: MeasurementBaseMode = "nominal",
) -> np.ndarray:
    return v_pu * voltage_measurement_base_ln_volts(bus_id, mode)


def complex_current_to_pu(
    i_complex: np.ndarray,
    bus_id: int,
    s_base_mva: float = 100.0,
) -> np.ndarray:
    return i_complex / current_base_amperes(bus_id, s_base_mva=s_base_mva)


def complex_current_from_pu(
    i_pu: np.ndarray,
    bus_id: int,
    s_base_mva: float = 100.0,
) -> np.ndarray:
    return i_pu * current_base_amperes(bus_id, s_base_mva=s_base_mva)


def magnitude_to_pu(
    magnitude: np.ndarray,
    bus_id: int,
    mode: MeasurementBaseMode = "nominal",
) -> np.ndarray:
    return magnitude / voltage_measurement_base_ln_volts(bus_id, mode)


def magnitude_from_pu(
    magnitude_pu: np.ndarray,
    bus_id: int,
    mode: MeasurementBaseMode = "nominal",
) -> np.ndarray:
    return magnitude_pu * voltage_measurement_base_ln_volts(bus_id, mode)


def pre_event_mean_pu(
    magnitude: np.ndarray,
    event: np.ndarray,
    bus_id: int,
    mode: MeasurementBaseMode = "nominal",
) -> float:
    mask = np.asarray(event) == 0
    if not np.any(mask):
        raise ValueError("No normal-event samples found for pre_event_mean_pu")
    mag_pu = magnitude_to_pu(np.asarray(magnitude)[mask], bus_id, mode)
    return float(np.mean(mag_pu))


def summarize_pre_event_voltage_pu(
    bus_to_magnitude: dict[int, np.ndarray],
    bus_to_event: dict[int, np.ndarray],
    bus_ids: Iterable[int],
    mode: MeasurementBaseMode = "nominal",
) -> dict[int, float]:
    summary: dict[int, float] = {}
    for bus_id in bus_ids:
        summary[bus_id] = pre_event_mean_pu(
            magnitude=bus_to_magnitude[bus_id],
            event=bus_to_event[bus_id],
            bus_id=bus_id,
            mode=mode,
        )
    return summary
