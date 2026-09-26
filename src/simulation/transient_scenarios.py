"""Calibrated IEEE-39 transient scenarios for hidden-bus reconstruction.

The module deliberately keeps two representations of every scenario:

* a full-state, noise-free ANDES trajectory for all 39 buses (ground truth), and
* a calibrated observation layer exposing only the eight competition PMUs.

Physical events are created inside ANDES.  Missing and bad data are applied to
the observation layer so that the physical truth is not silently corrupted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

# The Anaconda gmpy2 binary available in this Windows workspace is not ABI
# compatible with Python 3.13 and can crash after ANDES has completed.  SymPy's
# pure-Python ground types are deterministic for this workload and avoid the
# spurious non-zero process exit without changing the network equations.
os.environ.setdefault("SYMPY_GROUND_TYPES", "python")

from src.simulation.network_extraction import extract_all_bus_signals
from src.simulation.raw_static_case import (
    DEFAULT_RAW_CASE,
    apply_raw_static_case,
    identify_pmu_branch_mappings,
    parse_raw_static_case,
    replace_pmu_currents_with_branch_channels,
)


DEFAULT_PMU_BUSES = (2, 5, 6, 10, 19, 22, 29, 39)
GENERATOR_INERTIA_GROUPS = {
    "30_32": (30, 31, 32),
    "33_36": (33, 34, 35, 36),
    "37_39": (37, 38, 39),
}
RAW_SUFFIXES = (
    "VA_ANG",
    "VA_MAG",
    "VB_ANG",
    "VB_MAG",
    "VC_ANG",
    "VC_MAG",
    "IA_ANG",
    "IA_MAG",
    "IB_ANG",
    "IB_MAG",
    "IC_ANG",
    "IC_MAG",
    "Freq",
    "ROCOF",
)


def _mad_std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    med = float(np.median(arr))
    return float(1.4826 * np.median(np.abs(arr - med)))


def _wrap_deg(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return (arr + 180.0) % 360.0 - 180.0


def _circular_center_deg(values: np.ndarray) -> float:
    arr = np.deg2rad(np.asarray(values, dtype=float))
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0
    return float(np.rad2deg(np.angle(np.mean(np.exp(1j * arr)))))


def _bus_from_path(path: Path) -> int:
    match = re.search(r"Bus(\d+)", path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer bus from {path.name!r}")
    return int(match.group(1))


@dataclass(frozen=True)
class SignalCalibration:
    median: float
    robust_std: float
    noise_std: float
    ar1: float
    sample_count: int


@dataclass
class MeasurementCalibration:
    """Robust event-0 statistics in the units of the competition CSVs."""

    profiles: dict[str, dict[str, SignalCalibration]]
    nominal_frequency_hz: float = 60.0
    sample_rate_hz: float = 30.0
    source_dir: str = ""
    current_mappings: dict[str, dict[str, Any]] = field(default_factory=dict)
    angle_reference_mode: str = "per_channel"
    common_angle_offset_deg: float = 0.0
    voltage_angle_bias_deg: dict[str, float] = field(default_factory=dict)
    current_angle_bias_deg: dict[str, float] = field(default_factory=dict)
    common_angle_noise_fraction: float = 0.0
    noise_correlation_matrices: dict[str, list[list[float]]] = field(default_factory=dict)
    noise_ar1: dict[str, float] = field(default_factory=dict)
    frequency_filter: dict[str, Any] = field(default_factory=dict)

    def profile(self, bus: int, suffix: str) -> SignalCalibration:
        bus_key = str(int(bus))
        if bus_key in self.profiles and suffix in self.profiles[bus_key]:
            return self.profiles[bus_key][suffix]
        candidates = [p[suffix] for p in self.profiles.values() if suffix in p]
        if not candidates:
            return SignalCalibration(0.0, 0.0, 0.0, 0.0, 0)
        return SignalCalibration(
            median=float(np.median([p.median for p in candidates])),
            robust_std=float(np.median([p.robust_std for p in candidates])),
            noise_std=float(np.median([p.noise_std for p in candidates])),
            ar1=float(np.median([p.ar1 for p in candidates])),
            sample_count=int(sum(p.sample_count for p in candidates)),
        )

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nominal_frequency_hz": self.nominal_frequency_hz,
            "sample_rate_hz": self.sample_rate_hz,
            "source_dir": self.source_dir,
            "current_mappings": self.current_mappings,
            "angle_reference_mode": self.angle_reference_mode,
            "common_angle_offset_deg": self.common_angle_offset_deg,
            "voltage_angle_bias_deg": self.voltage_angle_bias_deg,
            "current_angle_bias_deg": self.current_angle_bias_deg,
            "common_angle_noise_fraction": self.common_angle_noise_fraction,
            "noise_correlation_matrices": self.noise_correlation_matrices,
            "noise_ar1": self.noise_ar1,
            "frequency_filter": self.frequency_filter,
            "profiles": {
                bus: {suffix: asdict(profile) for suffix, profile in signals.items()}
                for bus, signals in self.profiles.items()
            },
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    @classmethod
    def from_json(cls, path: str | Path) -> "MeasurementCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        profiles = {
            bus: {suffix: SignalCalibration(**values) for suffix, values in signals.items()}
            for bus, signals in payload["profiles"].items()
        }
        return cls(
            profiles=profiles,
            nominal_frequency_hz=float(payload.get("nominal_frequency_hz", 60.0)),
            sample_rate_hz=float(payload.get("sample_rate_hz", 30.0)),
            source_dir=str(payload.get("source_dir", "")),
            current_mappings=payload.get("current_mappings", {}),
            angle_reference_mode=str(payload.get("angle_reference_mode", "per_channel")),
            common_angle_offset_deg=float(payload.get("common_angle_offset_deg", 0.0)),
            voltage_angle_bias_deg={
                str(key): float(value)
                for key, value in payload.get("voltage_angle_bias_deg", {}).items()
            },
            current_angle_bias_deg={
                str(key): float(value)
                for key, value in payload.get("current_angle_bias_deg", {}).items()
            },
            common_angle_noise_fraction=float(payload.get("common_angle_noise_fraction", 0.0)),
            noise_correlation_matrices={
                str(key): [[float(cell) for cell in row] for row in value]
                for key, value in payload.get("noise_correlation_matrices", {}).items()
            },
            noise_ar1={str(key): float(value) for key, value in payload.get("noise_ar1", {}).items()},
            frequency_filter=dict(payload.get("frequency_filter", {})),
        )


def calibrate_from_event0(
    raw_dir: str | Path,
    pmu_buses: tuple[int, ...] = DEFAULT_PMU_BUSES,
    max_samples_per_bus: int = 60_000,
    raw_case_path: str | Path | None = DEFAULT_RAW_CASE,
) -> MeasurementCalibration:
    """Fit robust centers and colored-noise scales from real event-0 samples.

    A centered rolling median removes slow operating drift.  The remaining
    robust residual scale is used as the stationary PMU noise target.  This is
    intentionally conservative: it preserves more real variability than a
    first-difference-only white-noise estimate.
    """

    root = Path(raw_dir)
    paths = sorted(root.glob("Bus*.csv"), key=_bus_from_path)
    selected = {int(bus) for bus in pmu_buses}
    paths = [p for p in paths if _bus_from_path(p) in selected]
    if not paths:
        raise FileNotFoundError(f"No Bus*.csv files found in {root}")

    profiles: dict[str, dict[str, SignalCalibration]] = {}
    sample_intervals: list[float] = []
    for path in paths:
        bus = _bus_from_path(path)
        frame = pd.read_csv(path)
        mask = np.ones(len(frame), dtype=bool)
        if "Event" in frame:
            mask &= frame["Event"].fillna(0).to_numpy(dtype=float) == 0
        if "DATA_PRESENT" in frame:
            mask &= frame["DATA_PRESENT"].fillna(0).to_numpy(dtype=float) == 1
        frame = frame.loc[mask]
        if "TIMESTAMP" in frame and len(frame) > 2:
            delta = np.diff(frame["TIMESTAMP"].to_numpy(dtype=float))
            delta = delta[np.isfinite(delta) & (delta > 0)]
            if len(delta):
                sample_intervals.append(float(np.median(delta)))
        if len(frame) > max_samples_per_bus:
            # Keep consecutive samples: PMU innovation/noise statistics depend
            # on true temporal adjacency and are corrupted by uniform thinning.
            frame = frame.iloc[:max_samples_per_bus]

        bus_profiles: dict[str, SignalCalibration] = {}
        for suffix in RAW_SUFFIXES:
            column = f"BUS{bus}_{suffix}"
            if column not in frame:
                continue
            values = frame[column].to_numpy(dtype=float)
            finite = np.isfinite(values)
            values = values[finite]
            if len(values) < 10:
                continue

            if suffix.endswith("_ANG"):
                unwrapped = np.rad2deg(np.unwrap(np.deg2rad(values)))
                center = _circular_center_deg(values)
                series = pd.Series(unwrapped)
            else:
                center = float(np.median(values))
                series = pd.Series(values)
            window = min(301, max(11, (len(series) // 20) | 1))
            trend = series.rolling(window=window, center=True, min_periods=1).median()
            residual = (series - trend).to_numpy(dtype=float)
            robust_std = _mad_std(values)
            residual_std = _mad_std(residual)
            diff_std = _mad_std(np.diff(series.to_numpy(dtype=float))) / math.sqrt(2.0)
            noise_std = max(diff_std, 0.5 * residual_std)
            if suffix.endswith("_MAG"):
                noise_std = min(noise_std, 0.05 * max(abs(center), 1.0))
            elif suffix.endswith("_ANG"):
                noise_std = min(noise_std, 0.5)
            elif suffix == "Freq":
                noise_std = min(noise_std, 0.05)
            elif suffix == "ROCOF":
                noise_std = min(noise_std, 0.2)

            if len(residual) > 2 and np.std(residual[:-1]) > 1e-12 and np.std(residual[1:]) > 1e-12:
                ar1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
            else:
                ar1 = 0.0
            ar1 = float(np.clip(np.nan_to_num(ar1), 0.0, 0.98))
            bus_profiles[suffix] = SignalCalibration(
                median=center,
                robust_std=robust_std,
                noise_std=float(max(noise_std, 1e-12)),
                ar1=ar1,
                sample_count=int(len(values)),
            )
        profiles[str(bus)] = bus_profiles

    sample_rate = 30.0
    if sample_intervals:
        sample_rate = 1.0 / max(float(np.median(sample_intervals)), 1e-9)
    current_mappings: dict[str, dict[str, Any]] = {}
    if raw_case_path is not None and selected == set(DEFAULT_PMU_BUSES):
        # Use a clean interval immediately before the first physical event.
        from src.simulation.raw_andes_validation import discover_raw_event_windows, load_pmu_frames

        mapping_frames = load_pmu_frames(root)
        first_event = discover_raw_event_windows(mapping_frames)[1].onset_s
        if first_event is not None:
            current_mappings = identify_pmu_branch_mappings(
                parse_raw_static_case(raw_case_path),
                mapping_frames,
                baseline_end_s=float(first_event) - 0.2,
            )
    return MeasurementCalibration(
        profiles=profiles,
        sample_rate_hz=sample_rate,
        source_dir=str(root.resolve()),
        current_mappings=current_mappings,
    )


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    event_type: int
    target: str | int | None = None
    cyber_target: int | None = None
    severity: float = 0.10
    reactive_severity: float | None = None
    generation_mode: str = "torque_step"
    generation_device_bus: int | None = None
    generation_ramp_duration_s: float = 1.0
    generation_residual_ratio: float = 1.0
    reactive_residual_ratio: float | None = None
    generation_recovery_delay_s: float = 0.0
    generation_recovery_time_constant_s: float = 1.0
    generation_transient_amplitude_ratio: float = 0.0
    generation_transient_peak_s: float = 2.0
    excitation_severity: float = 0.0
    event_start_s: float = 2.0
    event_duration_s: float = 0.20
    line_reclose: bool = True
    event_restore_s: float | None = None
    simulation_end_s: float = 4.0
    tstep_s: float = 1.0 / 30.0
    operating_scale: float = 1.0
    generator_dispatch_scale: float = 1.0
    voltage_setpoint_pu: float | None = None
    inertia_scale: float = 1.0
    inertia_group_30_32: float = 1.0
    inertia_group_33_36: float = 1.0
    inertia_group_37_39: float = 1.0
    damping: float = 0.0
    governor_droop_scale: float = 1.0
    governor_time_scale: float = 1.0
    governor_lag_scale: float = 1.0
    turbine_time_scale: float = 1.0
    avr_gain_scale: float = 1.0
    avr_time_scale: float = 1.0
    pss_gain_scale: float = 1.0
    pss_enabled: bool = True
    governor_structure: str = "TGOV1N"
    machine_transient_reactance_scale: float = 1.0
    machine_time_constant_scale: float = 1.0
    machine_subtransient_reactance_scale: float = 1.0
    machine_subtransient_time_constant_scale: float = 1.0
    bus39_xd1_pu: float | None = None
    bus39_xq1_pu: float | None = None
    bus39_xd2_pu: float | None = None
    bus39_xq2_pu: float | None = None
    bus39_Td10_s: float | None = None
    bus39_Tq10_s: float | None = None
    bus39_Td20_s: float | None = None
    bus39_Tq20_s: float | None = None
    source_transformer_impedance_scale: float = 1.0
    bus39_source_impedance_scale: float = 1.0
    bus39_external_equivalent_reactance_pu: float = 0.0
    bus39_equivalent_placement: str = "network_side"
    load_frequency_exponent: float = 0.0
    load_voltage_exponent: float = 0.0
    fault_resistance_pu: float | None = None
    fault_reactance_pu: float | None = None
    fault_post_trip: str | None = None
    fault_post_trip_secondary: str | None = None
    raw_case_path: str | None = str(DEFAULT_RAW_CASE)
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.event_type not in range(9):
            raise ValueError("event_type must be in 0..8")
        if self.event_start_s <= 0 or self.simulation_end_s <= self.event_start_s:
            raise ValueError("Scenario timing must contain a positive pre-event and post-event interval")
        if self.event_restore_s is not None and not (
            self.event_start_s < self.event_restore_s < self.simulation_end_s
        ):
            raise ValueError("event_restore_s must lie after event_start_s and before simulation_end_s")
        if self.tstep_s <= 0:
            raise ValueError("tstep_s must be positive")
        if self.inertia_scale <= 0:
            raise ValueError("inertia_scale must be positive")
        if min(
            self.inertia_group_30_32,
            self.inertia_group_33_36,
            self.inertia_group_37_39,
        ) <= 0:
            raise ValueError("regional inertia scales must be positive")
        if self.damping < 0:
            raise ValueError("damping must be non-negative")
        if self.governor_droop_scale <= 0:
            raise ValueError("governor_droop_scale must be positive")
        if self.governor_time_scale <= 0:
            raise ValueError("governor_time_scale must be positive")
        if min(
            self.governor_lag_scale,
            self.turbine_time_scale,
            self.avr_gain_scale,
            self.avr_time_scale,
            self.pss_gain_scale,
            self.machine_transient_reactance_scale,
            self.machine_time_constant_scale,
            self.machine_subtransient_reactance_scale,
            self.machine_subtransient_time_constant_scale,
            self.source_transformer_impedance_scale,
            self.bus39_source_impedance_scale,
        ) <= 0:
            raise ValueError("dynamic component scales must be positive")
        if self.governor_structure not in {"TGOV1N", "IEESGO", "IEEEG1"}:
            raise ValueError("governor_structure must be TGOV1N, IEESGO, or IEEEG1")
        if self.bus39_external_equivalent_reactance_pu < 0:
            raise ValueError("bus39_external_equivalent_reactance_pu must be non-negative")
        if self.bus39_equivalent_placement not in {"network_side", "source_side"}:
            raise ValueError("bus39_equivalent_placement must be network_side or source_side")
        for name in (
            "bus39_xd1_pu", "bus39_xq1_pu", "bus39_xd2_pu", "bus39_xq2_pu",
            "bus39_Td10_s", "bus39_Tq10_s", "bus39_Td20_s", "bus39_Tq20_s",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if not -5.0 <= self.load_frequency_exponent <= 5.0:
            raise ValueError("load_frequency_exponent must be in [-5, 5]")
        if not -3.0 <= self.load_voltage_exponent <= 3.0:
            raise ValueError("load_voltage_exponent must be in [-3, 3]")
        if self.generation_mode not in {
            "torque_step",
            "mechanical_power_step",
            "mechanical_power_ramp",
            "generator_trip",
            "torque_excitation",
            "bus_injection_step",
            "bus_injection_ramp",
            "negative_load_step",
            "signed_bus_injection_step",
            "signed_bus_injection_ramp",
            "signed_bus_pq_injection_step",
            "signed_bus_pq_injection_ramp",
            "signed_bus_p_recovery_exponential",
            "signed_bus_p_delayed_recovery",
            "signed_bus_p_hold_ramp_recovery",
            "signed_bus_pq_recovery_exponential",
            "signed_bus_pq_delayed_recovery",
            "signed_bus_p_recovery_pulse",
            "signed_bus_pq_recovery_pulse",
        }:
            raise ValueError("unsupported generation_mode")
        if self.generation_ramp_duration_s <= 0:
            raise ValueError("generation_ramp_duration_s must be positive")
        if not 0.0 <= self.generation_residual_ratio <= 1.0:
            raise ValueError("generation_residual_ratio must be in [0, 1]")
        if self.reactive_residual_ratio is not None and not 0.0 <= self.reactive_residual_ratio <= 1.0:
            raise ValueError("reactive_residual_ratio must be in [0, 1]")
        if self.generation_recovery_delay_s < 0:
            raise ValueError("generation_recovery_delay_s must be non-negative")
        if self.generation_recovery_time_constant_s <= 0:
            raise ValueError("generation_recovery_time_constant_s must be positive")
        if not 0.0 <= self.generation_transient_amplitude_ratio <= 2.0:
            raise ValueError("generation_transient_amplitude_ratio must be in [0, 2]")
        if self.generation_transient_peak_s <= 0:
            raise ValueError("generation_transient_peak_s must be positive")
        if not 0.0 <= self.excitation_severity < 1.0:
            raise ValueError("excitation_severity must be in [0, 1)")
        if self.fault_resistance_pu is not None and self.fault_resistance_pu < 0:
            raise ValueError("fault_resistance_pu must be non-negative")
        if self.fault_reactance_pu is not None and self.fault_reactance_pu < 0:
            raise ValueError("fault_reactance_pu must be non-negative")


@dataclass
class ScenarioDataset:
    spec: ScenarioSpec
    bus_ids: np.ndarray
    pmu_buses: np.ndarray
    time_s: np.ndarray
    voltage_pu: np.ndarray
    angle_rad: np.ndarray
    frequency_hz: np.ndarray
    rocof_hz_s: np.ndarray
    observed_voltage_pu: np.ndarray
    observed_angle_rad: np.ndarray
    observed_frequency_hz: np.ndarray
    observed_mask: np.ndarray
    event_label: np.ndarray
    ybus: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            root / "trajectory.npz",
            bus_ids=self.bus_ids,
            pmu_buses=self.pmu_buses,
            time_s=self.time_s,
            voltage_pu=self.voltage_pu,
            angle_rad=self.angle_rad,
            frequency_hz=self.frequency_hz,
            rocof_hz_s=self.rocof_hz_s,
            observed_voltage_pu=self.observed_voltage_pu,
            observed_angle_rad=self.observed_angle_rad,
            observed_frequency_hz=self.observed_frequency_hz,
            observed_mask=self.observed_mask,
            event_label=self.event_label,
            ybus_real=np.real(self.ybus),
            ybus_imag=np.imag(self.ybus),
        )
        payload = {"spec": asdict(self.spec), "metadata": self.metadata}
        (root / "scenario.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return root

    @classmethod
    def load(cls, directory: str | Path) -> "ScenarioDataset":
        root = Path(directory)
        arrays = np.load(root / "trajectory.npz")
        payload = json.loads((root / "scenario.json").read_text(encoding="utf-8"))
        return cls(
            spec=ScenarioSpec(**payload["spec"]),
            bus_ids=arrays["bus_ids"],
            pmu_buses=arrays["pmu_buses"],
            time_s=arrays["time_s"],
            voltage_pu=arrays["voltage_pu"],
            angle_rad=arrays["angle_rad"],
            frequency_hz=arrays["frequency_hz"],
            rocof_hz_s=arrays["rocof_hz_s"],
            observed_voltage_pu=arrays["observed_voltage_pu"],
            observed_angle_rad=arrays["observed_angle_rad"],
            observed_frequency_hz=arrays["observed_frequency_hz"],
            observed_mask=arrays["observed_mask"],
            event_label=arrays["event_label"],
            ybus=arrays["ybus_real"] + 1j * arrays["ybus_imag"],
            metadata=payload.get("metadata", {}),
        )


def _ar1_noise(length: int, std: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    if std <= 0 or length <= 0:
        return np.zeros(length, dtype=float)
    rho = float(np.clip(rho, 0.0, 0.995))
    innovations = rng.normal(0.0, std * math.sqrt(max(1.0 - rho**2, 1e-6)), size=length)
    out = np.zeros(length, dtype=float)
    out[0] = rng.normal(0.0, std)
    for idx in range(1, length):
        out[idx] = rho * out[idx - 1] + innovations[idx]
    return out


def _correlated_ar1_noise(
    length: int,
    correlation: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit-variance AR(1) noise with a fixed cross-PMU correlation."""

    matrix = np.asarray(correlation, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    matrix = (vectors * np.clip(values, 1e-6, None)) @ vectors.T
    diagonal = np.sqrt(np.maximum(np.diag(matrix), 1e-9))
    matrix = matrix / np.outer(diagonal, diagonal)
    cholesky = np.linalg.cholesky(matrix + 1e-8 * np.eye(len(matrix)))
    rho = float(np.clip(rho, -0.995, 0.995))
    output = np.zeros((length, len(matrix)), dtype=float)
    output[0] = rng.normal(size=len(matrix)) @ cholesky.T
    innovation_scale = math.sqrt(max(1.0 - rho * rho, 1e-6))
    for idx in range(1, length):
        innovation = rng.normal(size=len(matrix)) @ cholesky.T
        output[idx] = rho * output[idx - 1] + innovation_scale * innovation
    return output


def _pmu_filter(values: np.ndarray, settings: dict[str, Any], *, derivative: bool = False) -> np.ndarray:
    """Apply one shared, reproducible PMU frequency/ROCOF estimator."""

    data = np.asarray(values, dtype=float)
    method = str(settings.get("method", "identity"))
    window = max(1, int(settings.get("window_samples", 1)))
    if method == "identity" or window <= 1:
        filtered = data.copy()
    elif method == "moving_average_causal":
        padded = np.pad(data, (window - 1, 0), mode="edge")
        filtered = np.convolve(padded, np.ones(window) / window, mode="valid")
    elif method == "exponential_lowpass":
        alpha = float(np.clip(settings.get("alpha", 2.0 / (window + 1.0)), 1e-4, 1.0))
        filtered = np.empty_like(data)
        filtered[0] = data[0]
        for idx in range(1, len(data)):
            filtered[idx] = alpha * data[idx] + (1.0 - alpha) * filtered[idx - 1]
    elif method == "savgol":
        from scipy.signal import savgol_filter

        actual = min(window if window % 2 else window + 1, len(data) - (1 - len(data) % 2))
        actual = max(actual, 3)
        filtered = savgol_filter(data, actual, min(2, actual - 1), mode="interp")
    else:
        raise ValueError(f"Unsupported PMU filter method {method!r}")
    if derivative:
        sample_rate = float(settings.get("sample_rate_hz", 30.0))
        filtered = np.gradient(filtered, 1.0 / sample_rate)
        rocof_window = max(1, int(settings.get("rocof_window_samples", 1)))
        if rocof_window > 1:
            padded = np.pad(filtered, (rocof_window - 1, 0), mode="edge")
            filtered = np.convolve(padded, np.ones(rocof_window) / rocof_window, mode="valid")
    return filtered


def _resolve_line(system, target: str | int | None, rng: np.random.Generator) -> tuple[str, str]:
    indices = list(system.Line.idx.v)
    endpoints = [(int(a), int(b)) for a, b in zip(system.Line.bus1.v, system.Line.bus2.v)]
    if isinstance(target, str) and "-" in target:
        a, b = (int(x) for x in target.split("-", 1))
        for idx, (left, right) in zip(indices, endpoints):
            if {a, b} == {left, right}:
                return str(idx), f"{left}-{right}"
        raise ValueError(f"Line {target!r} is not present in the ANDES IEEE-39 case")
    if target is not None:
        key = str(target)
        if key in {str(idx) for idx in indices}:
            pos = [str(idx) for idx in indices].index(key)
            left, right = endpoints[pos]
            return key, f"{left}-{right}"
    pos = int(rng.integers(0, len(indices)))
    left, right = endpoints[pos]
    return str(indices[pos]), f"{left}-{right}"


class DeviceNotFoundError(LookupError):
    """Raised when a requested physical device is absent at an exact bus."""


def _resolve_device_by_bus(model, target: str | int | None) -> tuple[str, int]:
    """Resolve an exact device bus without random or implicit substitution."""
    indices = [str(idx) for idx in model.idx.v]
    buses = [int(bus) for bus in model.bus.v]
    if target is None:
        raise DeviceNotFoundError("An explicit bus is required for physical device resolution")
    requested = int(target)
    if requested not in buses:
        model_name = getattr(model, "class_name", model.__class__.__name__)
        raise DeviceNotFoundError(
            f"No {model_name} device exists at BUS{requested}; available buses={buses}"
        )
    pos = buses.index(requested)
    return indices[pos], buses[pos]


def _build_andes_system(spec: ScenarioSpec, event_scheduler=None):
    import andes

    rng = np.random.default_rng(spec.seed)
    system = andes.load(
        andes.get_case("ieee39/ieee39_full.xlsx"),
        setup=False,
        no_output=True,
    )
    generator_physical_buses = np.asarray(system.GENROU.bus.v, dtype=int).copy()

    resolved: dict[str, Any] = {}
    if spec.raw_case_path:
        raw_case = parse_raw_static_case(spec.raw_case_path)
        resolved.update(apply_raw_static_case(system, raw_case))
        transformer_rows=[b.row for b in raw_case.branches if abs(b.tap-1.0)>1e-12 or abs(b.shift_deg)>1e-12]
        bus39_rows=[b.row for b in raw_case.branches if 39 in (b.from_bus,b.to_bus)]
        for row in transformer_rows:
            system.Line.r.v[row]*=spec.source_transformer_impedance_scale;system.Line.x.v[row]*=spec.source_transformer_impedance_scale
        for row in bus39_rows:
            system.Line.r.v[row]*=spec.bus39_source_impedance_scale;system.Line.x.v[row]*=spec.bus39_source_impedance_scale
        if spec.bus39_external_equivalent_reactance_pu > 0:
            # Preserve both authoritative RAW branch impedances and place one
            # explicit external-network boundary reactor between BUS39 and the
            # two incident external paths. This parameter is an R5B.4-only
            # Thevenin-equivalent hypothesis, not a hidden line rescaling.
            bus39_uid = int(np.flatnonzero(np.asarray(system.Bus.idx.v, dtype=int) == 39)[0])
            external_bus = 40
            system.add(
                "Bus", idx=external_bus, name="BUS39_EXTERNAL_EQ",
                Vn=float(system.Bus.Vn.v[bus39_uid]),
                v0=float(system.Bus.v0.v[bus39_uid]),
                a0=float(system.Bus.a0.v[bus39_uid]),
            )
            if spec.bus39_equivalent_placement == "network_side":
                for row in bus39_rows:
                    if int(system.Line.bus1.v[row]) == 39:
                        system.Line.bus1.v[row] = external_bus
                    if int(system.Line.bus2.v[row]) == 39:
                        system.Line.bus2.v[row] = external_bus
            else:
                source_positions = np.flatnonzero(generator_physical_buses == 39)
                if len(source_positions) != 1:
                    raise RuntimeError(f"Expected one source-side GENROU at BUS39, found {len(source_positions)}")
                source_pos = int(source_positions[0])
                static_idx = system.GENROU.gen.v[source_pos]
                if static_idx in system.Slack.idx.v:
                    static_pos = list(system.Slack.idx.v).index(static_idx)
                    system.Slack.bus.v[static_pos] = external_bus
                elif static_idx in system.PV.idx.v:
                    static_pos = list(system.PV.idx.v).index(static_idx)
                    system.PV.bus.v[static_pos] = external_bus
                else:
                    raise RuntimeError(f"Cannot locate linked static source {static_idx!r} for BUS39 GENROU")
                system.GENROU.bus.v[source_pos] = external_bus
            system.add(
                "Line", idx="R5B4_BUS39_EXTERNAL_EQ", bus1=external_bus, bus2=39,
                # Pin the connector to the system MVA base and the BUS39
                # voltage base.  ANDES otherwise applies the Line defaults
                # (100 MVA, 110 kV) during per-unit conversion; at this 345-kV
                # bus that silently turns a declared 0.066 pu(system) reactor
                # into roughly 0.65 pu(system).
                Sn=float(system.config.mva), Vn1=float(system.Bus.Vn.v[bus39_uid]),
                Vn2=float(system.Bus.Vn.v[bus39_uid]),
                r=0.0, x=float(spec.bus39_external_equivalent_reactance_pu), b=0.0,
            )
            resolved.update(
                bus39_external_equivalent_bus=external_bus,
                bus39_external_equivalent_reactance_pu=float(spec.bus39_external_equivalent_reactance_pu),
                bus39_equivalent_placement=spec.bus39_equivalent_placement,
                bus39_external_equivalent_original_branch_rows=bus39_rows,
                bus39_external_equivalent_raw_line_impedances_changed=False,
            )

    base_inertia = np.asarray(system.GENROU.M.v, dtype=float).copy()
    inertia = base_inertia * spec.inertia_scale
    generator_buses = np.asarray(system.GENROU.bus.v, dtype=int)
    regional_scales = {
        "30_32": spec.inertia_group_30_32,
        "33_36": spec.inertia_group_33_36,
        "37_39": spec.inertia_group_37_39,
    }
    for group, buses in GENERATOR_INERTIA_GROUPS.items():
        inertia[np.isin(generator_physical_buses, buses)] *= regional_scales[group]
    system.GENROU.M.v[:] = inertia
    system.GENROU.D.v[:] = np.full(system.GENROU.n, spec.damping, dtype=float)
    for parameter in ("xd1", "xq1"):
        values = np.asarray(getattr(system.GENROU, parameter).v, dtype=float)
        getattr(system.GENROU, parameter).v[:] = values * spec.machine_transient_reactance_scale
    for parameter in ("xd2", "xq2"):
        values = np.asarray(getattr(system.GENROU, parameter).v, dtype=float)
        getattr(system.GENROU, parameter).v[:] = values * spec.machine_transient_reactance_scale * spec.machine_subtransient_reactance_scale
    for parameter in ("Td10", "Tq10"):
        values = np.asarray(getattr(system.GENROU, parameter).v, dtype=float)
        getattr(system.GENROU, parameter).v[:] = values * spec.machine_time_constant_scale
    for parameter in ("Td20", "Tq20"):
        values = np.asarray(getattr(system.GENROU, parameter).v, dtype=float)
        getattr(system.GENROU, parameter).v[:] = values * spec.machine_time_constant_scale * spec.machine_subtransient_time_constant_scale
    bus39_machine = np.flatnonzero(generator_physical_buses == 39)
    if len(bus39_machine) != 1:
        raise RuntimeError(f"Expected one GENROU source at BUS39, found {len(bus39_machine)}")
    pos39 = int(bus39_machine[0])
    for parameter, value in (
        ("xd1", spec.bus39_xd1_pu), ("xq1", spec.bus39_xq1_pu),
        ("xd2", spec.bus39_xd2_pu), ("xq2", spec.bus39_xq2_pu),
        ("Td10", spec.bus39_Td10_s), ("Tq10", spec.bus39_Tq10_s),
        ("Td20", spec.bus39_Td20_s), ("Tq20", spec.bus39_Tq20_s),
    ):
        if value is not None:
            getattr(system.GENROU, parameter).v[pos39] = float(value)
    base_governor_droop = np.asarray(system.TGOV1N.R.v, dtype=float).copy() if system.TGOV1N.n else np.asarray([], dtype=float)
    scaled_governor_droop = np.asarray([], dtype=float)
    if system.TGOV1N.n:
        system.TGOV1N.R.v[:] = (
            np.asarray(system.TGOV1N.R.v, dtype=float) * spec.governor_droop_scale
        )
        scaled_governor_droop = np.asarray(system.TGOV1N.R.v, dtype=float).copy()
        for parameter, component_scale in (("T1", spec.governor_lag_scale), ("T2", spec.turbine_time_scale), ("T3", spec.turbine_time_scale)):
            values = np.asarray(getattr(system.TGOV1N, parameter).v, dtype=float)
            getattr(system.TGOV1N, parameter).v[:] = values * spec.governor_time_scale * component_scale
    if spec.governor_structure == "IEESGO" and system.TGOV1N.n:
        for idx,syn,t1,t2,t3,vmax,vmin in zip(system.TGOV1N.idx.v,system.TGOV1N.syn.v,system.TGOV1N.T1.v,system.TGOV1N.T2.v,system.TGOV1N.T3.v,system.TGOV1N.VMAX.v,system.TGOV1N.VMIN.v):
            system.add("IEESGO",idx=f"R5B_{idx}",syn=syn,T1=max(float(t1),.02),T2=max(float(t2),.02),T3=max(float(t3),.02),T4=.10,T5=max(float(t3),.20),T6=.50,K1=1.0,K2=0.0,K3=0.0,PMAX=float(vmax),PMIN=float(vmin))
        for parameter in system.TGOV1N.params.values():
            parameter.v = []
        system.TGOV1N.n = 0
        system.TGOV1N.uid = {}
    if spec.governor_structure == "IEEEG1" and system.TGOV1N.n:
        # Native single-shaft IEEE steam-governor initialization.  The shaft
        # fractions sum to one and PMAX/PMIN bracket the steady mechanical
        # input; no TGOV1N state or parameter is copied into IEEEG1.
        for idx, syn in zip(system.TGOV1N.idx.v, system.TGOV1N.syn.v):
            system.add(
                "IEEEG1", idx=f"R5B1_{idx}", syn=syn,
                K=20.0 / spec.governor_droop_scale,
                T1=0.50 * spec.governor_lag_scale,
                T2=0.10 * spec.governor_lag_scale,
                T3=0.10 * spec.governor_lag_scale,
                UO=0.20, UC=-0.20, PMAX=5.0, PMIN=0.0,
                T4=0.30 * spec.turbine_time_scale, K1=0.30, K2=0.0,
                T5=5.00 * spec.turbine_time_scale, K3=0.70, K4=0.0,
                T6=0.50 * spec.turbine_time_scale, K5=0.0, K6=0.0,
                T7=0.05 * spec.turbine_time_scale, K7=0.0, K8=0.0,
            )
        for parameter in system.TGOV1N.params.values():
            parameter.v = []
        system.TGOV1N.n = 0
        system.TGOV1N.uid = {}
    if system.IEEEX1.n:
        system.IEEEX1.KA.v[:] = np.asarray(system.IEEEX1.KA.v, dtype=float) * spec.avr_gain_scale
        for parameter in ("TR", "TA", "TC", "TB", "TE", "TF1"):
            values = np.asarray(getattr(system.IEEEX1, parameter).v, dtype=float)
            getattr(system.IEEEX1, parameter).v[:] = values * spec.avr_time_scale
    if system.IEEEST.n:
        system.IEEEST.u.v[:]=np.full(system.IEEEST.n,1 if spec.pss_enabled else 0,dtype=int)
        if spec.pss_enabled:system.IEEEST.KS.v[:]=np.asarray(system.IEEEST.KS.v,dtype=float)*spec.pss_gain_scale

    if abs(spec.operating_scale - 1.0) > 1e-12:
        system.PQ.p0.v[:] = np.asarray(system.PQ.p0.v, dtype=float) * spec.operating_scale
        system.PQ.q0.v[:] = np.asarray(system.PQ.q0.v, dtype=float) * spec.operating_scale
    if system.PV.n and abs(spec.generator_dispatch_scale - 1.0) > 1e-12:
        system.PV.p0.v[:] = np.asarray(system.PV.p0.v, dtype=float) * spec.generator_dispatch_scale
    if system.PV.n and spec.voltage_setpoint_pu is not None:
        system.PV.v0.v[:] = np.full(system.PV.n, float(spec.voltage_setpoint_pu), dtype=float)

    dynamic_load_indices: dict[str, str] = {}
    if abs(spec.load_frequency_exponent) > 1e-12 or abs(spec.load_voltage_exponent) > 1e-12:
        for pq_idx in list(system.PQ.idx.v):
            fload_idx = f"R4_FLOAD_{pq_idx}"
            system.add(
                "FLoad", idx=fload_idx, pq=pq_idx, kp=100.0, kq=100.0,
                ap=spec.load_voltage_exponent, aq=spec.load_voltage_exponent,
                bp=spec.load_frequency_exponent, bq=spec.load_frequency_exponent,
            )
            dynamic_load_indices[str(pq_idx)] = fload_idx

    if event_scheduler is not None:
        event_scheduler(
            system=system,
            base_spec=spec,
            rng=rng,
            resolved=resolved,
            dynamic_load_indices=dynamic_load_indices,
        )
    elif spec.event_type == 1:
        bus = int(spec.target) if spec.target is not None else int(rng.integers(1, 40))
        fault_clear_s = spec.event_start_s + spec.event_duration_s
        rf = 0.0 if spec.fault_resistance_pu is None else spec.fault_resistance_pu
        xf = (
            max(0.001, 0.12 * (1.0 - min(spec.severity, 0.995)))
            if spec.fault_reactance_pu is None
            else spec.fault_reactance_pu
        )
        system.add(
            "Fault",
            bus=bus,
            tf=spec.event_start_s,
            tc=fault_clear_s,
            rf=rf,
            xf=xf,
        )
        post_trip_line = None
        post_trip_line_secondary = None
        if spec.fault_post_trip:
            post_trip_line, endpoints = _resolve_line(system, spec.fault_post_trip, rng)
            system.add("Toggle", model="Line", dev=post_trip_line, t=fault_clear_s)
            resolved["fault_post_trip_target"] = f"LINE{endpoints}"
        if spec.fault_post_trip_secondary:
            post_trip_line_secondary, endpoints_secondary = _resolve_line(system, spec.fault_post_trip_secondary, rng)
            if post_trip_line_secondary == post_trip_line:
                raise ValueError("fault_post_trip_secondary must identify a distinct line")
            system.add("Toggle", model="Line", dev=post_trip_line_secondary, t=fault_clear_s)
            resolved["fault_post_trip_secondary_target"] = f"LINE{endpoints_secondary}"
        resolved.update(
            physical_type="fault",
            physical_target=f"BUS{bus}",
            fault_resistance_pu=float(rf),
            fault_reactance_pu=float(xf),
            fault_post_trip_line_idx=post_trip_line,
            fault_post_trip_secondary_line_idx=post_trip_line_secondary,
        )
    elif spec.event_type == 2:
        line_idx, endpoints = _resolve_line(system, spec.target, rng)
        system.add("Toggle", model="Line", dev=line_idx, t=spec.event_start_s)
        if spec.line_reclose and spec.event_duration_s > 0:
            system.add(
                "Toggle",
                model="Line",
                dev=line_idx,
                t=spec.event_start_s + spec.event_duration_s,
            )
        resolved.update(physical_type="line_outage", physical_target=f"LINE{endpoints}", line_idx=line_idx)
    elif spec.event_type in {3, 6}:
        temporal_bus_modes = {
            "signed_bus_p_recovery_exponential",
            "signed_bus_p_delayed_recovery",
            "signed_bus_p_hold_ramp_recovery",
            "signed_bus_pq_recovery_exponential",
            "signed_bus_pq_delayed_recovery",
            "signed_bus_p_recovery_pulse",
            "signed_bus_pq_recovery_pulse",
        }
        bus_modes = {
            "bus_injection_step", "bus_injection_ramp", "negative_load_step",
            "signed_bus_injection_step", "signed_bus_injection_ramp",
            "signed_bus_pq_injection_step", "signed_bus_pq_injection_ramp",
            *temporal_bus_modes,
        }
        if spec.generation_mode in bus_modes:
            bus = int(spec.target) if spec.target is not None else 2
            injection_indices = []
            def add_bus_injection(label: str, p0: float, q0: float, toggle_time: float) -> None:
                injection_idx = f"SIM_INJECTION_BUS{bus}_{label}"
                system.add("PQ", idx=injection_idx, bus=bus, p0=float(p0), q0=float(q0), u=0)
                system.add("Toggle", model="PQ", dev=injection_idx, t=toggle_time)
                if spec.event_restore_s is not None:
                    system.add("Toggle", model="PQ", dev=injection_idx, t=spec.event_restore_s)
                injection_indices.append(injection_idx)

            if spec.generation_mode in temporal_bus_modes:
                p_initial = float(spec.severity)
                uses_q = spec.generation_mode.startswith("signed_bus_pq_")
                q_initial = (0.0 if spec.reactive_severity is None else float(spec.reactive_severity)) if uses_q else 0.0
                add_bus_injection("INITIAL", p_initial, q_initial, spec.event_start_s)
                p_ratio = float(spec.generation_residual_ratio)
                q_ratio = p_ratio if spec.reactive_residual_ratio is None else float(spec.reactive_residual_ratio)
                recovery_start = spec.event_start_s + float(spec.generation_recovery_delay_s)
                if "delayed_recovery" in spec.generation_mode:
                    add_bus_injection("RECOVERY_00", -p_initial * (1.0 - p_ratio),
                                      -q_initial * (1.0 - q_ratio), recovery_start)
                elif "hold_ramp_recovery" in spec.generation_mode:
                    recovery_steps = 12
                    for step in range(1, recovery_steps + 1):
                        add_bus_injection(
                            f"RECOVERY_{step:02d}",
                            -p_initial * (1.0 - p_ratio) / recovery_steps,
                            -q_initial * (1.0 - q_ratio) / recovery_steps,
                            recovery_start + step * spec.generation_ramp_duration_s / recovery_steps,
                        )
                elif "recovery_pulse" in spec.generation_mode:
                    tau = float(spec.generation_recovery_time_constant_s)
                    peak_s = float(spec.generation_transient_peak_s)
                    amplitude = float(spec.generation_transient_amplitude_ratio)
                    available = max(spec.simulation_end_s - recovery_start - spec.tstep_s, spec.tstep_s)
                    recovery_steps = 28
                    previous_p = 1.0
                    previous_q = 1.0
                    for step in range(1, recovery_steps + 1):
                        elapsed = available * step / recovery_steps
                        pulse = amplitude * (elapsed / peak_s) * math.exp(1.0 - elapsed / peak_s)
                        current_p = p_ratio + (1.0 - p_ratio) * math.exp(-elapsed / tau) + pulse
                        current_q = q_ratio + (1.0 - q_ratio) * math.exp(-elapsed / tau) + pulse
                        add_bus_injection(
                            f"RECOVERY_{step:02d}",
                            p_initial * (current_p - previous_p),
                            q_initial * (current_q - previous_q),
                            recovery_start + elapsed,
                        )
                        previous_p, previous_q = current_p, current_q
                else:
                    tau = float(spec.generation_recovery_time_constant_s)
                    available = max(spec.simulation_end_s - recovery_start - spec.tstep_s, spec.tstep_s)
                    duration = min(5.0 * tau, available)
                    recovery_steps = 16
                    previous_p = 1.0
                    previous_q = 1.0
                    for step in range(1, recovery_steps + 1):
                        elapsed = duration * step / recovery_steps
                        current_p = p_ratio + (1.0 - p_ratio) * math.exp(-elapsed / tau)
                        current_q = q_ratio + (1.0 - q_ratio) * math.exp(-elapsed / tau)
                        add_bus_injection(
                            f"RECOVERY_{step:02d}",
                            p_initial * (current_p - previous_p),
                            q_initial * (current_q - previous_q),
                            recovery_start + elapsed,
                        )
                        previous_p, previous_q = current_p, current_q
            else:
                ramp_steps = 10 if spec.generation_mode in {"bus_injection_ramp", "signed_bus_injection_ramp", "signed_bus_pq_injection_ramp"} else 1
                for step in range(ramp_steps):
                    signed = spec.generation_mode in {"signed_bus_injection_step", "signed_bus_injection_ramp", "signed_bus_pq_injection_step", "signed_bus_pq_injection_ramp"}
                    p0 = float(spec.severity) / ramp_steps if signed else abs(float(spec.severity)) / ramp_steps
                    q_total = 0.0 if spec.reactive_severity is None else float(spec.reactive_severity)
                    q0 = q_total / ramp_steps if spec.generation_mode in {"signed_bus_pq_injection_step", "signed_bus_pq_injection_ramp"} else 0.0
                    add_bus_injection(f"{step:02d}", p0, q0,
                                      spec.event_start_s + step * spec.generation_ramp_duration_s / ramp_steps)
            gen_idx = ",".join(injection_indices)
        else:
            device_target = spec.generation_device_bus if spec.generation_device_bus is not None else spec.target
            gen_idx, device_bus = _resolve_device_by_bus(system.GENROU, device_target)
            bus = int(spec.target) if spec.target is not None else device_bus
        if spec.generation_mode in bus_modes:
            pass
        elif spec.generation_mode == "generator_trip":
            system.add("Toggle", model="GENROU", dev=gen_idx, t=spec.event_start_s)
            if spec.event_restore_s is not None:
                system.add("Toggle", model="GENROU", dev=gen_idx, t=spec.event_restore_s)
        else:
            factor = max(0.01, 1.0 - abs(spec.severity))
            ramp_steps = 10 if spec.generation_mode == "mechanical_power_ramp" else 1
            step_factor = factor ** (1.0 / ramp_steps)
            for step in range(ramp_steps):
                alter_time = spec.event_start_s + step * spec.generation_ramp_duration_s / ramp_steps
                system.add("Alter", model="GENROU", dev=gen_idx, src="tm0", attr="v", method="*", amount=step_factor, t=alter_time)
            if spec.generation_mode == "torque_excitation":
                exciter_indices = [str(idx) for idx in system.IEEEX1.idx.v]
                exciter_syn = [str(idx) for idx in system.IEEEX1.syn.v]
                if gen_idx not in exciter_syn:
                    raise ValueError(f"No IEEEX1 exciter is linked to {gen_idx}")
                exciter_idx = exciter_indices[exciter_syn.index(gen_idx)]
                excitation_factor = 1.0 - spec.excitation_severity
                system.add(
                    "Alter",
                    model="IEEEX1",
                    dev=exciter_idx,
                    src="vref0",
                    attr="v",
                    method="*",
                    amount=excitation_factor,
                    t=spec.event_start_s,
                )
                if spec.event_restore_s is not None:
                    system.add(
                        "Alter",
                        model="IEEEX1",
                        dev=exciter_idx,
                        src="vref0",
                        attr="v",
                        method="*",
                        amount=1.0 / excitation_factor,
                        t=spec.event_restore_s,
                    )
                resolved["exciter_idx"] = exciter_idx
            if spec.event_restore_s is not None:
                system.add(
                    "Alter",
                    model="GENROU",
                    dev=gen_idx,
                    src="tm0",
                    attr="v",
                    method="*",
                    amount=1.0 / factor,
                    t=spec.event_restore_s,
                )
        resolved.update(
            physical_type="generation_change",
            physical_target=f"BUS{bus}",
            generator_idx=gen_idx,
            generation_device_bus=(None if spec.generation_mode in bus_modes else int(device_bus)),
            generation_mode=spec.generation_mode,
            generation_residual_ratio=spec.generation_residual_ratio,
            reactive_residual_ratio=spec.reactive_residual_ratio,
            generation_recovery_delay_s=spec.generation_recovery_delay_s,
            generation_recovery_time_constant_s=spec.generation_recovery_time_constant_s,
            generation_transient_amplitude_ratio=spec.generation_transient_amplitude_ratio,
            generation_transient_peak_s=spec.generation_transient_peak_s,
            excitation_severity=spec.excitation_severity,
        )
    elif spec.event_type == 4:
        system.PQ.config.p2p = 1.0
        system.PQ.config.p2i = 0.0
        system.PQ.config.p2z = 0.0
        system.PQ.config.q2q = 1.0
        system.PQ.config.q2i = 0.0
        system.PQ.config.q2z = 0.0
        load_idx, bus = _resolve_device_by_bus(system.PQ, spec.target)
        p_factor = max(0.01, 1.0 + spec.severity)
        q_change = spec.severity if spec.reactive_severity is None else spec.reactive_severity
        q_factor = max(0.01, 1.0 + q_change)
        alter_model = "FLoad" if str(load_idx) in dynamic_load_indices else "PQ"
        alter_idx = dynamic_load_indices.get(str(load_idx), load_idx)
        sources = ("pv0", "qv0") if alter_model == "FLoad" else ("Ppf", "Qpf")
        for source, factor in zip(sources, (p_factor, q_factor)):
            system.add("Alter", model=alter_model, dev=alter_idx, src=source, attr="v", method="*", amount=factor, t=spec.event_start_s)
            if spec.event_restore_s is not None:
                system.add("Alter", model=alter_model, dev=alter_idx, src=source, attr="v", method="*", amount=1.0/factor, t=spec.event_restore_s)
        resolved.update(physical_type="load_change", physical_target=f"BUS{bus}", load_idx=load_idx, dynamic_load_idx=dynamic_load_indices.get(str(load_idx)))
    elif spec.event_type == 8:
        # A mixed event is deliberately outside the single-event catalogue.
        system.PQ.config.p2p = 1.0
        system.PQ.config.p2i = 0.0
        system.PQ.config.p2z = 0.0
        system.PQ.config.q2q = 1.0
        system.PQ.config.q2i = 0.0
        system.PQ.config.q2z = 0.0
        gen_idx, gen_bus = _resolve_device_by_bus(system.GENROU, spec.target)
        load_idx, load_bus = _resolve_device_by_bus(system.PQ, spec.target)
        system.add(
            "Alter",
            model="GENROU",
            dev=gen_idx,
            src="tm0",
            attr="v",
            method="*",
            amount=max(0.3, 1.0 - abs(spec.severity)),
            t=spec.event_start_s,
        )
        for source in ("Ppf", "Qpf"):
            system.add(
                "Alter",
                model="PQ",
                dev=load_idx,
                src=source,
                attr="v",
                method="*",
                amount=1.0 + 0.5 * abs(spec.severity),
                t=spec.event_start_s + spec.tstep_s,
            )
        resolved.update(
            physical_type="mixed_generation_load",
            physical_target=f"BUS{gen_bus}+BUS{load_bus}",
            generator_idx=gen_idx,
            load_idx=load_idx,
        )
    else:
        resolved.update(physical_type="normal", physical_target=None)

    resolved["actual_andes_runtime_parameters"] = {
        "inertia_scale": float(spec.inertia_scale),
        "GENROU_M_base": base_inertia.tolist(),
        "GENROU_M_applied": np.asarray(system.GENROU.M.v, dtype=float).tolist(),
        "governor_droop_scale": float(spec.governor_droop_scale),
        "TGOV1N_R_base": base_governor_droop.tolist(),
        "TGOV1N_R_scaled_before_structure_conversion": scaled_governor_droop.tolist(),
        "governor_structure": str(spec.governor_structure),
        "IEEEG1_K_if_used": float(20.0 / spec.governor_droop_scale),
        "event_severity_backend": None if spec.severity is None else float(spec.severity),
        "event_reactive_severity_backend": None if spec.reactive_severity is None else float(spec.reactive_severity),
        "generation_transient_amplitude_ratio": float(spec.generation_transient_amplitude_ratio),
        "generation_transient_peak_s": float(spec.generation_transient_peak_s),
    }

    system.setup()
    if system.PFlow.run() is False:
        raise RuntimeError(f"ANDES power flow failed for {spec.scenario_id}")
    system.TDS.config.tf = float(spec.simulation_end_s)
    system.TDS.config.tstep = float(spec.tstep_s)
    if hasattr(system.TDS.config, "criteria"):
        system.TDS.config.criteria = 0
    ok_tds = system.TDS.run()
    init_ok = getattr(system.TDS, "test_ok", None)
    resolved["tds_initialization_ok"] = init_ok
    if init_ok is False:
        raise RuntimeError(f"ANDES TDS initialization failed for {spec.scenario_id}")
    if ok_tds is False:
        available_time = np.asarray(getattr(system.dae.ts, "t", []), dtype=float)
        reached_horizon = (
            len(available_time) >= 3
            and float(available_time[-1]) >= spec.simulation_end_s - 2.0 * spec.tstep_s
        )
        if not reached_horizon:
            raise RuntimeError(f"ANDES TDS failed before the requested horizon for {spec.scenario_id}")
        resolved["andes_returned_false_at_horizon"] = True
    return system, resolved


def _event_label(spec: ScenarioSpec, time_s: np.ndarray) -> np.ndarray:
    t = np.asarray(time_s, dtype=float)
    if spec.event_type == 0:
        return np.zeros(len(t), dtype=int)
    if spec.event_type == 1:
        active = (t >= spec.event_start_s) & (t <= spec.event_start_s + spec.event_duration_s)
    elif spec.event_type in {5, 7}:
        active = (t >= spec.event_start_s) & (t <= spec.event_start_s + spec.event_duration_s)
    elif spec.event_restore_s is not None:
        active = (t >= spec.event_start_s) & (t <= spec.event_restore_s)
    else:
        active = t >= spec.event_start_s
    return np.where(active, spec.event_type, 0).astype(int)


def _calibrated_observations(
    spec: ScenarioSpec,
    calibration: MeasurementCalibration,
    bus_ids: np.ndarray,
    pmu_buses: np.ndarray,
    voltage: np.ndarray,
    angle: np.ndarray,
    frequency: np.ndarray,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec.seed + 10_007)
    positions = {int(bus): idx for idx, bus in enumerate(bus_ids)}
    obs_idx = [positions[int(bus)] for bus in pmu_buses]
    v = voltage[:, obs_idx].copy()
    a = angle[:, obs_idx].copy()
    f = frequency[:, obs_idx].copy()
    mask = np.ones_like(v, dtype=bool)

    for col, bus in enumerate(pmu_buses):
        v_profile = calibration.profile(int(bus), "VA_MAG")
        median_mag = max(abs(v_profile.median), 1.0)
        v_sigma = float(np.clip(v_profile.noise_std / median_mag, 1e-6, 0.02))
        a_profile = calibration.profile(int(bus), "VA_ANG")
        a_sigma = math.radians(float(np.clip(a_profile.noise_std, 1e-5, 0.5)))
        f_profile = calibration.profile(int(bus), "Freq")
        f_sigma = float(np.clip(f_profile.noise_std, 1e-6, 0.05))
        v[:, col] += _ar1_noise(len(time_s), v_sigma, v_profile.ar1, rng)
        a[:, col] += _ar1_noise(len(time_s), a_sigma, a_profile.ar1, rng)
        f[:, col] += _ar1_noise(len(time_s), f_sigma, f_profile.ar1, rng)

    cyber_bus = int(spec.cyber_target or pmu_buses[int(rng.integers(0, len(pmu_buses)))])
    if cyber_bus not in set(int(x) for x in pmu_buses):
        raise ValueError(f"cyber_target BUS{cyber_bus} is not one of the observed PMUs")
    cyber_col = list(int(x) for x in pmu_buses).index(cyber_bus)
    cyber_active = (time_s >= spec.event_start_s) & (time_s <= spec.event_start_s + spec.event_duration_s)
    if spec.event_type in {5, 6}:
        mask[cyber_active, cyber_col] = False
        v[~mask] = np.nan
        a[~mask] = np.nan
        f[~mask] = np.nan
    if spec.event_type in {7, 8}:
        # Coordinated but physically inconsistent corruption at one PMU.
        v[cyber_active, cyber_col] *= 1.0 + max(0.02, 0.5 * abs(spec.severity))
        a[cyber_active, cyber_col] += math.radians(2.0 + 20.0 * abs(spec.severity))
        f[cyber_active, cyber_col] += 0.05 + 0.5 * abs(spec.severity)
    return v, a, f, mask


def _export_observed_csvs(
    dataset: ScenarioDataset,
    signals_by_bus: dict[str, dict[str, np.ndarray]],
    calibration: MeasurementCalibration,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(dataset.spec.seed + 23_003)
    event = dataset.event_label
    common_fraction = float(np.clip(calibration.common_angle_noise_fraction, 0.0, 0.99))
    reference_profile = calibration.profile(int(dataset.pmu_buses[-1]), "VA_ANG")
    common_angle_noise = _ar1_noise(
        len(dataset.time_s),
        reference_profile.noise_std * common_fraction,
        reference_profile.ar1,
        np.random.default_rng(dataset.spec.seed + 23_001),
    )
    def noise_family(suffix: str) -> str:
        if suffix in {"Freq", "ROCOF"}:
            return suffix
        return ("V" if suffix.startswith("V") else "I") + ("_ANG" if suffix.endswith("_ANG") else "_MAG")

    correlated_noise: dict[str, np.ndarray] = {}
    for family, matrix in calibration.noise_correlation_matrices.items():
        array = np.asarray(matrix, dtype=float)
        if array.shape != (len(dataset.pmu_buses), len(dataset.pmu_buses)):
            raise ValueError(
                f"Noise correlation {family!r} has shape {array.shape}; "
                f"expected {(len(dataset.pmu_buses), len(dataset.pmu_buses))}"
            )
        correlated_noise[family] = _correlated_ar1_noise(
            len(dataset.time_s),
            array,
            calibration.noise_ar1.get(family, 0.0),
            np.random.default_rng(dataset.spec.seed + 31_000 + len(correlated_noise)),
        )
    for col, bus_value in enumerate(dataset.pmu_buses):
        bus = int(bus_value)
        signals = signals_by_bus[str(bus)]
        frame: dict[str, np.ndarray] = {"TIMESTAMP": dataset.time_s.copy()}
        for suffix in RAW_SUFFIXES:
            clean = np.asarray(signals[suffix], dtype=float).copy()
            profile = calibration.profile(bus, suffix)
            pre = dataset.time_s < dataset.spec.event_start_s
            family = noise_family(suffix)
            residual_noise = (
                correlated_noise[family][:, col] * profile.noise_std
                if family in correlated_noise
                else _ar1_noise(len(clean), profile.noise_std, profile.ar1, rng)
            )
            if suffix == "Freq" and calibration.frequency_filter:
                clean = _pmu_filter(np.asarray(signals["Freq"], dtype=float), calibration.frequency_filter)
            elif suffix == "ROCOF" and calibration.frequency_filter:
                clean = _pmu_filter(
                    np.asarray(signals["Freq"], dtype=float),
                    calibration.frequency_filter,
                    derivative=True,
                )
            if suffix.endswith("_ANG"):
                if calibration.angle_reference_mode == "common_reference":
                    bias_map = (
                        calibration.voltage_angle_bias_deg
                        if suffix.startswith("V")
                        else calibration.current_angle_bias_deg
                    )
                    bias = float(bias_map.get(str(bus), 0.0))
                    clean = _wrap_deg(
                        clean + calibration.common_angle_offset_deg + bias
                    )
                    independent_std = profile.noise_std * math.sqrt(1.0 - common_fraction**2)
                    noisy = _wrap_deg(
                        clean
                        + common_angle_noise
                        + residual_noise * (independent_std / max(profile.noise_std, 1e-12))
                    )
                else:
                    clean = _wrap_deg(clean + profile.median - _circular_center_deg(clean[pre]))
                    noisy = _wrap_deg(clean + residual_noise)
            elif suffix.endswith("_MAG"):
                clean_med = float(np.median(clean[pre])) if np.any(pre) else float(np.median(clean))
                if abs(clean_med) > 1e-12 and abs(profile.median) > 1e-12:
                    clean *= profile.median / clean_med
                noisy = clean + residual_noise
            elif suffix == "Freq":
                clean += profile.median - float(np.median(clean[pre]))
                noisy = clean + residual_noise
            else:
                noisy = clean + residual_noise
            cyber_bus = int(dataset.spec.cyber_target or -1)
            cyber_active = (
                (dataset.time_s >= dataset.spec.event_start_s)
                & (dataset.time_s <= dataset.spec.event_start_s + dataset.spec.event_duration_s)
            )
            if dataset.spec.event_type in {7, 8} and bus == cyber_bus:
                if suffix.endswith("_MAG"):
                    noisy[cyber_active] *= 1.0 + max(0.02, 0.5 * abs(dataset.spec.severity))
                elif suffix.endswith("_ANG"):
                    noisy[cyber_active] = _wrap_deg(
                        noisy[cyber_active] + 2.0 + 20.0 * abs(dataset.spec.severity)
                    )
                elif suffix == "Freq":
                    noisy[cyber_active] += 0.05 + 0.5 * abs(dataset.spec.severity)
            frame[f"BUS{bus}_{suffix}"] = noisy

        data_present = dataset.observed_mask[:, col].astype(int)
        for suffix in RAW_SUFFIXES:
            values = frame[f"BUS{bus}_{suffix}"].copy()
            values[data_present == 0] = np.nan
            frame[f"BUS{bus}_{suffix}"] = values
        frame["DATA_PRESENT"] = data_present
        frame["Event"] = event
        pd.DataFrame(frame).to_csv(directory / f"Bus{bus}_Competition_Data_nanmask.csv", index=False)


def simulate_scenario(
    spec: ScenarioSpec,
    calibration: MeasurementCalibration,
    output_dir: str | Path | None = None,
    pmu_buses: tuple[int, ...] = DEFAULT_PMU_BUSES,
    event_scheduler=None,
    event_labeler=None,
) -> ScenarioDataset:
    """Run ANDES, expose eight calibrated PMUs, and optionally persist artifacts."""

    system, resolved = _build_andes_system(spec, event_scheduler=event_scheduler)
    fault_bus = str(spec.target) if spec.event_type == 1 and spec.target is not None else "0"
    signals_by_bus, meta = extract_all_bus_signals(system, fault_bus)
    replace_pmu_currents_with_branch_channels(system, signals_by_bus, calibration.current_mappings)
    bus_ids = np.asarray(sorted((int(x) for x in meta["bus_ids_common"])), dtype=int)
    columns = [str(x) for x in bus_ids]
    time_s = np.asarray(meta["t"], dtype=float)
    voltage = meta["v_df"][columns].to_numpy(dtype=float)
    angle = meta["a_df"][columns].to_numpy(dtype=float)
    frequency = np.column_stack([signals_by_bus[str(bus)]["Freq"] for bus in bus_ids])
    rocof = np.column_stack([signals_by_bus[str(bus)]["ROCOF"] for bus in bus_ids])
    pmus = np.asarray(tuple(int(x) for x in pmu_buses), dtype=int)
    observed_v, observed_a, observed_f, observed_mask = _calibrated_observations(
        spec,
        calibration,
        bus_ids,
        pmus,
        voltage,
        angle,
        frequency,
        time_s,
    )
    labels = _event_label(spec, time_s) if event_labeler is None else np.asarray(
        event_labeler(time_s), dtype=int
    )
    if labels.shape != time_s.shape:
        raise ValueError("event_labeler must return one integer label per TDS time sample")
    dataset = ScenarioDataset(
        spec=spec,
        bus_ids=bus_ids,
        pmu_buses=pmus,
        time_s=time_s,
        voltage_pu=voltage,
        angle_rad=angle,
        frequency_hz=frequency,
        rocof_hz_s=rocof,
        observed_voltage_pu=observed_v,
        observed_angle_rad=observed_a,
        observed_frequency_hz=observed_f,
        observed_mask=observed_mask,
        event_label=labels,
        ybus=np.asarray(meta["ybus"], dtype=complex),
        metadata={
            **resolved,
            "observed_pmus": [int(x) for x in pmus],
            "hidden_buses": [int(x) for x in bus_ids if int(x) not in set(pmus)],
            "andes_case": "ieee39/ieee39_full.xlsx",
            "ground_truth_bus_count": int(len(bus_ids)),
        },
    )
    if output_dir is not None:
        root = dataset.save(output_dir)
        _export_observed_csvs(dataset, signals_by_bus, calibration, root / "pmu")
    return dataset
