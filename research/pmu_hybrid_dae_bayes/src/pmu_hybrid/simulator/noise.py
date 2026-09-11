"""Deterministic PMU corruption model: noise, biases, outliers and integrity masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pmu_hybrid.utils.seeds import rng_for


@dataclass(frozen=True)
class MeasurementNoiseSpec:
    voltage_std_pu: float = 5e-4
    current_std_pu: float = 1e-3
    frequency_std_hz: float = 1e-3
    rocof_std_hz_s: float = 1e-2
    ar1: float = 0.0
    cross_pmu_correlation: float = 0.0
    voltage_gain_bias: float = 0.0
    current_gain_bias: float = 0.0
    phase_bias_rad: float = 0.0
    frequency_bias_hz: float = 0.0
    quantization_pu: float | None = None
    outlier_probability: float = 0.0
    outlier_scale: float = 10.0

    def __post_init__(self) -> None:
        if abs(self.ar1) >= 1.0 or not 0.0 <= self.cross_pmu_correlation < 1.0:
            raise ValueError("AR(1) and cross-PMU correlation must be in [-1,1) and [0,1)")
        if not 0.0 <= self.outlier_probability <= 1.0:
            raise ValueError("outlier_probability must be in [0,1]")


@dataclass(frozen=True)
class CorruptedMeasurements:
    voltage_pu: np.ndarray
    current_pu: np.ndarray
    frequency_hz: np.ndarray
    rocof_hz_s: np.ndarray
    data_present: np.ndarray


def _colored_noise(shape: tuple[int, int], standard_deviation: float, spec: MeasurementNoiseSpec, rng: np.random.Generator) -> np.ndarray:
    innovation = rng.normal(size=shape)
    if spec.cross_pmu_correlation:
        common = rng.normal(size=(shape[0], 1))
        innovation = np.sqrt(1.0 - spec.cross_pmu_correlation) * innovation + np.sqrt(spec.cross_pmu_correlation) * common
    output = np.empty(shape, dtype=float)
    output[0] = innovation[0]
    for index in range(1, shape[0]):
        output[index] = spec.ar1 * output[index - 1] + np.sqrt(1.0 - spec.ar1**2) * innovation[index]
    return standard_deviation * output


def corrupt_measurements(voltage_pu: np.ndarray, current_pu: np.ndarray, frequency_hz: np.ndarray, rocof_hz_s: np.ndarray, data_present: np.ndarray, spec: MeasurementNoiseSpec, *, seed: int, namespace: str) -> CorruptedMeasurements:
    """Apply a seeded observation-only corruption; no physical q is altered."""
    voltage = np.asarray(voltage_pu, dtype=complex)
    current = np.asarray(current_pu, dtype=complex)
    frequency = np.asarray(frequency_hz, dtype=float)
    rocof = np.asarray(rocof_hz_s, dtype=float)
    present = np.asarray(data_present, dtype=bool)
    if voltage.shape != current.shape or voltage.shape != frequency.shape or voltage.shape != rocof.shape or voltage.shape != present.shape:
        raise ValueError("All PMU arrays must share shape (frames, observed_buses)")
    rng = rng_for(seed, namespace)
    phase = np.exp(1j * spec.phase_bias_rad)
    voltage_noisy = voltage * (1.0 + spec.voltage_gain_bias) * phase
    current_noisy = current * (1.0 + spec.current_gain_bias) * phase
    voltage_noisy = voltage_noisy + _colored_noise(voltage.shape, spec.voltage_std_pu, spec, rng) + 1j * _colored_noise(voltage.shape, spec.voltage_std_pu, spec, rng)
    current_noisy = current_noisy + _colored_noise(current.shape, spec.current_std_pu, spec, rng) + 1j * _colored_noise(current.shape, spec.current_std_pu, spec, rng)
    frequency_noisy = frequency + spec.frequency_bias_hz + _colored_noise(frequency.shape, spec.frequency_std_hz, spec, rng)
    rocof_noisy = rocof + _colored_noise(rocof.shape, spec.rocof_std_hz_s, spec, rng)
    if spec.outlier_probability:
        outliers = rng.random(voltage.shape) < spec.outlier_probability
        voltage_noisy = voltage_noisy + outliers * spec.outlier_scale * spec.voltage_std_pu * (rng.standard_t(3, size=voltage.shape) + 1j * rng.standard_t(3, size=voltage.shape))
        current_noisy = current_noisy + outliers * spec.outlier_scale * spec.current_std_pu * (rng.standard_t(3, size=current.shape) + 1j * rng.standard_t(3, size=current.shape))
    if spec.quantization_pu is not None:
        step = spec.quantization_pu
        if step <= 0:
            raise ValueError("quantization_pu must be positive")
        voltage_noisy = step * np.round(voltage_noisy.real / step) + 1j * step * np.round(voltage_noisy.imag / step)
        current_noisy = step * np.round(current_noisy.real / step) + 1j * step * np.round(current_noisy.imag / step)
    voltage_noisy = np.where(present, voltage_noisy, np.nan + 1j * np.nan)
    current_noisy = np.where(present, current_noisy, np.nan + 1j * np.nan)
    frequency_noisy = np.where(present, frequency_noisy, np.nan)
    rocof_noisy = np.where(present, rocof_noisy, np.nan)
    return CorruptedMeasurements(voltage_noisy, current_noisy, frequency_noisy, rocof_noisy, present)
