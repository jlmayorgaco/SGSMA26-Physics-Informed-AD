"""RAW-to-ANDES transient validation utilities.

Absolute PMU phase angles rotate continuously in the competition recordings.
Comparisons therefore use a local pre-event operating point and remove only the
pre-event linear angle trend. Voltage/current magnitudes retain percentage
deviations, while active/reactive powers retain physical MW/MVAr deviations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.simulation.transient_scenarios import (
    DEFAULT_PMU_BUSES,
    MeasurementCalibration,
    ScenarioDataset,
    ScenarioSpec,
    simulate_scenario,
)


SIGNALS = (
    "V_MAG_PCT",
    "I_MAG_PCT",
    "V_ANG_DEG",
    "I_ANG_DEG",
    "FREQ_MHZ",
    "ROCOF_MHZ_S",
    "P_MW",
    "Q_MVAR",
)

SIGNAL_LABELS = {
    "V_MAG_PCT": r"$\Delta |V_A|$ [%]",
    "I_MAG_PCT": r"$\Delta |I_A|$ [%]",
    "V_ANG_DEG": r"$\Delta \theta_{V_A}$ [deg]",
    "I_ANG_DEG": r"$\Delta \theta_{I_A}$ [deg]",
    "FREQ_MHZ": r"$\Delta f$ [mHz]",
    "ROCOF_MHZ_S": r"$\Delta$ROCOF [mHz/s]",
    "P_MW": r"$\Delta P_{3\phi}$ [MW]",
    "Q_MVAR": r"$\Delta Q_{3\phi}$ [MVAr]",
}

EVENT_NAMES = {
    0: "Normal operation",
    1: "Three-phase fault in the BUS39-BUS1 corridor",
    2: "Line outage LINE23-24",
    3: "Generation change at BUS2 / GENROU BUS30",
    4: "Load change at BUS7",
    5: "Missing data at PMU29",
    6: "Generation change plus missing PMU29",
    7: "Bad PMU data",
    8: "Unknown/mixed proxy (no RAW reference)",
}


@dataclass(frozen=True)
class RawEventWindow:
    event_type: int
    onset_s: float | None
    target: str | int | None
    cyber_target: int | None
    duration_s: float


@dataclass
class TunedAndesParameters:
    inertia_scale: float = 1.0
    inertia_group_30_32: float = 1.0
    inertia_group_33_36: float = 1.0
    inertia_group_37_39: float = 1.0
    damping: float = 0.0
    governor_droop_scale: float = 1.0
    fault_severity: float = 0.8
    fault_bus: int = 39
    fault_resistance_pu: float = 0.0
    fault_reactance_pu: float = 0.018
    fault_post_trip: str | None = None
    fault_duration_s: float = 0.10
    line_outage_duration_s: float = 0.10
    generation_severity: float = 0.10
    generation_mode: str = "torque_step"
    excitation_severity: float = 0.0
    load_severity: float = 0.10
    load_q_severity: float = 0.10
    event3_fit_score: float | None = None
    event1_fit_score: float | None = None
    event4_fit_score: float | None = None
    event2_fit_score: float | None = None
    event2_blind_score: float | None = None
    event6_transfer_score: float | None = None
    event1_alignment_s: float = 0.0
    event2_alignment_s: float = 0.0
    event3_alignment_s: float = 0.0
    event4_alignment_s: float = 0.0
    event6_alignment_s: float = 0.0
    train_objective_score: float | None = None
    holdout_objective_score: float | None = None

    def to_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return target

    @classmethod
    def from_json(cls, path: str | Path) -> "TunedAndesParameters":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def alignment_for_event(self, event_type: int) -> float:
        return float(getattr(self, f"event{event_type}_alignment_s", 0.0))


def _bus_from_path(path: Path) -> int:
    match = re.search(r"Bus(\d+)", path.name, re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not infer bus number from {path.name}")
    return int(match.group(1))


def load_pmu_frames(directory: str | Path) -> dict[int, pd.DataFrame]:
    root = Path(directory)
    frames = {_bus_from_path(path): pd.read_csv(path) for path in root.glob("Bus*.csv")}
    missing = sorted(set(DEFAULT_PMU_BUSES) - set(frames))
    if missing:
        raise FileNotFoundError(f"Missing PMU files for buses {missing} in {root}")
    return {bus: frames[bus] for bus in DEFAULT_PMU_BUSES}


def discover_raw_event_windows(frames: dict[int, pd.DataFrame]) -> dict[int, RawEventWindow]:
    """Discover first event onset from row-aligned RAW PMUs.

    Event 3 starts concurrently with event 6: seven PMUs label the physical
    generation change as 3 while PMU29 labels the combined missing-data event 6.
    """

    first = frames[DEFAULT_PMU_BUSES[0]]
    time_s = first["TIMESTAMP"].to_numpy(dtype=float)
    events = np.column_stack(
        [frames[bus]["Event"].fillna(0).to_numpy(dtype=int) for bus in DEFAULT_PMU_BUSES]
    )

    def onset(label: int) -> float | None:
        locations = np.argwhere(events == label)
        if len(locations) == 0:
            return None
        return float(time_s[int(np.min(locations[:, 0]))])

    def first_segment_duration(label: int) -> float:
        rows = np.flatnonzero(np.any(events == label, axis=1))
        if len(rows) == 0:
            return 0.0
        breaks = np.flatnonzero(np.diff(rows) > 1)
        segment = rows[: breaks[0] + 1] if len(breaks) else rows
        return float(time_s[segment[-1]] - time_s[segment[0]] + np.median(np.diff(time_s)))

    normal = events.max(axis=1) == 0
    normal_idx = np.flatnonzero(normal & (time_s > 900.0))
    normal_onset = float(time_s[normal_idx[len(normal_idx) // 3]]) if len(normal_idx) else 0.0
    event3_onset = onset(3)
    event6_onset = onset(6)
    if event6_onset is not None:
        event3_onset = min(x for x in (event3_onset, event6_onset) if x is not None)
    return {
        0: RawEventWindow(0, normal_onset, None, None, 0.0),
        1: RawEventWindow(1, onset(1), 39, None, 0.10),
        2: RawEventWindow(2, onset(2), "23-24", None, 0.0),
        3: RawEventWindow(3, event3_onset, 30, None, 0.0),
        4: RawEventWindow(4, onset(4), 7, None, 0.0),
        5: RawEventWindow(5, onset(5), None, 29, first_segment_duration(5)),
        6: RawEventWindow(6, event6_onset, 30, 29, first_segment_duration(6)),
        7: RawEventWindow(7, onset(7), None, 2, first_segment_duration(7)),
        8: RawEventWindow(8, None, 30, 39, 0.8),
    }


def three_phase_power(frame: pd.DataFrame, bus: int) -> np.ndarray:
    apparent = np.zeros(len(frame), dtype=complex)
    for phase in "ABC":
        v_mag = pd.to_numeric(frame[f"BUS{bus}_V{phase}_MAG"], errors="coerce").to_numpy(float)
        v_ang = pd.to_numeric(frame[f"BUS{bus}_V{phase}_ANG"], errors="coerce").to_numpy(float)
        i_mag = pd.to_numeric(frame[f"BUS{bus}_I{phase}_MAG"], errors="coerce").to_numpy(float)
        i_ang = pd.to_numeric(frame[f"BUS{bus}_I{phase}_ANG"], errors="coerce").to_numpy(float)
        voltage = v_mag * np.exp(1j * np.deg2rad(v_ang))
        current = i_mag * np.exp(1j * np.deg2rad(i_ang))
        apparent += voltage * np.conj(current)
    return apparent / 1e6


def _interpolate_finite(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if np.sum(finite) < 2:
        return values.copy()
    out = values.copy()
    missing_inside = ~finite & (time_s >= time_s[finite][0]) & (time_s <= time_s[finite][-1])
    out[missing_inside] = np.interp(time_s[missing_inside], time_s[finite], values[finite])
    return out


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).median().to_numpy(float)


def robust_frequency_deviations(
    time_rel_s: np.ndarray,
    frequency_hz: np.ndarray,
    pre_window: tuple[float, float] = (-1.5, -0.2),
    smoothing_s: float = 0.30,
) -> tuple[np.ndarray, np.ndarray]:
    """Return filtered frequency and a consistent derivative in mHz units.

    The supplied ROCOF channels use visibly different device algorithms and
    scales.  We therefore apply the same short median plus Savitzky-Golay
    filter to every RAW and simulated frequency channel, then derive ROCOF
    from that common trace.  Missing samples remain missing in the result.
    """

    from scipy.signal import savgol_filter

    time_rel_s = np.asarray(time_rel_s, dtype=float)
    values = np.asarray(frequency_hz, dtype=float)
    finite = np.isfinite(values) & np.isfinite(time_rel_s)
    pre = finite & (time_rel_s >= pre_window[0]) & (time_rel_s <= pre_window[1])
    if np.sum(finite) < 7 or np.sum(pre) < 3:
        nan = np.full_like(values, np.nan)
        return nan, nan.copy()

    # Savitzky-Golay requires a finite working vector.  Endpoint extrapolation
    # is used only inside the filter; the original missing-data mask is restored
    # before returning, so missing PMU samples are never presented as observed.
    filled = np.interp(
        time_rel_s,
        time_rel_s[finite],
        values[finite],
        left=float(values[finite][0]),
        right=float(values[finite][-1]),
    )
    dt_values = np.diff(time_rel_s[np.isfinite(time_rel_s)])
    dt_values = dt_values[dt_values > 0]
    if len(dt_values) == 0:
        nan = np.full_like(values, np.nan)
        return nan, nan.copy()
    dt = float(np.median(dt_values))
    median_filtered = _smooth(filled, window=3)
    preferred = max(5, int(round(smoothing_s / dt)) | 1)
    maximum = len(values) if len(values) % 2 else len(values) - 1
    window = min(preferred, maximum)
    if window < 5:
        window = 5
    frequency_filtered = savgol_filter(
        median_filtered,
        window_length=window,
        polyorder=min(2, window - 2),
        mode="interp",
    )
    rocof_hz_s = savgol_filter(
        median_filtered,
        window_length=window,
        polyorder=min(2, window - 2),
        deriv=1,
        delta=dt,
        mode="interp",
    )
    baseline_frequency = float(np.nanmedian(frequency_filtered[pre]))
    baseline_rocof = float(np.nanmedian(rocof_hz_s[pre]))
    frequency_mhz = 1_000.0 * (frequency_filtered - baseline_frequency)
    rocof_mhz_s = 1_000.0 * (rocof_hz_s - baseline_rocof)
    frequency_mhz[~finite] = np.nan
    rocof_mhz_s[~finite] = np.nan
    return frequency_mhz, rocof_mhz_s


def transform_trace(
    time_rel_s: np.ndarray,
    values: np.ndarray,
    kind: str,
    pre_window: tuple[float, float] = (-1.5, -0.2),
) -> np.ndarray:
    """Convert a PMU trace to a comparable local transient deviation."""

    time_rel_s = np.asarray(time_rel_s, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    pre = finite & (time_rel_s >= pre_window[0]) & (time_rel_s <= pre_window[1])
    if np.sum(pre) < 3:
        return np.full_like(values, np.nan)

    if kind in {"V_ANG_DEG", "I_ANG_DEG"}:
        filled = _interpolate_finite(time_rel_s, values)
        valid = np.isfinite(filled)
        unwrapped = np.full_like(filled, np.nan)
        unwrapped[valid] = np.rad2deg(np.unwrap(np.deg2rad(filled[valid])))
        coefficients = np.polyfit(time_rel_s[pre], unwrapped[pre], 1)
        result = unwrapped - np.polyval(coefficients, time_rel_s)
        result -= float(np.nanmedian(result[pre]))
        result[~finite] = np.nan
        return _smooth(result)

    if kind == "FREQ_MHZ":
        frequency_mhz, _ = robust_frequency_deviations(
            time_rel_s,
            values,
            pre_window=pre_window,
        )
        return frequency_mhz

    baseline = float(np.nanmedian(values[pre]))
    if kind in {"V_MAG_PCT", "I_MAG_PCT"}:
        if abs(baseline) < 1e-12:
            return np.full_like(values, np.nan)
        result = 100.0 * (values / baseline - 1.0)
    elif kind == "ROCOF_MHZ_S":
        result = 1_000.0 * (values - baseline)
    elif kind in {"P_MW", "Q_MVAR"}:
        result = values - baseline
    else:
        raise KeyError(kind)
    return _smooth(result)


def transformed_event_frame(
    frame: pd.DataFrame,
    bus: int,
    onset_s: float,
    time_bounds: tuple[float, float] = (-2.0, 5.0),
) -> pd.DataFrame:
    time_rel = frame["TIMESTAMP"].to_numpy(dtype=float) - float(onset_s)
    select = (time_rel >= time_bounds[0]) & (time_rel <= time_bounds[1])
    time_rel = time_rel[select]
    apparent = three_phase_power(frame, bus)
    raw = {
        "V_MAG_PCT": frame[f"BUS{bus}_VA_MAG"].to_numpy(dtype=float)[select],
        "I_MAG_PCT": frame[f"BUS{bus}_IA_MAG"].to_numpy(dtype=float)[select],
        "V_ANG_DEG": frame[f"BUS{bus}_VA_ANG"].to_numpy(dtype=float)[select],
        "I_ANG_DEG": frame[f"BUS{bus}_IA_ANG"].to_numpy(dtype=float)[select],
        "P_MW": apparent.real[select],
        "Q_MVAR": apparent.imag[select],
    }
    output = {"time_s": time_rel}
    for kind, values in raw.items():
        output[kind] = transform_trace(time_rel, values, kind)
    output["FREQ_MHZ"], output["ROCOF_MHZ_S"] = robust_frequency_deviations(
        time_rel,
        frame[f"BUS{bus}_Freq"].to_numpy(dtype=float)[select],
    )
    if "DATA_PRESENT" in frame:
        output["data_present"] = frame["DATA_PRESENT"].to_numpy(dtype=float)[select]
    return pd.DataFrame(output)


def resample_transformed(frame: pd.DataFrame, grid: np.ndarray) -> dict[str, np.ndarray]:
    time_s = frame["time_s"].to_numpy(float)
    output: dict[str, np.ndarray] = {}
    for signal in SIGNALS:
        values = frame[signal].to_numpy(float)
        finite = np.isfinite(values)
        if np.sum(finite) < 2:
            output[signal] = np.full(len(grid), np.nan)
            continue
        interpolated = np.interp(grid, time_s[finite], values[finite])
        interpolated[(grid < time_s[finite][0]) | (grid > time_s[finite][-1])] = np.nan
        if "data_present" in frame:
            present = np.interp(grid, time_s, frame["data_present"].to_numpy(float), left=0.0, right=0.0)
            interpolated[present < 0.5] = np.nan
        output[signal] = interpolated
    return output


def comparison_cube(
    frames: dict[int, pd.DataFrame],
    onset_s: float,
    grid: np.ndarray,
) -> dict[str, np.ndarray]:
    by_signal = {signal: [] for signal in SIGNALS}
    for bus in DEFAULT_PMU_BUSES:
        transformed = transformed_event_frame(frames[bus], bus, onset_s)
        resampled = resample_transformed(transformed, grid)
        for signal in SIGNALS:
            by_signal[signal].append(resampled[signal])
    return {signal: np.asarray(rows, dtype=float) for signal, rows in by_signal.items()}


def raw_baseline_levels(
    frames: dict[int, pd.DataFrame],
    onset_s: float,
) -> pd.DataFrame:
    rows = []
    for bus, frame in frames.items():
        time_s = frame["TIMESTAMP"].to_numpy(float)
        pre = (time_s >= onset_s - 1.5) & (time_s <= onset_s - 0.2)
        power = three_phase_power(frame, bus)
        rows.append(
            {
                "bus": bus,
                "voltage_kv_phase": float(np.nanmedian(frame.loc[pre, f"BUS{bus}_VA_MAG"]) / 1e3),
                "current_a": float(np.nanmedian(frame.loc[pre, f"BUS{bus}_IA_MAG"])),
                "frequency_hz": float(np.nanmedian(frame.loc[pre, f"BUS{bus}_Freq"])),
                "p_mw": float(np.nanmedian(power.real[pre])),
                "q_mvar": float(np.nanmedian(power.imag[pre])),
            }
        )
    return pd.DataFrame(rows)


def _dataset_cube(dataset: ScenarioDataset, grid: np.ndarray) -> dict[str, np.ndarray]:
    positions = {int(bus): idx for idx, bus in enumerate(dataset.bus_ids)}
    time_rel = dataset.time_s - dataset.spec.event_start_s
    output = {
        signal: []
        for signal in ("V_MAG_PCT", "V_ANG_DEG", "FREQ_MHZ", "ROCOF_MHZ_S")
    }
    for bus in DEFAULT_PMU_BUSES:
        idx = positions[bus]
        raw_values = {
            "V_MAG_PCT": dataset.voltage_pu[:, idx],
            "V_ANG_DEG": np.rad2deg(dataset.angle_rad[:, idx]),
        }
        for signal, values in raw_values.items():
            transformed = transform_trace(time_rel, values, signal)
            finite = np.isfinite(transformed)
            output[signal].append(np.interp(grid, time_rel[finite], transformed[finite]))
        frequency, rocof = robust_frequency_deviations(
            time_rel,
            dataset.frequency_hz[:, idx],
        )
        for signal, values in (("FREQ_MHZ", frequency), ("ROCOF_MHZ_S", rocof)):
            finite = np.isfinite(values)
            output[signal].append(np.interp(grid, time_rel[finite], values[finite]))
    return {signal: np.asarray(rows) for signal, rows in output.items()}


def normalized_fit_score(
    raw_cube: dict[str, np.ndarray],
    simulated_cube: dict[str, np.ndarray],
    grid: np.ndarray,
    signals: Iterable[str],
    bus_indices: Iterable[int] | None = None,
    signal_weights: dict[str, float] | None = None,
    huber_delta: float = 1.5,
) -> float:
    """Robust normalized quasi-likelihood for transient calibration."""

    event = grid >= 0.0
    floors = {
        "V_MAG_PCT": 0.5,
        "V_ANG_DEG": 0.2,
        "FREQ_MHZ": 10.0,
        "ROCOF_MHZ_S": 20.0,
    }
    selected = None if bus_indices is None else np.asarray(tuple(bus_indices), dtype=int)
    scores: list[float] = []
    weights: list[float] = []
    for signal in signals:
        raw = raw_cube[signal]
        sim = simulated_cube[signal]
        if selected is not None:
            raw = raw[selected]
            sim = sim[selected]
        raw = raw[:, event]
        sim = sim[:, event]
        valid = np.isfinite(raw) & np.isfinite(sim)
        if np.sum(valid) < 10:
            continue
        scale = max(float(np.nanpercentile(np.abs(raw[valid]), 90)), floors[signal])
        residual = (raw[valid] - sim[valid]) / scale
        absolute = np.abs(residual)
        loss = np.where(
            absolute <= huber_delta,
            0.5 * residual**2,
            huber_delta * (absolute - 0.5 * huber_delta),
        )
        scores.append(float(np.sqrt(2.0 * np.mean(loss))))
        weights.append(float((signal_weights or {}).get(signal, 1.0)))
    return float(np.average(scores, weights=weights)) if scores else float("inf")


def best_event_alignment(
    raw_frames: dict[int, pd.DataFrame],
    onset_s: float,
    simulated_cube: dict[str, np.ndarray],
    grid: np.ndarray,
    signals: Iterable[str],
    bus_indices: Iterable[int] | None = None,
    candidate_shifts_s: Iterable[float] | None = None,
) -> tuple[float, float, dict[str, np.ndarray]]:
    """Find the RAW event-marker correction that maximizes robust agreement."""

    if candidate_shifts_s is None:
        candidate_shifts_s = np.arange(-0.20, 0.2001, 1.0 / 30.0)
    best = (float("inf"), 0.0, None)
    weights = {
        "V_MAG_PCT": 0.38,
        "V_ANG_DEG": 0.30,
        "FREQ_MHZ": 0.22,
        "ROCOF_MHZ_S": 0.10,
    }
    for shift in candidate_shifts_s:
        raw_cube = comparison_cube(raw_frames, float(onset_s) + float(shift), grid)
        score = normalized_fit_score(
            raw_cube,
            simulated_cube,
            grid,
            signals,
            bus_indices=bus_indices,
            signal_weights=weights,
        )
        if score < best[0]:
            best = (score, float(shift), raw_cube)
    return best[1], best[0], best[2]


def scenario_for_event(
    event_type: int,
    parameters: TunedAndesParameters,
    window: RawEventWindow,
    scenario_id: str,
    seed: int,
    simulation_end_s: float = 7.0,
    event_start_s: float = 2.0,
) -> ScenarioSpec:
    severity = {
        1: parameters.fault_severity,
        3: parameters.generation_severity,
        4: parameters.load_severity,
        6: parameters.generation_severity,
        8: parameters.generation_severity,
    }.get(event_type, 0.10)
    duration = window.duration_s
    if event_type == 1:
        duration = parameters.fault_duration_s
    elif event_type == 2:
        duration = parameters.line_outage_duration_s
    return ScenarioSpec(
        scenario_id=scenario_id,
        event_type=event_type,
        target=parameters.fault_bus if event_type == 1 else window.target,
        cyber_target=window.cyber_target,
        severity=severity,
        reactive_severity=parameters.load_q_severity if event_type == 4 else None,
        generation_mode=parameters.generation_mode if event_type in {3, 6} else "torque_step",
        excitation_severity=parameters.excitation_severity if event_type in {3, 6} else 0.0,
        event_start_s=event_start_s,
        event_duration_s=duration,
        simulation_end_s=simulation_end_s,
        inertia_scale=parameters.inertia_scale,
        inertia_group_30_32=parameters.inertia_group_30_32,
        inertia_group_33_36=parameters.inertia_group_33_36,
        inertia_group_37_39=parameters.inertia_group_37_39,
        damping=parameters.damping,
        governor_droop_scale=parameters.governor_droop_scale,
        fault_resistance_pu=parameters.fault_resistance_pu if event_type == 1 else None,
        fault_reactance_pu=parameters.fault_reactance_pu if event_type == 1 else None,
        fault_post_trip=parameters.fault_post_trip if event_type == 1 else None,
        seed=seed,
    )


def tune_andes_dynamics(
    raw_frames: dict[int, pd.DataFrame],
    windows: dict[int, RawEventWindow],
    calibration: MeasurementCalibration,
    cache_path: str | Path | None = None,
    force: bool = False,
) -> tuple[TunedAndesParameters, pd.DataFrame]:
    """Fit a robust multi-event dynamic model with spatial PMU holdout.

    BUS2/5/6/10/19/22 form the fitting set. BUS29 and BUS39 are never used to
    select parameters, so their scores provide a sensor-location holdout. E2 is
    also evaluated once before its reclose time is adapted. E6 reuses the E3
    physical occurrence and is therefore only an availability-transfer check,
    not an independent disturbance validation.
    """

    if cache_path and Path(cache_path).exists() and not force:
        params = TunedAndesParameters.from_json(cache_path)
        trace_path = Path(cache_path).with_name("tuning_trace.csv")
        trace = pd.read_csv(trace_path) if trace_path.exists() else pd.DataFrame()
        return params, trace

    grid = np.arange(-1.5, 3.0 + 1e-9, 1.0 / 30.0)
    shifts = tuple(float(x) for x in np.arange(-0.20, 0.2001, 1.0 / 30.0))
    fit_signals = ("V_MAG_PCT", "V_ANG_DEG", "FREQ_MHZ", "ROCOF_MHZ_S")
    train_indices = tuple(range(6))
    holdout_indices = (6, 7)
    raw_shift_cubes = {
        event: {
            shift: comparison_cube(
                raw_frames,
                float(windows[event].onset_s) + shift,
                grid,
            )
            for shift in shifts
        }
        for event in (1, 2, 3, 4, 6)
    }
    weights = {
        "V_MAG_PCT": 0.38,
        "V_ANG_DEG": 0.30,
        "FREQ_MHZ": 0.22,
        "ROCOF_MHZ_S": 0.10,
    }
    rows: list[dict[str, object]] = []

    baseline_params = TunedAndesParameters(
        inertia_scale=0.75,
        fault_bus=39,
        fault_resistance_pu=0.0,
        fault_reactance_pu=0.018,
        fault_post_trip=None,
        fault_duration_s=0.10,
        line_outage_duration_s=0.40,
        generation_severity=0.90,
        generation_mode="torque_step",
        excitation_severity=0.0,
        load_severity=0.30,
        load_q_severity=0.0,
    )

    def score_cube(
        raw_cube: dict[str, np.ndarray],
        simulated_cube: dict[str, np.ndarray],
        bus_indices: Iterable[int] | None,
    ) -> float:
        return normalized_fit_score(
            raw_cube,
            simulated_cube,
            grid,
            fit_signals,
            bus_indices=bus_indices,
            signal_weights=weights,
        )

    def evaluate_candidate(
        event: int,
        candidate: TunedAndesParameters,
        stage: str,
        *,
        alignment_override: float | None = None,
        record: bool = True,
        extra: dict[str, object] | None = None,
    ) -> tuple[float, float, float, float]:
        spec = scenario_for_event(
            event,
            candidate,
            windows[event],
            f"{stage}_event{event}",
            seed=202_600 + event,
            simulation_end_s=5.0,
            event_start_s=1.5,
        )
        try:
            simulated = _dataset_cube(simulate_scenario(spec, calibration), grid)
            if alignment_override is None:
                alignment, train_score, _ = min(
                    (
                        (
                            shift,
                            score_cube(raw_shift_cubes[event][shift], simulated, train_indices),
                            score_cube(raw_shift_cubes[event][shift], simulated, train_indices)
                            + 0.015 * (shift / 0.10) ** 2,
                        )
                        for shift in shifts
                    ),
                    key=lambda item: item[2],
                )
            else:
                alignment = min(shifts, key=lambda shift: abs(shift - alignment_override))
                train_score = score_cube(
                    raw_shift_cubes[event][alignment], simulated, train_indices
                )
            raw_cube = raw_shift_cubes[event][alignment]
            all_score = score_cube(raw_cube, simulated, None)
            holdout_score = score_cube(raw_cube, simulated, holdout_indices)
        except (RuntimeError, ValueError, FloatingPointError):
            alignment = float(alignment_override or 0.0)
            train_score = all_score = holdout_score = float("inf")
        if record:
            rows.append(
                {
                    "stage": stage,
                    "event_type": event,
                    "score": train_score,
                    "train_score": train_score,
                    "holdout_score": holdout_score,
                    "all_pmu_score": all_score,
                    "alignment_s": alignment,
                    "inertia_scale": candidate.inertia_scale,
                    "inertia_group_30_32": candidate.inertia_group_30_32,
                    "inertia_group_33_36": candidate.inertia_group_33_36,
                    "inertia_group_37_39": candidate.inertia_group_37_39,
                    "damping": candidate.damping,
                    "governor_droop_scale": candidate.governor_droop_scale,
                    "generation_mode": candidate.generation_mode,
                    "excitation_severity": candidate.excitation_severity,
                    **(extra or {}),
                }
            )
        return train_score, holdout_score, all_score, alignment

    for event in (1, 2, 3, 4, 6):
        evaluate_candidate(event, baseline_params, "baseline_robust")

    best: tuple[float, TunedAndesParameters, float] = (
        float("inf"), baseline_params, 0.0
    )
    for inertia in (0.50, 0.75, 1.00, 1.25):
        for damping in (0.0, 0.25, 1.0):
            for severity in (0.70, 0.85, 0.95):
                candidate = replace(
                    baseline_params,
                    inertia_scale=inertia,
                    inertia_group_30_32=1.0,
                    inertia_group_33_36=1.0,
                    inertia_group_37_39=1.0,
                    damping=damping,
                    generation_severity=severity,
                )
                score, _, _, alignment = evaluate_candidate(
                    3,
                    candidate,
                    "event3_global_dynamics",
                    extra={"severity": severity},
                )
                objective = score + 0.01 * ((inertia - 1.0) / 0.5) ** 2 + 0.005 * damping**2
                if objective < best[0]:
                    best = (objective, candidate, alignment)
    params = best[1]
    params.event3_alignment_s = best[2]

    # Test the event mechanism itself. A torque step preserves electrical
    # connection and AVR voltage support, whereas a generator trip removes the
    # source instantaneously.  RAW shows a localized step-like voltage loss at
    # BUS2, so both hypotheses must be compared before fine tuning inertia.
    best_generation = best
    for inertia in (0.50, 0.75, 1.00, 1.25):
        for damping in (0.0, 0.25, 1.0):
            candidate = replace(
                params,
                inertia_scale=inertia,
                damping=damping,
                generation_mode="generator_trip",
                excitation_severity=0.0,
            )
            score, _, _, alignment = evaluate_candidate(
                3,
                candidate,
                "event3_generation_hypothesis",
                extra={"hypothesis": "generator_trip"},
            )
            objective = score + 0.01 * ((inertia - 1.0) / 0.5) ** 2 + 0.005 * damping**2
            if objective < best_generation[0]:
                best_generation = (objective, candidate, alignment)
    for excitation in (0.03, 0.05, 0.08):
        candidate = replace(
            params,
            generation_mode="torque_excitation",
            excitation_severity=excitation,
        )
        score, _, _, alignment = evaluate_candidate(
            3,
            candidate,
            "event3_generation_hypothesis",
            extra={"hypothesis": "torque_excitation"},
        )
        if score < best_generation[0]:
            best_generation = (score, candidate, alignment)
    params = best_generation[1]
    params.event3_alignment_s = best_generation[2]

    best_fault: tuple[float, TunedAndesParameters, float] = (
        float("inf"), params, 0.0
    )
    for resistance in (0.0, 0.005, 0.020):
        for reactance in (0.001, 0.010, 0.030, 0.060):
            for duration in (2.0 / 30.0, 0.10, 4.0 / 30.0):
                candidate = replace(
                    params,
                    fault_bus=39,
                    fault_resistance_pu=resistance,
                    fault_reactance_pu=reactance,
                    fault_duration_s=duration,
                    fault_post_trip=None,
                )
                score, _, _, alignment = evaluate_candidate(
                    1,
                    candidate,
                    "event1_fault_impedance",
                    extra={
                        "fault_bus": 39,
                        "fault_resistance_pu": resistance,
                        "fault_reactance_pu": reactance,
                        "duration_s": duration,
                        "fault_post_trip": None,
                    },
                )
                objective = score + 0.002 * resistance**2 + 0.002 * reactance**2
                if objective < best_fault[0]:
                    best_fault = (objective, candidate, alignment)

    for bus in (1, 9):
        candidate = replace(best_fault[1], fault_bus=bus)
        score, _, _, alignment = evaluate_candidate(
            1,
            candidate,
            "event1_fault_location",
            extra={"fault_bus": bus},
        )
        if score < best_fault[0]:
            best_fault = (score, candidate, alignment)
    for post_trip in ("1-39", "9-39"):
        candidate = replace(best_fault[1], fault_post_trip=post_trip)
        score, _, _, alignment = evaluate_candidate(
            1,
            candidate,
            "event1_post_fault_topology",
            extra={"fault_post_trip": post_trip},
        )
        if score < best_fault[0]:
            best_fault = (score, candidate, alignment)
    params = best_fault[1]
    params.event1_alignment_s = best_fault[2]

    best_load: tuple[float, TunedAndesParameters, float] = (
        float("inf"), params, 0.0
    )
    for p_severity in (0.20, 0.30, 0.40, 0.50):
        for q_severity in (-0.10, 0.00, 0.10, 0.20):
            candidate = replace(
                params,
                load_severity=p_severity,
                load_q_severity=q_severity,
            )
            score, _, _, alignment = evaluate_candidate(
                4,
                candidate,
                "event4_load_pq",
                extra={
                    "severity": p_severity,
                    "reactive_severity": q_severity,
                },
            )
            if score < best_load[0]:
                best_load = (score, candidate, alignment)
    params = best_load[1]
    params.event4_alignment_s = best_load[2]

    # Coordinate-search common and regional inertia against all calibration events.
    common_candidates = [params]
    for scale in (0.85, 1.15):
        common_candidates.append(replace(params, inertia_scale=params.inertia_scale * scale))
    for damping in (0.25, 0.50):
        common_candidates.append(replace(params, damping=damping))
    for field_name in (
        "inertia_group_30_32",
        "inertia_group_33_36",
        "inertia_group_37_39",
    ):
        for scale in (0.85, 1.15):
            common_candidates.append(replace(params, **{field_name: scale}))
    for droop_scale in (2.0, 4.0, 8.0):
        for inertia in (0.30, 0.50, 0.75):
            common_candidates.append(
                replace(
                    params,
                    inertia_scale=inertia,
                    governor_droop_scale=droop_scale,
                )
            )

    joint_best: tuple[float, TunedAndesParameters, dict[int, float], dict[int, float]] = (
        float("inf"), params, {}, {}
    )
    for candidate_index, candidate in enumerate(common_candidates):
        event_scores: dict[int, float] = {}
        event_alignments: dict[int, float] = {}
        for event in (1, 3, 4):
            score, _, _, alignment = evaluate_candidate(
                event,
                candidate,
                "joint_multi_event",
                extra={"candidate": candidate_index},
            )
            event_scores[event] = score
            event_alignments[event] = alignment
        regularization = 0.015 * (
            (candidate.inertia_group_30_32 - 1.0) ** 2
            + (candidate.inertia_group_33_36 - 1.0) ** 2
            + (candidate.inertia_group_37_39 - 1.0) ** 2
        ) + 0.005 * candidate.damping**2 + 0.003 * np.log2(candidate.governor_droop_scale) ** 2
        objective = float(np.mean(list(event_scores.values()))) + regularization
        if objective < joint_best[0]:
            joint_best = (objective, candidate, event_scores, event_alignments)
    params = joint_best[1]
    params.event1_alignment_s = joint_best[3][1]
    params.event3_alignment_s = joint_best[3][3]
    params.event4_alignment_s = joint_best[3][4]
    params.event1_fit_score = joint_best[2][1]
    params.event3_fit_score = joint_best[2][3]
    params.event4_fit_score = joint_best[2][4]

    # E2 transfer prediction before looking at its reclose time.
    transfer_alignment = float(
        np.median(
            [
                params.event1_alignment_s,
                params.event3_alignment_s,
                params.event4_alignment_s,
            ]
        )
    )
    blind_e2 = replace(params, line_outage_duration_s=0.0)
    blind_score, _, _, _ = evaluate_candidate(
        2,
        blind_e2,
        "event2_blind_transfer",
        alignment_override=transfer_alignment,
        extra={"duration_s": 0.0},
    )
    params.event2_blind_score = blind_score

    best_line: tuple[float, TunedAndesParameters, float] = (
        float("inf"), params, 0.0
    )
    for duration in (0.10, 0.20, 0.40, 0.80, 1.20):
        candidate = replace(params, line_outage_duration_s=duration)
        score, _, _, alignment = evaluate_candidate(
            2,
            candidate,
            "event2_adapted_reclose",
            extra={"duration_s": duration},
        )
        if score < best_line[0]:
            best_line = (score, candidate, alignment)
    params = best_line[1]
    params.event2_fit_score = best_line[0]
    params.event2_alignment_s = best_line[2]

    params.event6_alignment_s = params.event3_alignment_s
    transfer_score, _, _, _ = evaluate_candidate(
        6,
        params,
        "event6_availability_transfer",
        alignment_override=params.event6_alignment_s,
    )
    params.event6_transfer_score = transfer_score

    final_train = []
    final_holdout = []
    for event in (1, 3, 4):
        train, holdout, _, _ = evaluate_candidate(
            event,
            params,
            "final_validation",
            alignment_override=params.alignment_for_event(event),
        )
        final_train.append(train)
        final_holdout.append(holdout)
    params.train_objective_score = float(np.mean(final_train))
    params.holdout_objective_score = float(np.mean(final_holdout))

    trace = pd.DataFrame(rows)
    if cache_path:
        params.to_json(cache_path)
        trace.to_csv(Path(cache_path).with_name("tuning_trace.csv"), index=False)
    return params, trace


def comparison_metrics(
    raw_cube: dict[str, np.ndarray] | None,
    simulated_cube: dict[str, np.ndarray],
    grid: np.ndarray,
    event_type: int,
) -> pd.DataFrame:
    rows = []
    for signal in SIGNALS:
        if raw_cube is None:
            rows.append(
                {"event_type": event_type, "signal": signal, "rmse": np.nan, "mae": np.nan, "correlation": np.nan}
            )
            continue
        raw = raw_cube[signal]
        sim = simulated_cube[signal]
        valid = np.isfinite(raw) & np.isfinite(sim) & (grid[None, :] >= 0.0)
        if np.sum(valid) < 3:
            rmse = mae = corr = np.nan
        else:
            error = sim[valid] - raw[valid]
            rmse = float(np.sqrt(np.mean(error**2)))
            mae = float(np.mean(np.abs(error)))
            raw_values = raw[valid]
            sim_values = sim[valid]
            corr = (
                float(np.corrcoef(raw_values, sim_values)[0, 1])
                if np.std(raw_values) > 1e-12 and np.std(sim_values) > 1e-12
                else np.nan
            )
        rows.append({"event_type": event_type, "signal": signal, "rmse": rmse, "mae": mae, "correlation": corr})
    return pd.DataFrame(rows)
