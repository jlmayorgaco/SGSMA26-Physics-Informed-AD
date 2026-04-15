from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.data_loader import load_synth_bus_data, load_synth_metadata


VoltageRepresentation = Literal["A", "B", "C", "positive_sequence"]


@dataclass(frozen=True)
class SwingYBusConfig:
    representation: str = "positive_sequence"
    ridge: float = 1e-8

    # Reduced swing-like dynamics
    inertia: float = 4.0
    damping: float = 1.2

    # Dynamic correction gains
    angle_gain: float = 0.10
    mag_gain_freq: float = 2.5
    mag_gain_rocof: float = 0.08

    # Safety clamps
    min_mag_scale: float = 0.85
    max_mag_scale: float = 1.15


@dataclass(frozen=True)
class SwingYBusSummary:
    scenario_id: str
    representation: str
    n_total_buses: int
    n_pmu_buses: int
    n_hidden_buses: int
    n_timestamps: int
    mag_rmse_overall: float
    mag_rmse_normal: float
    mag_rmse_event: float
    angle_mae_deg_overall: float
    angle_mae_deg_normal: float
    angle_mae_deg_event: float


def _phasor_from_mag_angle(magnitude: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    return magnitude * np.exp(1j * np.deg2rad(angle_deg))


def _wrap_angle_deg(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def _positive_sequence_from_abc(
    va_mag: np.ndarray,
    va_ang: np.ndarray,
    vb_mag: np.ndarray,
    vb_ang: np.ndarray,
    vc_mag: np.ndarray,
    vc_ang: np.ndarray,
) -> np.ndarray:
    va = _phasor_from_mag_angle(va_mag, va_ang)
    vb = _phasor_from_mag_angle(vb_mag, vb_ang)
    vc = _phasor_from_mag_angle(vc_mag, vc_ang)

    a = np.exp(1j * 2.0 * np.pi / 3.0)
    return (va + a * vb + (a**2) * vc) / 3.0


def _extract_bus_complex_voltage(
    df_bus: pd.DataFrame,
    representation: VoltageRepresentation,
) -> np.ndarray:
    if representation == "A":
        return _phasor_from_mag_angle(df_bus["VA_MAG"].to_numpy(), df_bus["VA_ANG"].to_numpy())

    if representation == "B":
        return _phasor_from_mag_angle(df_bus["VB_MAG"].to_numpy(), df_bus["VB_ANG"].to_numpy())

    if representation == "C":
        return _phasor_from_mag_angle(df_bus["VC_MAG"].to_numpy(), df_bus["VC_ANG"].to_numpy())

    if representation == "positive_sequence":
        return _positive_sequence_from_abc(
            df_bus["VA_MAG"].to_numpy(),
            df_bus["VA_ANG"].to_numpy(),
            df_bus["VB_MAG"].to_numpy(),
            df_bus["VB_ANG"].to_numpy(),
            df_bus["VC_MAG"].to_numpy(),
            df_bus["VC_ANG"].to_numpy(),
        )

    raise ValueError(f"Unsupported representation: {representation}")


def _build_susceptance_matrix_from_metadata(
    metadata: dict,
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """
    Build a Laplacian/susceptance surrogate from branch reactances.

    This is not the full complex Ybus yet.
    It is the same structural model used in the static baseline.
    """
    buses = sorted(int(bus) for bus in metadata["output_buses"])
    bus_to_idx = {bus: idx for idx, bus in enumerate(buses)}

    n_buses = len(buses)
    B = np.zeros((n_buses, n_buses), dtype=float)

    branch_reactance = metadata["branch_reactance_pu"]
    branches = metadata["ieee39_branches"]

    for from_bus, to_bus in branches:
        key_forward = f"{from_bus}-{to_bus}"
        key_reverse = f"{to_bus}-{from_bus}"

        if key_forward in branch_reactance:
            x_pu = float(branch_reactance[key_forward])
        elif key_reverse in branch_reactance:
            x_pu = float(branch_reactance[key_reverse])
        else:
            raise KeyError(f"Missing reactance for branch ({from_bus}, {to_bus})")

        if np.isclose(x_pu, 0.0):
            raise ValueError(f"Zero reactance for branch ({from_bus}, {to_bus})")

        b_ij = 1.0 / x_pu

        i = bus_to_idx[int(from_bus)]
        j = bus_to_idx[int(to_bus)]

        B[i, i] += b_ij
        B[j, j] += b_ij
        B[i, j] -= b_ij
        B[j, i] -= b_ij

    return B, buses, bus_to_idx


def _load_bus_frames(
    scenario_id: str,
    representation: VoltageRepresentation,
) -> dict[int, pd.DataFrame]:
    metadata = load_synth_metadata(scenario_id=scenario_id)
    buses = sorted(int(bus) for bus in metadata["output_buses"])

    frames: dict[int, pd.DataFrame] = {}

    for bus_id in buses:
        df_bus = load_synth_bus_data(scenario_id=scenario_id, bus_id=bus_id)

        v_complex = _extract_bus_complex_voltage(df_bus, representation)

        frames[bus_id] = pd.DataFrame(
            {
                "TIMESTAMP": df_bus["TIMESTAMP"].to_numpy(),
                "Event": df_bus["Event"].to_numpy(),
                "Freq": df_bus["Freq"].to_numpy(),
                "ROCOF": df_bus["ROCOF"].to_numpy(),
                "V_complex": v_complex,
            }
        )

    return frames


def _build_static_hidden_mapping(
    B: np.ndarray,
    buses: list[int],
    pmu_buses: list[int],
    ridge: float,
) -> tuple[list[int], np.ndarray]:
    """
    Returns:
        hidden_buses
        G such that V_hidden_static = G @ V_pmu
    """
    bus_to_idx = {bus: idx for idx, bus in enumerate(buses)}

    pmu_idx = [bus_to_idx[bus] for bus in pmu_buses]
    hidden_buses = [bus for bus in buses if bus not in pmu_buses]
    hidden_idx = [bus_to_idx[bus] for bus in hidden_buses]

    B_hh = B[np.ix_(hidden_idx, hidden_idx)]
    B_hp = B[np.ix_(hidden_idx, pmu_idx)]

    B_hh_reg = B_hh + ridge * np.eye(B_hh.shape[0])
    G = -np.linalg.solve(B_hh_reg, B_hp)

    return hidden_buses, G


def _row_normalize_abs_weights(matrix: np.ndarray) -> np.ndarray:
    weights = np.abs(matrix).astype(float)
    row_sums = weights.sum(axis=1, keepdims=True)

    zero_rows = np.isclose(row_sums, 0.0).flatten()
    if np.any(zero_rows):
        weights[zero_rows, :] = 1.0
        row_sums = weights.sum(axis=1, keepdims=True)

    return weights / row_sums


def _estimate_hidden_with_swing_correction(
    v_pmu: np.ndarray,
    f_pmu: np.ndarray,
    rocof_pmu: np.ndarray,
    G: np.ndarray,
    dt: float,
    nominal_frequency_hz: float,
    config: SwingYBusConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core estimator.

    Static:
        V_hidden_static = G @ V_pmu

    Dynamic:
        local frequency / ROCOF from PMU-weighted mixtures
        reduced swing-like update on hidden frequency deviation
        accumulated phase correction from hidden local frequency
        magnitude correction from hidden local frequency + ROCOF
    """
    n_timestamps = v_pmu.shape[0]
    n_hidden = G.shape[0]

    # Static hidden voltage estimate
    v_hidden_static = (G @ v_pmu.T).T  # (T, H)

    # Local PMU mixing weights per hidden bus
    W = _row_normalize_abs_weights(G)  # (H, P)

    # Local hidden dynamic surrogates
    local_freq_dev = (W @ (f_pmu - nominal_frequency_hz).T).T  # (T, H)
    local_rocof = (W @ rocof_pmu.T).T  # (T, H)

    # Reduced swing-like states
    hidden_freq_dev_hat = np.zeros((n_timestamps, n_hidden), dtype=float)
    hidden_phase_deg = np.zeros((n_timestamps, n_hidden), dtype=float)

    alpha = config.damping / max(config.inertia, 1e-12)

    for t in range(1, n_timestamps):
        hidden_freq_dev_hat[t, :] = (
            hidden_freq_dev_hat[t - 1, :]
            + dt * (-alpha * hidden_freq_dev_hat[t - 1, :] + local_rocof[t - 1, :])
        )

        hidden_phase_deg[t, :] = (
            hidden_phase_deg[t - 1, :]
            + 360.0 * dt * hidden_freq_dev_hat[t - 1, :]
        )

    mag_scale = (
        1.0
        + config.mag_gain_freq * (hidden_freq_dev_hat / nominal_frequency_hz)
        + config.mag_gain_rocof * (local_rocof / nominal_frequency_hz)
    )
    mag_scale = np.clip(mag_scale, config.min_mag_scale, config.max_mag_scale)

    angle_corr_rad = np.deg2rad(config.angle_gain * hidden_phase_deg)

    v_hidden_hat = (
        mag_scale
        * np.abs(v_hidden_static)
        * np.exp(1j * (np.angle(v_hidden_static) + angle_corr_rad))
    )

    return v_hidden_hat, hidden_freq_dev_hat, hidden_phase_deg


def _summarize_results(
    results_df: pd.DataFrame,
    scenario_id: str,
    representation: str,
    n_total_buses: int,
    n_pmu_buses: int,
    n_hidden_buses: int,
    n_timestamps: int,
) -> SwingYBusSummary:
    def _mag_rmse(frame: pd.DataFrame) -> float:
        return float(np.sqrt(np.mean(np.square(frame["mag_error"].to_numpy()))))

    def _ang_mae(frame: pd.DataFrame) -> float:
        return float(np.mean(np.abs(frame["angle_error_deg"].to_numpy())))

    overall = results_df
    normal = results_df[results_df["Event"] == 0]
    event = results_df[results_df["Event"] != 0]

    return SwingYBusSummary(
        scenario_id=scenario_id,
        representation=representation,
        n_total_buses=n_total_buses,
        n_pmu_buses=n_pmu_buses,
        n_hidden_buses=n_hidden_buses,
        n_timestamps=n_timestamps,
        mag_rmse_overall=_mag_rmse(overall),
        mag_rmse_normal=_mag_rmse(normal),
        mag_rmse_event=_mag_rmse(event),
        angle_mae_deg_overall=_ang_mae(overall),
        angle_mae_deg_normal=_ang_mae(normal),
        angle_mae_deg_event=_ang_mae(event),
    )


def run_swing_ybus_estimator(
    scenario_id: str = "SIM_0001",
    config: SwingYBusConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, SwingYBusSummary]:
    """
    Reduced swing-informed Ybus estimator.

    Returns
    -------
    results_df
        Long dataframe with per-hidden-bus, per-timestamp estimates and errors.
    per_bus_summary_df
        Aggregated metrics per hidden bus.
    summary
        Global summary dataclass.
    """
    if config is None:
        config = SwingYBusConfig()

    metadata = load_synth_metadata(scenario_id=scenario_id)

    nominal_frequency_hz = float(metadata["nominal_frequency_hz"])
    pmu_buses = sorted(int(bus) for bus in metadata["pmu_buses"])
    all_buses = sorted(int(bus) for bus in metadata["output_buses"])

    bus_frames = _load_bus_frames(
        scenario_id=scenario_id,
        representation=config.representation,
    )

    base_bus = all_buses[0]
    timestamps = bus_frames[base_bus]["TIMESTAMP"].to_numpy()
    events = bus_frames[base_bus]["Event"].to_numpy()

    if len(timestamps) < 2:
        raise ValueError("Need at least two timestamps to run the dynamic estimator")

    dt = float(np.median(np.diff(timestamps)))

    B, buses, _ = _build_susceptance_matrix_from_metadata(metadata)
    hidden_buses, G = _build_static_hidden_mapping(
        B=B,
        buses=buses,
        pmu_buses=pmu_buses,
        ridge=config.ridge,
    )

    v_pmu = np.column_stack(
        [bus_frames[bus_id]["V_complex"].to_numpy() for bus_id in pmu_buses]
    )
    f_pmu = np.column_stack(
        [bus_frames[bus_id]["Freq"].to_numpy() for bus_id in pmu_buses]
    )
    rocof_pmu = np.column_stack(
        [bus_frames[bus_id]["ROCOF"].to_numpy() for bus_id in pmu_buses]
    )

    v_hidden_hat, hidden_freq_dev_hat, hidden_phase_deg = _estimate_hidden_with_swing_correction(
        v_pmu=v_pmu,
        f_pmu=f_pmu,
        rocof_pmu=rocof_pmu,
        G=G,
        dt=dt,
        nominal_frequency_hz=nominal_frequency_hz,
        config=config,
    )

    records: list[dict] = []

    for hidden_idx, bus_id in enumerate(hidden_buses):
        v_true = bus_frames[bus_id]["V_complex"].to_numpy()
        v_hat = v_hidden_hat[:, hidden_idx]

        mag_true = np.abs(v_true)
        mag_hat = np.abs(v_hat)

        ang_true = np.rad2deg(np.angle(v_true))
        ang_hat = np.rad2deg(np.angle(v_hat))
        ang_err = _wrap_angle_deg(ang_hat - ang_true)

        for t_idx in range(len(timestamps)):
            records.append(
                {
                    "TIMESTAMP": float(timestamps[t_idx]),
                    "Event": int(events[t_idx]),
                    "bus_id": int(bus_id),
                    "mag_true": float(mag_true[t_idx]),
                    "mag_hat": float(mag_hat[t_idx]),
                    "mag_error": float(mag_hat[t_idx] - mag_true[t_idx]),
                    "mag_abs_error": float(np.abs(mag_hat[t_idx] - mag_true[t_idx])),
                    "angle_true_deg": float(ang_true[t_idx]),
                    "angle_hat_deg": float(ang_hat[t_idx]),
                    "angle_error_deg": float(ang_err[t_idx]),
                    "angle_abs_error_deg": float(np.abs(ang_err[t_idx])),
                    "hidden_freq_dev_hat_hz": float(hidden_freq_dev_hat[t_idx, hidden_idx]),
                    "hidden_phase_corr_deg": float(hidden_phase_deg[t_idx, hidden_idx]),
                }
            )

    results_df = pd.DataFrame.from_records(records)

    per_bus_summary_df = (
        results_df.groupby("bus_id", as_index=False)
        .agg(
            mag_rmse=("mag_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            mag_mae=("mag_abs_error", "mean"),
            angle_mae_deg=("angle_abs_error_deg", "mean"),
        )
        .sort_values(["mag_rmse", "angle_mae_deg"], ascending=[True, True])
        .reset_index(drop=True)
    )

    summary = _summarize_results(
        results_df=results_df,
        scenario_id=scenario_id,
        representation=config.representation,
        n_total_buses=len(all_buses),
        n_pmu_buses=len(pmu_buses),
        n_hidden_buses=len(hidden_buses),
        n_timestamps=len(timestamps),
    )

    return results_df, per_bus_summary_df, summary


def summary_to_dict(summary: SwingYBusSummary) -> dict:
    return asdict(summary)