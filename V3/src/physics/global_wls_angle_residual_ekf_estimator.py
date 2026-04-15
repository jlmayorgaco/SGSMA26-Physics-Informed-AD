from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.utils.data_loader import load_synth_bus_data, load_synth_metadata
from src.utils.per_unit import (
    MeasurementBaseMode,
    complex_voltage_to_pu,
    magnitude_from_pu,
)


VoltageRepresentation = Literal["A", "B", "C", "positive_sequence"]


@dataclass(frozen=True)
class GlobalWLSAngleResidualEKFConfig:
    representation: str = "positive_sequence"
    n_modes: int = 8

    # Per-unit measurement base convention.
    # "nominal"      — uses nominal bus base kV from the network model.
    # "override_345" — overrides buses 20 and 30-38 to 345 kV base, testing
    #                  whether ANDES reports those voltages at backbone scale.
    measurement_base_mode: MeasurementBaseMode = "nominal"

    # Static Ybus prior for hidden-bus complex voltage
    static_ridge: float = 1e-8

    # Global WLS prior for angle field
    obs_weight_ang: float = 2.0e4
    smooth_weight_ang: float = 10.0
    temporal_ref_weight_ang: float = 150.0
    ridge_prior_ang: float = 1.0e-6

    # Dynamic residual EKF over angle only
    coherent_angle_coupling: float = 0.15
    rocof_decay: float = 0.96

    q_ang: float = 5.0e-5
    q_freq: float = 2.0e-5
    q_rocof: float = 5.0e-4

    p0_ang: float = 5.0e-4
    p0_freq: float = 1.0e-4
    p0_rocof: float = 1.0e-3

    r_ang_deg: float = 0.03
    r_freq: float = 2.0e-3
    r_rocof: float = 2.0e-2


@dataclass(frozen=True)
class GlobalWLSAngleResidualEKFSummary:
    scenario_id: str
    representation: str
    n_total_buses: int
    n_pmu_buses: int
    n_hidden_buses: int
    n_timestamps: int
    n_modes: int
    mag_rmse_overall: float
    mag_rmse_normal: float
    mag_rmse_event: float
    angle_mae_deg_overall: float
    angle_mae_deg_normal: float
    angle_mae_deg_event: float


def _phasor_from_mag_angle(magnitude: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    return magnitude * np.exp(1j * np.deg2rad(angle_deg))


def _positive_sequence_from_abc(
    xa_mag: np.ndarray,
    xa_ang: np.ndarray,
    xb_mag: np.ndarray,
    xb_ang: np.ndarray,
    xc_mag: np.ndarray,
    xc_ang: np.ndarray,
) -> np.ndarray:
    xa = _phasor_from_mag_angle(xa_mag, xa_ang)
    xb = _phasor_from_mag_angle(xb_mag, xb_ang)
    xc = _phasor_from_mag_angle(xc_mag, xc_ang)

    a = np.exp(1j * 2.0 * np.pi / 3.0)
    return (xa + a * xb + (a**2) * xc) / 3.0


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


def _wrap_angle_deg(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def _build_susceptance_laplacian(
    metadata: dict[str, Any],
) -> tuple[np.ndarray, list[int], dict[int, int]]:
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
            raise ValueError(f"Zero reactance found for branch ({from_bus}, {to_bus})")

        b_ij = 1.0 / x_pu

        i = bus_to_idx[int(from_bus)]
        j = bus_to_idx[int(to_bus)]

        B[i, i] += b_ij
        B[j, j] += b_ij
        B[i, j] -= b_ij
        B[j, i] -= b_ij

    return B, buses, bus_to_idx


def _build_graph_basis(B: np.ndarray, n_modes: int) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    order = np.argsort(eigenvalues)
    U = eigenvectors[:, order][:, : min(n_modes, eigenvectors.shape[1])].copy()

    for col_idx in range(U.shape[1]):
        if np.sum(U[:, col_idx]) < 0.0:
            U[:, col_idx] *= -1.0

    return U


def _load_bus_frames(
    scenario_id: str,
    representation: VoltageRepresentation,
    measurement_base_mode: MeasurementBaseMode = "nominal",
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
                "DATA_PRESENT": df_bus["DATA_PRESENT"].to_numpy(),
                "Freq": df_bus["Freq"].to_numpy(),
                "ROCOF": df_bus["ROCOF"].to_numpy(),
                "V_complex_actual": v_complex,
                "V_complex_pu": complex_voltage_to_pu(v_complex, bus_id, mode=measurement_base_mode),
            }
        )

    return frames


def _complete_pmu_snapshot(
    pmu_complex_obs: np.ndarray,
    pmu_present: np.ndarray,
    fallback_complex: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fill missing PMU complex voltages with previous/fallback values.
    """
    completed = pmu_complex_obs.astype(np.complex128).copy()
    present_mask = pmu_present > 0.5

    if np.all(present_mask):
        return completed

    if fallback_complex is not None:
        completed[~present_mask] = fallback_complex[~present_mask]
        return completed

    if np.any(present_mask):
        fill_value = np.mean(completed[present_mask])
    else:
        fill_value = 1.0 + 0.0j

    completed[~present_mask] = fill_value
    return completed


def _static_hidden_complex_from_pmus(
    B: np.ndarray,
    buses: list[int],
    pmu_buses: list[int],
    v_pmu: np.ndarray,
    ridge: float,
) -> tuple[list[int], np.ndarray]:
    bus_to_idx = {bus: idx for idx, bus in enumerate(buses)}

    pmu_idx = [bus_to_idx[bus] for bus in pmu_buses]
    hidden_buses = [bus for bus in buses if bus not in pmu_buses]
    hidden_idx = [bus_to_idx[bus] for bus in hidden_buses]

    B_hh = B[np.ix_(hidden_idx, hidden_idx)]
    B_hp = B[np.ix_(hidden_idx, pmu_idx)]

    lhs = B_hh + ridge * np.eye(B_hh.shape[0])
    rhs = -B_hp @ v_pmu

    v_hidden = np.linalg.solve(lhs, rhs)
    return hidden_buses, v_hidden


def _build_full_static_prior(
    B: np.ndarray,
    buses: list[int],
    pmu_buses: list[int],
    pmu_v_complex_completed: np.ndarray,
    ridge: float,
) -> np.ndarray:
    hidden_buses, v_hidden = _static_hidden_complex_from_pmus(
        B=B,
        buses=buses,
        pmu_buses=pmu_buses,
        v_pmu=pmu_v_complex_completed,
        ridge=ridge,
    )

    full_v: dict[int, complex] = {}
    for local_idx, bus_id in enumerate(pmu_buses):
        full_v[bus_id] = complex(pmu_v_complex_completed[local_idx])
    for local_idx, bus_id in enumerate(hidden_buses):
        full_v[bus_id] = complex(v_hidden[local_idx])

    return np.array([full_v[bus_id] for bus_id in buses], dtype=np.complex128)


def _solve_global_wls_field(
    n_buses: int,
    observed_indices: np.ndarray,
    observed_values: np.ndarray,
    laplacian: np.ndarray,
    obs_weight: float,
    smooth_weight: float,
    ridge: float,
    reference_field: np.ndarray | None = None,
    reference_weight: float = 0.0,
) -> np.ndarray:
    if len(observed_indices) == 0:
        if reference_field is None:
            return np.zeros(n_buses, dtype=float)
        return reference_field.copy()

    S = np.zeros((len(observed_indices), n_buses), dtype=float)
    S[np.arange(len(observed_indices)), observed_indices] = 1.0

    I = np.eye(n_buses, dtype=float)

    lhs = obs_weight * (S.T @ S) + smooth_weight * laplacian + ridge * I
    rhs = obs_weight * (S.T @ observed_values)

    if reference_field is not None and reference_weight > 0.0:
        lhs = lhs + reference_weight * I
        rhs = rhs + reference_weight * reference_field

    return np.linalg.solve(lhs, rhs)


def _initial_angle_reference_from_first_snapshot(
    buses: list[int],
    pmu_buses: list[int],
    bus_to_idx: dict[int, int],
    B: np.ndarray,
    bus_frames: dict[int, pd.DataFrame],
    config: GlobalWLSAngleResidualEKFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    pmu_present0 = np.array(
        [bus_frames[bus_id]["DATA_PRESENT"].iloc[0] for bus_id in pmu_buses],
        dtype=float,
    )
    pmu_v0 = np.array(
        [bus_frames[bus_id]["V_complex_pu"].iloc[0] for bus_id in pmu_buses],
        dtype=np.complex128,
    )

    pmu_v0_completed = _complete_pmu_snapshot(
        pmu_complex_obs=pmu_v0,
        pmu_present=pmu_present0,
        fallback_complex=None,
    )

    full_static0 = _build_full_static_prior(
        B=B,
        buses=buses,
        pmu_buses=pmu_buses,
        pmu_v_complex_completed=pmu_v0_completed,
        ridge=config.static_ridge,
    )

    prior_mag0 = np.abs(full_static0)
    prior_ang0 = np.rad2deg(np.angle(full_static0))

    pmu_indices = np.array([bus_to_idx[bus] for bus in pmu_buses], dtype=int)
    observed_indices = pmu_indices[pmu_present0 > 0.5]
    observed_values = np.rad2deg(np.angle(pmu_v0[pmu_present0 > 0.5]))

    if len(observed_indices) > 0:
        prior_ang0 = _solve_global_wls_field(
            n_buses=len(buses),
            observed_indices=observed_indices,
            observed_values=observed_values,
            laplacian=B,
            obs_weight=config.obs_weight_ang,
            smooth_weight=config.smooth_weight_ang,
            ridge=config.ridge_prior_ang,
            reference_field=prior_ang0,
            reference_weight=config.temporal_ref_weight_ang,
        )

    return prior_mag0, prior_ang0


def _predict_step(
    x: np.ndarray,
    P: np.ndarray,
    dt: float,
    coherent_gain: float,
    config: GlobalWLSAngleResidualEKFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_modes = len(x) - 2

    a_ang = x[:n_modes].copy()
    freq_dev = float(x[-2])
    rocof = float(x[-1])

    a_ang[0] += coherent_gain * 360.0 * dt * freq_dev * config.coherent_angle_coupling
    freq_dev_next = freq_dev + dt * rocof
    rocof_next = config.rocof_decay * rocof

    x_pred = np.concatenate([a_ang, np.array([freq_dev_next, rocof_next])])

    F = np.eye(len(x))
    F[0, n_modes] = coherent_gain * 360.0 * dt * config.coherent_angle_coupling
    F[n_modes, n_modes + 1] = dt
    F[n_modes + 1, n_modes + 1] = config.rocof_decay

    q_diag = [config.q_ang] * n_modes + [config.q_freq, config.q_rocof]
    Q = np.diag(q_diag)

    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def _measurement_update(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    U_p: np.ndarray,
    pmu_ang_residual: np.ndarray,
    pmu_present: np.ndarray,
    mean_freq_dev: float,
    mean_rocof: float,
    config: GlobalWLSAngleResidualEKFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_modes = len(x_pred) - 2

    rows: list[np.ndarray] = []
    values: list[float] = []
    predictions: list[float] = []
    noise_diag: list[float] = []

    freq_state_idx = n_modes
    rocof_state_idx = n_modes + 1

    for local_idx in range(U_p.shape[0]):
        if pmu_present[local_idx] < 0.5:
            continue

        h_ang = np.zeros(len(x_pred))
        h_ang[:n_modes] = U_p[local_idx, :]
        rows.append(h_ang)
        values.append(float(pmu_ang_residual[local_idx]))
        predictions.append(float(h_ang @ x_pred))
        noise_diag.append(config.r_ang_deg)

    h_freq = np.zeros(len(x_pred))
    h_freq[freq_state_idx] = 1.0
    rows.append(h_freq)
    values.append(float(mean_freq_dev))
    predictions.append(float(h_freq @ x_pred))
    noise_diag.append(config.r_freq)

    h_rocof = np.zeros(len(x_pred))
    h_rocof[rocof_state_idx] = 1.0
    rows.append(h_rocof)
    values.append(float(mean_rocof))
    predictions.append(float(h_rocof @ x_pred))
    noise_diag.append(config.r_rocof)

    H = np.vstack(rows)
    y = np.array(values, dtype=float)
    yhat = np.array(predictions, dtype=float)
    R = np.diag(noise_diag)

    innovation = y - yhat

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_upd = x_pred + K @ innovation
    P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred

    return x_upd, P_upd


def _reconstruct_angle_field(
    prior_ang: np.ndarray,
    U: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    n_modes = U.shape[1]
    a_ang = x[:n_modes]
    return prior_ang + U @ a_ang


def _summarize_results(
    results_df: pd.DataFrame,
    scenario_id: str,
    representation: str,
    n_total_buses: int,
    n_pmu_buses: int,
    n_hidden_buses: int,
    n_timestamps: int,
    n_modes: int,
) -> GlobalWLSAngleResidualEKFSummary:
    def _mag_rmse(frame: pd.DataFrame) -> float:
        return float(np.sqrt(np.mean(np.square(frame["mag_error"].to_numpy()))))

    def _ang_mae(frame: pd.DataFrame) -> float:
        return float(np.mean(np.abs(frame["angle_error_deg"].to_numpy())))

    overall = results_df
    normal = results_df[results_df["Event"] == 0]
    event = results_df[results_df["Event"] != 0]

    return GlobalWLSAngleResidualEKFSummary(
        scenario_id=scenario_id,
        representation=representation,
        n_total_buses=n_total_buses,
        n_pmu_buses=n_pmu_buses,
        n_hidden_buses=n_hidden_buses,
        n_timestamps=n_timestamps,
        n_modes=n_modes,
        mag_rmse_overall=_mag_rmse(overall),
        mag_rmse_normal=_mag_rmse(normal),
        mag_rmse_event=_mag_rmse(event),
        angle_mae_deg_overall=_ang_mae(overall),
        angle_mae_deg_normal=_ang_mae(normal),
        angle_mae_deg_event=_ang_mae(event),
    )


def run_global_wls_angle_residual_ekf_estimator(
    scenario_id: str = "SIM_0001",
    config: GlobalWLSAngleResidualEKFConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, GlobalWLSAngleResidualEKFSummary]:
    if config is None:
        config = GlobalWLSAngleResidualEKFConfig()

    metadata = load_synth_metadata(scenario_id=scenario_id)
    pmu_buses = sorted(int(bus) for bus in metadata["pmu_buses"])
    all_buses = sorted(int(bus) for bus in metadata["output_buses"])
    nominal_frequency_hz = float(metadata["nominal_frequency_hz"])

    B, buses, bus_to_idx = _build_susceptance_laplacian(metadata)
    U = _build_graph_basis(B, config.n_modes)

    pmu_indices_global = np.array([bus_to_idx[bus] for bus in pmu_buses], dtype=int)
    hidden_buses = [bus for bus in buses if bus not in pmu_buses]
    hidden_indices_global = np.array([bus_to_idx[bus] for bus in hidden_buses], dtype=int)

    U_p = U[pmu_indices_global, :]
    bus_frames = _load_bus_frames(
        scenario_id=scenario_id,
        representation=config.representation,
        measurement_base_mode=config.measurement_base_mode,
    )

    base_bus = all_buses[0]
    timestamps = bus_frames[base_bus]["TIMESTAMP"].to_numpy()
    events = bus_frames[base_bus]["Event"].to_numpy()

    _, prior_ang_prev = _initial_angle_reference_from_first_snapshot(
        buses=buses,
        pmu_buses=pmu_buses,
        bus_to_idx=bus_to_idx,
        B=B,
        bus_frames=bus_frames,
        config=config,
    )

    n_modes = U.shape[1]
    x = np.zeros(n_modes + 2, dtype=float)
    P = np.diag([config.p0_ang] * n_modes + [config.p0_freq, config.p0_rocof])

    coherent_gain = 1.0 / max(abs(np.mean(U[:, 0])), 1e-6)

    fallback_pmu_complex: np.ndarray | None = None
    records: list[dict[str, Any]] = []

    for t_idx in range(len(timestamps)):
        if t_idx > 0:
            dt = float(timestamps[t_idx] - timestamps[t_idx - 1])
            x, P = _predict_step(
                x=x,
                P=P,
                dt=dt,
                coherent_gain=coherent_gain,
                config=config,
            )

        pmu_v_complex_obs = np.array(
            [bus_frames[bus_id]["V_complex_pu"].iloc[t_idx] for bus_id in pmu_buses],
            dtype=np.complex128,
        )
        pmu_ang_obs = np.array(
            [np.rad2deg(np.angle(bus_frames[bus_id]["V_complex_pu"].iloc[t_idx])) for bus_id in pmu_buses],
            dtype=float,
        )
        pmu_freq = np.array(
            [bus_frames[bus_id]["Freq"].iloc[t_idx] for bus_id in pmu_buses],
            dtype=float,
        )
        pmu_rocof = np.array(
            [bus_frames[bus_id]["ROCOF"].iloc[t_idx] for bus_id in pmu_buses],
            dtype=float,
        )
        pmu_present = np.array(
            [bus_frames[bus_id]["DATA_PRESENT"].iloc[t_idx] for bus_id in pmu_buses],
            dtype=float,
        )

        pmu_v_complex_completed = _complete_pmu_snapshot(
            pmu_complex_obs=pmu_v_complex_obs,
            pmu_present=pmu_present,
            fallback_complex=fallback_pmu_complex,
        )
        fallback_pmu_complex = pmu_v_complex_completed.copy()

        # Static prior for magnitude and hidden phasors
        full_static_prior = _build_full_static_prior(
            B=B,
            buses=buses,
            pmu_buses=pmu_buses,
            pmu_v_complex_completed=pmu_v_complex_completed,
            ridge=config.static_ridge,
        )
        prior_mag = np.abs(full_static_prior)

        # Global WLS prior for angle field
        valid_mask = pmu_present > 0.5
        observed_indices = pmu_indices_global[valid_mask]
        observed_values = pmu_ang_obs[valid_mask]

        prior_ang = _solve_global_wls_field(
            n_buses=len(buses),
            observed_indices=observed_indices,
            observed_values=observed_values,
            laplacian=B,
            obs_weight=config.obs_weight_ang,
            smooth_weight=config.smooth_weight_ang,
            ridge=config.ridge_prior_ang,
            reference_field=prior_ang_prev,
            reference_weight=config.temporal_ref_weight_ang,
        )

        pmu_ang_prior = prior_ang[pmu_indices_global]
        pmu_ang_residual = _wrap_angle_deg(pmu_ang_obs - pmu_ang_prior)

        mean_freq_dev = float(np.mean(pmu_freq[valid_mask]) - nominal_frequency_hz) if np.any(valid_mask) else 0.0
        mean_rocof = float(np.mean(pmu_rocof[valid_mask])) if np.any(valid_mask) else 0.0

        x, P = _measurement_update(
            x_pred=x,
            P_pred=P,
            U_p=U_p,
            pmu_ang_residual=pmu_ang_residual,
            pmu_present=pmu_present,
            mean_freq_dev=mean_freq_dev,
            mean_rocof=mean_rocof,
            config=config,
        )

        ang_hat_deg_all = _reconstruct_angle_field(
            prior_ang=prior_ang,
            U=U,
            x=x,
        )

        # Anchor PMU buses exactly where PMU data is present
        ang_hat_deg_all[pmu_indices_global[valid_mask]] = pmu_ang_obs[valid_mask]

        prior_ang_prev = ang_hat_deg_all.copy()

        for hidden_local_idx, bus_id in enumerate(hidden_buses):
            global_idx = hidden_indices_global[hidden_local_idx]
            v_true_actual = bus_frames[bus_id]["V_complex_actual"].iloc[t_idx]

            mag_true = float(np.abs(v_true_actual))
            ang_true = float(np.rad2deg(np.angle(v_true_actual)))

            mag_hat_actual = float(magnitude_from_pu(prior_mag[global_idx], bus_id, mode=config.measurement_base_mode))
            ang_hat = float(ang_hat_deg_all[global_idx])
            ang_err = float(_wrap_angle_deg(np.array([ang_hat - ang_true]))[0])

            records.append(
                {
                    "TIMESTAMP": float(timestamps[t_idx]),
                    "Event": int(events[t_idx]),
                    "bus_id": int(bus_id),
                    "mag_true": mag_true,
                    "mag_hat": mag_hat_actual,
                    "mag_error": mag_hat_actual - mag_true,
                    "mag_abs_error": abs(mag_hat_actual - mag_true),
                    "angle_true_deg": ang_true,
                    "angle_hat_deg": ang_hat,
                    "angle_error_deg": ang_err,
                    "angle_abs_error_deg": abs(ang_err),
                    "freq_dev_hat_hz": float(x[-2]),
                    "rocof_hat_hz_per_s": float(x[-1]),
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
        n_modes=n_modes,
    )

    return results_df, per_bus_summary_df, summary


def summary_to_dict(summary: GlobalWLSAngleResidualEKFSummary) -> dict[str, Any]:
    return asdict(summary)