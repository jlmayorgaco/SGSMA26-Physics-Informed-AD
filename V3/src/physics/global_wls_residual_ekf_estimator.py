from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.utils.data_loader import load_synth_bus_data, load_synth_metadata


VoltageRepresentation = Literal["A", "B", "C", "positive_sequence"]


BUS_BASE_KV_LL: dict[int, float] = {
    1: 345.0, 2: 345.0, 3: 345.0, 4: 345.0, 5: 345.0, 6: 345.0, 7: 345.0,
    8: 345.0, 9: 345.0, 10: 345.0, 11: 345.0, 12: 345.0, 13: 345.0, 14: 345.0,
    15: 345.0, 16: 345.0, 17: 345.0, 18: 345.0, 19: 345.0, 20: 110.0, 21: 345.0,
    22: 345.0, 23: 345.0, 24: 345.0, 25: 345.0, 26: 345.0, 27: 345.0, 28: 345.0,
    29: 345.0, 30: 13.8, 31: 13.8, 32: 13.8, 33: 13.8, 34: 13.8, 35: 13.8,
    36: 13.8, 37: 13.8, 38: 13.8, 39: 345.0,
}


@dataclass(frozen=True)
class GlobalWLSResidualEKFConfig:
    representation: str = "positive_sequence"
    n_modes: int = 8

    # Global WLS prior
    obs_weight_mag: float = 2.0e4
    obs_weight_ang: float = 2.0e4
    smooth_weight_mag: float = 5.0
    smooth_weight_ang: float = 10.0
    temporal_ref_weight_mag: float = 100.0
    temporal_ref_weight_ang: float = 150.0
    ridge_prior: float = 1.0e-6

    # EKF residual
    coherent_angle_coupling: float = 0.15
    rocof_decay: float = 0.96

    q_mag: float = 1.0e-6
    q_ang: float = 5.0e-5
    q_freq: float = 2.0e-5
    q_rocof: float = 5.0e-4

    p0_mag: float = 1.0e-4
    p0_ang: float = 5.0e-4
    p0_freq: float = 1.0e-4
    p0_rocof: float = 1.0e-3

    r_mag: float = 2.0e-5
    r_ang_deg: float = 0.03
    r_freq: float = 2.0e-3
    r_rocof: float = 2.0e-2

    min_mag_pu: float = 0.70
    max_mag_pu: float = 1.30


@dataclass(frozen=True)
class GlobalWLSResidualEKFSummary:
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


def _bus_base_voltage_ln_volts(bus_id: int) -> float:
    return BUS_BASE_KV_LL[bus_id] * 1e3 / np.sqrt(3.0)


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
) -> dict[int, pd.DataFrame]:
    metadata = load_synth_metadata(scenario_id=scenario_id)
    buses = sorted(int(bus) for bus in metadata["output_buses"])

    frames: dict[int, pd.DataFrame] = {}

    for bus_id in buses:
        df_bus = load_synth_bus_data(scenario_id=scenario_id, bus_id=bus_id)

        v_complex = _extract_bus_complex_voltage(df_bus, representation)
        v_base_ln = _bus_base_voltage_ln_volts(bus_id)

        frames[bus_id] = pd.DataFrame(
            {
                "TIMESTAMP": df_bus["TIMESTAMP"].to_numpy(),
                "Event": df_bus["Event"].to_numpy(),
                "DATA_PRESENT": df_bus["DATA_PRESENT"].to_numpy(),
                "Freq": df_bus["Freq"].to_numpy(),
                "ROCOF": df_bus["ROCOF"].to_numpy(),
                "V_complex_actual": v_complex,
                "V_complex_pu": v_complex / v_base_ln,
            }
        )

    return frames


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


def _initial_reference_from_first_pmu_snapshot(
    buses: list[int],
    pmu_buses: list[int],
    bus_to_idx: dict[int, int],
    bus_frames: dict[int, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    n_buses = len(buses)
    mag0 = np.zeros(n_buses, dtype=float)
    ang0 = np.zeros(n_buses, dtype=float)

    pmu_indices = [bus_to_idx[bus] for bus in pmu_buses]

    pmu_mag0 = np.array(
        [np.abs(bus_frames[bus_id]["V_complex_pu"].iloc[0]) for bus_id in pmu_buses],
        dtype=float,
    )
    pmu_ang0 = np.array(
        [np.rad2deg(np.angle(bus_frames[bus_id]["V_complex_pu"].iloc[0])) for bus_id in pmu_buses],
        dtype=float,
    )

    mag0[:] = float(np.mean(pmu_mag0))
    ang0[:] = float(np.mean(pmu_ang0))

    for idx, bus_idx in enumerate(pmu_indices):
        mag0[bus_idx] = pmu_mag0[idx]
        ang0[bus_idx] = pmu_ang0[idx]

    return mag0, ang0


def _predict_step(
    x: np.ndarray,
    P: np.ndarray,
    dt: float,
    coherent_gain: float,
    config: GlobalWLSResidualEKFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_modes = (len(x) - 2) // 2

    a_mag = x[:n_modes].copy()
    a_ang = x[n_modes : 2 * n_modes].copy()
    freq_dev = float(x[-2])
    rocof = float(x[-1])

    a_ang[0] += coherent_gain * 360.0 * dt * freq_dev * config.coherent_angle_coupling
    freq_dev_next = freq_dev + dt * rocof
    rocof_next = config.rocof_decay * rocof

    x_pred = np.concatenate([a_mag, a_ang, np.array([freq_dev_next, rocof_next])])

    F = np.eye(len(x))
    F[n_modes, 2 * n_modes] = coherent_gain * 360.0 * dt * config.coherent_angle_coupling
    F[2 * n_modes, 2 * n_modes + 1] = dt
    F[2 * n_modes + 1, 2 * n_modes + 1] = config.rocof_decay

    q_diag = (
        [config.q_mag] * n_modes
        + [config.q_ang] * n_modes
        + [config.q_freq, config.q_rocof]
    )
    Q = np.diag(q_diag)

    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def _measurement_update(
    x_pred: np.ndarray,
    P_pred: np.ndarray,
    U_p: np.ndarray,
    pmu_mag_residual: np.ndarray,
    pmu_ang_residual: np.ndarray,
    pmu_present: np.ndarray,
    mean_freq_dev: float,
    mean_rocof: float,
    config: GlobalWLSResidualEKFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_modes = (len(x_pred) - 2) // 2

    rows: list[np.ndarray] = []
    values: list[float] = []
    predictions: list[float] = []
    noise_diag: list[float] = []

    freq_state_idx = 2 * n_modes
    rocof_state_idx = 2 * n_modes + 1

    for local_idx in range(U_p.shape[0]):
        if pmu_present[local_idx] < 0.5:
            continue

        h_mag = np.zeros(len(x_pred))
        h_mag[:n_modes] = U_p[local_idx, :]
        rows.append(h_mag)
        values.append(float(pmu_mag_residual[local_idx]))
        predictions.append(float(h_mag @ x_pred))
        noise_diag.append(config.r_mag)

        h_ang = np.zeros(len(x_pred))
        h_ang[n_modes : 2 * n_modes] = U_p[local_idx, :]
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


def _reconstruct_fields(
    prior_mag: np.ndarray,
    prior_ang: np.ndarray,
    U: np.ndarray,
    x: np.ndarray,
    config: GlobalWLSResidualEKFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_modes = U.shape[1]

    a_mag = x[:n_modes]
    a_ang = x[n_modes : 2 * n_modes]

    mag_hat = prior_mag + U @ a_mag
    ang_hat = prior_ang + U @ a_ang

    mag_hat = np.clip(mag_hat, config.min_mag_pu, config.max_mag_pu)
    return mag_hat, ang_hat


def _summarize_results(
    results_df: pd.DataFrame,
    scenario_id: str,
    representation: str,
    n_total_buses: int,
    n_pmu_buses: int,
    n_hidden_buses: int,
    n_timestamps: int,
    n_modes: int,
) -> GlobalWLSResidualEKFSummary:
    def _mag_rmse(frame: pd.DataFrame) -> float:
        return float(np.sqrt(np.mean(np.square(frame["mag_error"].to_numpy()))))

    def _ang_mae(frame: pd.DataFrame) -> float:
        return float(np.mean(np.abs(frame["angle_error_deg"].to_numpy())))

    overall = results_df
    normal = results_df[results_df["Event"] == 0]
    event = results_df[results_df["Event"] != 0]

    return GlobalWLSResidualEKFSummary(
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


def run_global_wls_residual_ekf_estimator(
    scenario_id: str = "SIM_0001",
    config: GlobalWLSResidualEKFConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, GlobalWLSResidualEKFSummary]:
    if config is None:
        config = GlobalWLSResidualEKFConfig()

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
    )

    base_bus = all_buses[0]
    timestamps = bus_frames[base_bus]["TIMESTAMP"].to_numpy()
    events = bus_frames[base_bus]["Event"].to_numpy()

    prior_mag_prev, prior_ang_prev = _initial_reference_from_first_pmu_snapshot(
        buses=buses,
        pmu_buses=pmu_buses,
        bus_to_idx=bus_to_idx,
        bus_frames=bus_frames,
    )

    n_modes = U.shape[1]
    x = np.zeros(2 * n_modes + 2, dtype=float)
    P = np.diag(
        [config.p0_mag] * n_modes
        + [config.p0_ang] * n_modes
        + [config.p0_freq, config.p0_rocof]
    )

    coherent_gain = 1.0 / max(abs(np.mean(U[:, 0])), 1e-6)

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

        pmu_mag = np.array(
            [np.abs(bus_frames[bus_id]["V_complex_pu"].iloc[t_idx]) for bus_id in pmu_buses],
            dtype=float,
        )
        pmu_ang = np.array(
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

        valid_mask = pmu_present > 0.5
        observed_indices = pmu_indices_global[valid_mask]

        prior_mag = _solve_global_wls_field(
            n_buses=len(buses),
            observed_indices=observed_indices,
            observed_values=pmu_mag[valid_mask],
            laplacian=B,
            obs_weight=config.obs_weight_mag,
            smooth_weight=config.smooth_weight_mag,
            ridge=config.ridge_prior,
            reference_field=prior_mag_prev,
            reference_weight=config.temporal_ref_weight_mag,
        )

        prior_ang = _solve_global_wls_field(
            n_buses=len(buses),
            observed_indices=observed_indices,
            observed_values=pmu_ang[valid_mask],
            laplacian=B,
            obs_weight=config.obs_weight_ang,
            smooth_weight=config.smooth_weight_ang,
            ridge=config.ridge_prior,
            reference_field=prior_ang_prev,
            reference_weight=config.temporal_ref_weight_ang,
        )

        pmu_mag_prior = prior_mag[pmu_indices_global]
        pmu_ang_prior = prior_ang[pmu_indices_global]

        pmu_mag_residual = pmu_mag - pmu_mag_prior
        pmu_ang_residual = _wrap_angle_deg(pmu_ang - pmu_ang_prior)

        mean_freq_dev = float(np.mean(pmu_freq[valid_mask]) - nominal_frequency_hz) if np.any(valid_mask) else 0.0
        mean_rocof = float(np.mean(pmu_rocof[valid_mask])) if np.any(valid_mask) else 0.0

        x, P = _measurement_update(
            x_pred=x,
            P_pred=P,
            U_p=U_p,
            pmu_mag_residual=pmu_mag_residual,
            pmu_ang_residual=pmu_ang_residual,
            pmu_present=pmu_present,
            mean_freq_dev=mean_freq_dev,
            mean_rocof=mean_rocof,
            config=config,
        )

        mag_hat_pu_all, ang_hat_deg_all = _reconstruct_fields(
            prior_mag=prior_mag,
            prior_ang=prior_ang,
            U=U,
            x=x,
            config=config,
        )

        # Anchor PMU buses exactly when data is present
        mag_hat_pu_all[pmu_indices_global[valid_mask]] = pmu_mag[valid_mask]
        ang_hat_deg_all[pmu_indices_global[valid_mask]] = pmu_ang[valid_mask]

        prior_mag_prev = mag_hat_pu_all.copy()
        prior_ang_prev = ang_hat_deg_all.copy()

        for hidden_local_idx, bus_id in enumerate(hidden_buses):
            global_idx = hidden_indices_global[hidden_local_idx]
            v_true_actual = bus_frames[bus_id]["V_complex_actual"].iloc[t_idx]

            mag_true = float(np.abs(v_true_actual))
            ang_true = float(np.rad2deg(np.angle(v_true_actual)))

            mag_hat_actual = float(mag_hat_pu_all[global_idx] * _bus_base_voltage_ln_volts(bus_id))
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


def summary_to_dict(summary: GlobalWLSResidualEKFSummary) -> dict[str, Any]:
    return asdict(summary)