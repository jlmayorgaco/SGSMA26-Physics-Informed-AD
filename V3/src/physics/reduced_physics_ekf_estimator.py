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
class ReducedPhysicsEKFConfig:
    representation: str = "positive_sequence"
    n_modes: int = 8
    s_base_mva: float = 100.0

    # Initial hidden reference from static graph interpolation
    static_ridge: float = 1e-8

    # Process model
    rocof_decay: float = 0.96
    coherent_angle_coupling: float = 1.0

    # Process noise
    q_mag: float = 5e-5
    q_ang: float = 5e-4
    q_freq: float = 2e-5
    q_rocof: float = 5e-4

    # Initial covariance
    p0_mag: float = 1e-3
    p0_ang: float = 1e-2
    p0_freq: float = 1e-3
    p0_rocof: float = 1e-2

    # Measurement noise
    r_mag: float = 2e-4
    r_ang_deg: float = 0.10
    r_freq: float = 2e-3
    r_rocof: float = 2e-2

    # Safety
    min_mag_pu: float = 0.70
    max_mag_pu: float = 1.30


@dataclass(frozen=True)
class ReducedPhysicsEKFSummary:
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
    eigenvectors = eigenvectors[:, order]

    n_modes = min(n_modes, eigenvectors.shape[1])
    U = eigenvectors[:, :n_modes].copy()

    # Enforce stable sign convention for reproducibility
    for col_idx in range(U.shape[1]):
        if np.sum(U[:, col_idx]) < 0.0:
            U[:, col_idx] *= -1.0

    return U


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


def _solve_weighted_ls(A: np.ndarray, b: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    lhs = A.T @ A + ridge * np.eye(A.shape[1])
    rhs = A.T @ b
    return np.linalg.solve(lhs, rhs)


def _initialize_reference_fields(
    metadata: dict[str, Any],
    bus_frames: dict[int, pd.DataFrame],
    B: np.ndarray,
    buses: list[int],
    config: ReducedPhysicsEKFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    pmu_buses = sorted(int(bus) for bus in metadata["pmu_buses"])

    v_pmu0 = np.array(
        [bus_frames[bus_id]["V_complex_pu"].iloc[0] for bus_id in pmu_buses],
        dtype=np.complex128,
    )

    hidden_buses, v_hidden0 = _static_hidden_complex_from_pmus(
        B=B,
        buses=buses,
        pmu_buses=pmu_buses,
        v_pmu=v_pmu0,
        ridge=config.static_ridge,
    )

    full_v0: dict[int, complex] = {}
    for bus_id in pmu_buses:
        full_v0[bus_id] = complex(bus_frames[bus_id]["V_complex_pu"].iloc[0])
    for idx, bus_id in enumerate(hidden_buses):
        full_v0[bus_id] = complex(v_hidden0[idx])

    v0_ordered = np.array([full_v0[bus_id] for bus_id in buses], dtype=np.complex128)

    mag_ref_pu = np.abs(v0_ordered)
    ang_ref_deg = np.rad2deg(np.angle(v0_ordered))

    return mag_ref_pu, ang_ref_deg


def _initial_state_from_pmus(
    U_p: np.ndarray,
    mag_ref_pmu: np.ndarray,
    ang_ref_pmu: np.ndarray,
    pmu_mag0: np.ndarray,
    pmu_ang0: np.ndarray,
    nominal_frequency_hz: float,
    pmu_freq0: np.ndarray,
    pmu_rocof0: np.ndarray,
) -> np.ndarray:
    mag_dev0 = pmu_mag0 - mag_ref_pmu
    ang_dev0 = _wrap_angle_deg(pmu_ang0 - ang_ref_pmu)

    a_mag0 = _solve_weighted_ls(U_p, mag_dev0)
    a_ang0 = _solve_weighted_ls(U_p, ang_dev0)

    freq_dev0 = float(np.mean(pmu_freq0) - nominal_frequency_hz)
    rocof0 = float(np.mean(pmu_rocof0))

    return np.concatenate([a_mag0, a_ang0, np.array([freq_dev0, rocof0])])


def _predict_step(
    x: np.ndarray,
    P: np.ndarray,
    dt: float,
    coherent_gain: float,
    config: ReducedPhysicsEKFConfig,
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
    pmu_bus_indices: list[int],
    U_p: np.ndarray,
    mag_ref_pmu: np.ndarray,
    ang_ref_pmu: np.ndarray,
    pmu_mag: np.ndarray,
    pmu_ang: np.ndarray,
    pmu_freq: np.ndarray,
    pmu_rocof: np.ndarray,
    pmu_present: np.ndarray,
    nominal_frequency_hz: float,
    config: ReducedPhysicsEKFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_modes = (len(x_pred) - 2) // 2

    observed_rows: list[np.ndarray] = []
    observed_values: list[float] = []
    observed_predictions: list[float] = []
    noise_values: list[float] = []

    freq_state_idx = 2 * n_modes
    rocof_state_idx = 2 * n_modes + 1

    for local_idx, _ in enumerate(pmu_bus_indices):
        if pmu_present[local_idx] < 0.5:
            continue

        # Magnitude measurement
        h_mag = np.zeros(len(x_pred))
        h_mag[:n_modes] = U_p[local_idx, :]
        y_mag = float(pmu_mag[local_idx] - mag_ref_pmu[local_idx])
        yhat_mag = float(h_mag @ x_pred)

        observed_rows.append(h_mag)
        observed_values.append(y_mag)
        observed_predictions.append(yhat_mag)
        noise_values.append(config.r_mag)

        # Angle measurement
        h_ang = np.zeros(len(x_pred))
        h_ang[n_modes : 2 * n_modes] = U_p[local_idx, :]
        y_ang = float(_wrap_angle_deg(np.array([pmu_ang[local_idx] - ang_ref_pmu[local_idx]]))[0])
        yhat_ang = float(h_ang @ x_pred)

        observed_rows.append(h_ang)
        observed_values.append(y_ang)
        observed_predictions.append(yhat_ang)
        noise_values.append(config.r_ang_deg)

        # Frequency measurement
        h_freq = np.zeros(len(x_pred))
        h_freq[freq_state_idx] = 1.0
        y_freq = float(pmu_freq[local_idx] - nominal_frequency_hz)
        yhat_freq = float(h_freq @ x_pred)

        observed_rows.append(h_freq)
        observed_values.append(y_freq)
        observed_predictions.append(yhat_freq)
        noise_values.append(config.r_freq)

        # ROCOF measurement
        h_rocof = np.zeros(len(x_pred))
        h_rocof[rocof_state_idx] = 1.0
        y_rocof = float(pmu_rocof[local_idx])
        yhat_rocof = float(h_rocof @ x_pred)

        observed_rows.append(h_rocof)
        observed_values.append(y_rocof)
        observed_predictions.append(yhat_rocof)
        noise_values.append(config.r_rocof)

    if not observed_rows:
        return x_pred, P_pred

    H = np.vstack(observed_rows)
    y = np.array(observed_values)
    yhat = np.array(observed_predictions)
    R = np.diag(noise_values)

    innovation = y - yhat
    innovation = innovation.astype(float)

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_upd = x_pred + K @ innovation
    P_upd = (np.eye(len(x_pred)) - K @ H) @ P_pred

    return x_upd, P_upd


def _reconstruct_fields(
    x: np.ndarray,
    U: np.ndarray,
    mag_ref_pu: np.ndarray,
    ang_ref_deg: np.ndarray,
    config: ReducedPhysicsEKFConfig,
) -> tuple[np.ndarray, np.ndarray]:
    n_modes = U.shape[1]

    a_mag = x[:n_modes]
    a_ang = x[n_modes : 2 * n_modes]

    mag_hat_pu = mag_ref_pu + U @ a_mag
    ang_hat_deg = ang_ref_deg + U @ a_ang

    mag_hat_pu = np.clip(mag_hat_pu, config.min_mag_pu, config.max_mag_pu)
    return mag_hat_pu, ang_hat_deg


def _summarize_results(
    results_df: pd.DataFrame,
    scenario_id: str,
    representation: str,
    n_total_buses: int,
    n_pmu_buses: int,
    n_hidden_buses: int,
    n_timestamps: int,
    n_modes: int,
) -> ReducedPhysicsEKFSummary:
    def _mag_rmse(frame: pd.DataFrame) -> float:
        return float(np.sqrt(np.mean(np.square(frame["mag_error"].to_numpy()))))

    def _ang_mae(frame: pd.DataFrame) -> float:
        return float(np.mean(np.abs(frame["angle_error_deg"].to_numpy())))

    overall = results_df
    normal = results_df[results_df["Event"] == 0]
    event = results_df[results_df["Event"] != 0]

    return ReducedPhysicsEKFSummary(
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


def run_reduced_physics_ekf_estimator(
    scenario_id: str = "SIM_0001",
    config: ReducedPhysicsEKFConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, ReducedPhysicsEKFSummary]:
    if config is None:
        config = ReducedPhysicsEKFConfig()

    metadata = load_synth_metadata(scenario_id=scenario_id)
    pmu_buses = sorted(int(bus) for bus in metadata["pmu_buses"])
    all_buses = sorted(int(bus) for bus in metadata["output_buses"])
    nominal_frequency_hz = float(metadata["nominal_frequency_hz"])

    B, buses, bus_to_idx = _build_susceptance_laplacian(metadata)
    U = _build_graph_basis(B, config.n_modes)

    pmu_indices_global = [bus_to_idx[bus] for bus in pmu_buses]
    hidden_buses = [bus for bus in buses if bus not in pmu_buses]
    hidden_indices_global = [bus_to_idx[bus] for bus in hidden_buses]

    U_p = U[pmu_indices_global, :]

    bus_frames = _load_bus_frames(
        scenario_id=scenario_id,
        representation=config.representation,
    )

    base_bus = all_buses[0]
    timestamps = bus_frames[base_bus]["TIMESTAMP"].to_numpy()
    events = bus_frames[base_bus]["Event"].to_numpy()

    mag_ref_pu, ang_ref_deg = _initialize_reference_fields(
        metadata=metadata,
        bus_frames=bus_frames,
        B=B,
        buses=buses,
        config=config,
    )

    mag_ref_pmu = mag_ref_pu[pmu_indices_global]
    ang_ref_pmu = ang_ref_deg[pmu_indices_global]

    pmu_mag0 = np.array(
        [np.abs(bus_frames[bus_id]["V_complex_pu"].iloc[0]) for bus_id in pmu_buses],
        dtype=float,
    )
    pmu_ang0 = np.array(
        [np.rad2deg(np.angle(bus_frames[bus_id]["V_complex_pu"].iloc[0])) for bus_id in pmu_buses],
        dtype=float,
    )
    pmu_freq0 = np.array(
        [bus_frames[bus_id]["Freq"].iloc[0] for bus_id in pmu_buses],
        dtype=float,
    )
    pmu_rocof0 = np.array(
        [bus_frames[bus_id]["ROCOF"].iloc[0] for bus_id in pmu_buses],
        dtype=float,
    )

    x = _initial_state_from_pmus(
        U_p=U_p,
        mag_ref_pmu=mag_ref_pmu,
        ang_ref_pmu=ang_ref_pmu,
        pmu_mag0=pmu_mag0,
        pmu_ang0=pmu_ang0,
        nominal_frequency_hz=nominal_frequency_hz,
        pmu_freq0=pmu_freq0,
        pmu_rocof0=pmu_rocof0,
    )

    n_modes = U.shape[1]
    P = np.diag(
        [config.p0_mag] * n_modes
        + [config.p0_ang] * n_modes
        + [config.p0_freq, config.p0_rocof]
    )

    coherent_gain = 1.0 / np.mean(U[:, 0])

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

        x, P = _measurement_update(
            x_pred=x,
            P_pred=P,
            pmu_bus_indices=pmu_indices_global,
            U_p=U_p,
            mag_ref_pmu=mag_ref_pmu,
            ang_ref_pmu=ang_ref_pmu,
            pmu_mag=pmu_mag,
            pmu_ang=pmu_ang,
            pmu_freq=pmu_freq,
            pmu_rocof=pmu_rocof,
            pmu_present=pmu_present,
            nominal_frequency_hz=nominal_frequency_hz,
            config=config,
        )

        mag_hat_pu_all, ang_hat_deg_all = _reconstruct_fields(
            x=x,
            U=U,
            mag_ref_pu=mag_ref_pu,
            ang_ref_deg=ang_ref_deg,
            config=config,
        )

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
        n_modes=U.shape[1],
    )

    return results_df, per_bus_summary_df, summary


def summary_to_dict(summary: ReducedPhysicsEKFSummary) -> dict[str, Any]:
    return asdict(summary)