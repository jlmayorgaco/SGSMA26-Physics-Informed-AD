from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.data_loader import load_synth_bus_data, load_synth_metadata


VoltageRepresentation = Literal["A", "B", "C", "positive_sequence"]


# IEEE 39-bus nominal line-line kV values from the competition guide.
# Used to convert measured line-neutral RMS volts and line currents to per-unit.
BUS_BASE_KV_LL: dict[int, float] = {
    1: 345.0,
    2: 345.0,
    3: 345.0,
    4: 345.0,
    5: 345.0,
    6: 345.0,
    7: 345.0,
    8: 345.0,
    9: 345.0,
    10: 345.0,
    11: 345.0,
    12: 345.0,
    13: 345.0,
    14: 345.0,
    15: 345.0,
    16: 345.0,
    17: 345.0,
    18: 345.0,
    19: 345.0,
    20: 110.0,
    21: 345.0,
    22: 345.0,
    23: 345.0,
    24: 345.0,
    25: 345.0,
    26: 345.0,
    27: 345.0,
    28: 345.0,
    29: 345.0,
    30: 13.8,
    31: 13.8,
    32: 13.8,
    33: 13.8,
    34: 13.8,
    35: 13.8,
    36: 13.8,
    37: 13.8,
    38: 13.8,
    39: 345.0,
}


@dataclass(frozen=True)
class YBusCurrentEstimatorConfig:
    representation: str = "positive_sequence"
    s_base_mva: float = 100.0

    # Linear solve regularization
    ridge: float = 1e-8
    current_weight: float = 1.0
    prior_weight: float = 10.0
    l2_weight: float = 1e-6

    # PMU handling
    pmu_imputation: str = "ffill"

    # Current sign handling
    auto_current_sign: bool = True
    fixed_current_sign: float = 1.0
    sign_calibration_samples: int = 120


@dataclass(frozen=True)
class YBusCurrentEstimatorSummary:
    scenario_id: str
    representation: str
    current_sign_used: float
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

    raise ValueError(f"Unsupported voltage representation: {representation}")


def _extract_bus_complex_current(
    df_bus: pd.DataFrame,
    representation: VoltageRepresentation,
) -> np.ndarray:
    if representation == "A":
        return _phasor_from_mag_angle(df_bus["IA_MAG"].to_numpy(), df_bus["IA_ANG"].to_numpy())

    if representation == "B":
        return _phasor_from_mag_angle(df_bus["IB_MAG"].to_numpy(), df_bus["IB_ANG"].to_numpy())

    if representation == "C":
        return _phasor_from_mag_angle(df_bus["IC_MAG"].to_numpy(), df_bus["IC_ANG"].to_numpy())

    if representation == "positive_sequence":
        return _positive_sequence_from_abc(
            df_bus["IA_MAG"].to_numpy(),
            df_bus["IA_ANG"].to_numpy(),
            df_bus["IB_MAG"].to_numpy(),
            df_bus["IB_ANG"].to_numpy(),
            df_bus["IC_MAG"].to_numpy(),
            df_bus["IC_ANG"].to_numpy(),
        )

    raise ValueError(f"Unsupported current representation: {representation}")


def _bus_base_voltage_ln_volts(bus_id: int) -> float:
    kv_ll = BUS_BASE_KV_LL[bus_id]
    return kv_ll * 1e3 / np.sqrt(3.0)


def _bus_base_current_amperes(bus_id: int, s_base_mva: float) -> float:
    kv_ll = BUS_BASE_KV_LL[bus_id]
    return (s_base_mva * 1e6) / (np.sqrt(3.0) * kv_ll * 1e3)


def _build_ybus_from_metadata(
    metadata: dict,
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """
    Build a complex Ybus surrogate from branch reactances only.

    For z = jx:
        y = 1 / (j x) = -j / x
        Y_ii += y
        Y_ij -= y

    This is still a simplified Ybus because resistances, shunts, taps,
    and detailed transformer modeling are not included.
    """
    buses = sorted(int(bus) for bus in metadata["output_buses"])
    bus_to_idx = {bus: idx for idx, bus in enumerate(buses)}
    n_buses = len(buses)

    Y = np.zeros((n_buses, n_buses), dtype=np.complex128)

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

        y_series = 1.0 / (1j * x_pu)

        i = bus_to_idx[int(from_bus)]
        j = bus_to_idx[int(to_bus)]

        Y[i, i] += y_series
        Y[j, j] += y_series
        Y[i, j] -= y_series
        Y[j, i] -= y_series

    return Y, buses, bus_to_idx


def _load_bus_frames(
    scenario_id: str,
    representation: VoltageRepresentation,
    s_base_mva: float,
) -> dict[int, pd.DataFrame]:
    metadata = load_synth_metadata(scenario_id=scenario_id)
    buses = sorted(int(bus) for bus in metadata["output_buses"])

    frames: dict[int, pd.DataFrame] = {}

    for bus_id in buses:
        df_bus = load_synth_bus_data(scenario_id=scenario_id, bus_id=bus_id)

        v_actual = _extract_bus_complex_voltage(df_bus, representation)
        i_actual = _extract_bus_complex_current(df_bus, representation)

        v_base_ln = _bus_base_voltage_ln_volts(bus_id)
        i_base = _bus_base_current_amperes(bus_id, s_base_mva)

        frames[bus_id] = pd.DataFrame(
            {
                "TIMESTAMP": df_bus["TIMESTAMP"].to_numpy(),
                "Event": df_bus["Event"].to_numpy(),
                "DATA_PRESENT": df_bus["DATA_PRESENT"].to_numpy(),
                "V_complex_actual": v_actual,
                "I_complex_actual": i_actual,
                "V_complex_pu": v_actual / v_base_ln,
                "I_complex_pu": i_actual / i_base,
            }
        )

    return frames


def _ffill_complex(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    result = values.astype(np.complex128).copy()
    valid_mask = valid_mask.astype(bool)

    last_valid: complex | None = None

    for idx in range(len(result)):
        is_valid = valid_mask[idx] and np.isfinite(result[idx].real) and np.isfinite(result[idx].imag)
        if is_valid:
            last_valid = result[idx]
        else:
            if last_valid is not None:
                result[idx] = last_valid

    if last_valid is None:
        return np.zeros_like(result)

    first_valid_idx = None
    for idx in range(len(result)):
        is_valid = valid_mask[idx] and np.isfinite(values[idx].real) and np.isfinite(values[idx].imag)
        if is_valid:
            first_valid_idx = idx
            break

    if first_valid_idx is not None and first_valid_idx > 0:
        result[:first_valid_idx] = result[first_valid_idx]

    return result


def _apply_pmu_imputation(
    values_2d: np.ndarray,
    valid_mask_2d: np.ndarray,
    mode: str,
) -> np.ndarray:
    if mode != "ffill":
        raise ValueError(f"Unsupported pmu_imputation mode: {mode}")

    filled = values_2d.copy()
    for col_idx in range(filled.shape[1]):
        filled[:, col_idx] = _ffill_complex(filled[:, col_idx], valid_mask_2d[:, col_idx])

    return filled


def _solve_linear_system(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _static_prior_hidden_voltage(
    Y_hh: np.ndarray,
    Y_hp: np.ndarray,
    v_pmu_pu: np.ndarray,
    ridge: float,
) -> np.ndarray:
    lhs = Y_hh + ridge * np.eye(Y_hh.shape[0], dtype=np.complex128)
    rhs = -Y_hp @ v_pmu_pu
    return _solve_linear_system(lhs, rhs)


def _choose_current_sign(
    Y_pp: np.ndarray,
    Y_ph: np.ndarray,
    Y_hh: np.ndarray,
    Y_hp: np.ndarray,
    v_pmu_pu: np.ndarray,
    i_pmu_pu: np.ndarray,
    data_present: np.ndarray,
    config: YBusCurrentEstimatorConfig,
    events: np.ndarray,
) -> float:
    if not config.auto_current_sign:
        return float(config.fixed_current_sign)

    normal_indices = np.where(events == 0)[0]
    if len(normal_indices) == 0:
        return float(config.fixed_current_sign)

    calibration_indices = normal_indices[: config.sign_calibration_samples]
    if len(calibration_indices) == 0:
        return float(config.fixed_current_sign)

    candidate_signs = [+1.0, -1.0]
    mean_residual_by_sign: dict[float, float] = {}

    for sign in candidate_signs:
        residuals: list[float] = []

        for t_idx in calibration_indices:
            v_p = v_pmu_pu[t_idx, :]
            i_p = sign * i_pmu_pu[t_idx, :]
            obs_mask = data_present[t_idx, :] > 0.5

            if np.sum(obs_mask) == 0:
                continue

            v_h_prior = _static_prior_hidden_voltage(
                Y_hh=Y_hh,
                Y_hp=Y_hp,
                v_pmu_pu=v_p,
                ridge=config.ridge,
            )

            rhs_obs = i_p[obs_mask] - Y_pp[np.ix_(obs_mask, np.arange(Y_pp.shape[1]))] @ v_p
            residual = Y_ph[np.ix_(obs_mask, np.arange(Y_ph.shape[1]))] @ v_h_prior - rhs_obs

            residuals.append(float(np.linalg.norm(residual) / np.sqrt(np.sum(obs_mask))))

        if len(residuals) == 0:
            mean_residual_by_sign[sign] = np.inf
        else:
            mean_residual_by_sign[sign] = float(np.mean(residuals))

    return min(mean_residual_by_sign, key=mean_residual_by_sign.get)


def _estimate_hidden_voltage_pu(
    Y_pp: np.ndarray,
    Y_ph: np.ndarray,
    Y_hh: np.ndarray,
    Y_hp: np.ndarray,
    v_pmu_pu: np.ndarray,
    i_pmu_pu: np.ndarray,
    data_present: np.ndarray,
    config: YBusCurrentEstimatorConfig,
) -> tuple[np.ndarray, float]:
    n_timestamps = v_pmu_pu.shape[0]
    n_hidden = Y_hh.shape[0]

    hidden_hat = np.zeros((n_timestamps, n_hidden), dtype=np.complex128)
    residual_accumulator = 0.0

    l2_identity = np.eye(n_hidden, dtype=np.complex128)

    for t_idx in range(n_timestamps):
        v_p = v_pmu_pu[t_idx, :]
        i_p = i_pmu_pu[t_idx, :]
        obs_mask = data_present[t_idx, :] > 0.5

        v_h_prior = _static_prior_hidden_voltage(
            Y_hh=Y_hh,
            Y_hp=Y_hp,
            v_pmu_pu=v_p,
            ridge=config.ridge,
        )

        if np.sum(obs_mask) == 0:
            hidden_hat[t_idx, :] = v_h_prior
            continue

        Y_ph_obs = Y_ph[np.ix_(obs_mask, np.arange(n_hidden))]
        Y_pp_obs_full = Y_pp[np.ix_(obs_mask, np.arange(Y_pp.shape[1]))]
        rhs_current = i_p[obs_mask] - Y_pp_obs_full @ v_p

        lhs = (
            config.current_weight * (Y_ph_obs.conj().T @ Y_ph_obs)
            + config.prior_weight * l2_identity
            + config.l2_weight * l2_identity
        )
        rhs = (
            config.current_weight * (Y_ph_obs.conj().T @ rhs_current)
            + config.prior_weight * v_h_prior
        )

        v_h_hat = _solve_linear_system(lhs, rhs)
        hidden_hat[t_idx, :] = v_h_hat

        residual = Y_ph_obs @ v_h_hat - rhs_current
        residual_accumulator += float(np.linalg.norm(residual) / np.sqrt(np.sum(obs_mask)))

    mean_current_residual = residual_accumulator / max(n_timestamps, 1)
    return hidden_hat, mean_current_residual


def _summarize_results(
    results_df: pd.DataFrame,
    scenario_id: str,
    representation: str,
    current_sign_used: float,
    n_total_buses: int,
    n_pmu_buses: int,
    n_hidden_buses: int,
    n_timestamps: int,
) -> YBusCurrentEstimatorSummary:
    def _mag_rmse(frame: pd.DataFrame) -> float:
        return float(np.sqrt(np.mean(np.square(frame["mag_error"].to_numpy()))))

    def _ang_mae(frame: pd.DataFrame) -> float:
        return float(np.mean(np.abs(frame["angle_error_deg"].to_numpy())))

    overall = results_df
    normal = results_df[results_df["Event"] == 0]
    event = results_df[results_df["Event"] != 0]

    return YBusCurrentEstimatorSummary(
        scenario_id=scenario_id,
        representation=representation,
        current_sign_used=float(current_sign_used),
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


def run_ybus_current_estimator(
    scenario_id: str = "SIM_0001",
    config: YBusCurrentEstimatorConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, YBusCurrentEstimatorSummary]:
    """
    Current-informed hidden-bus voltage estimator.

    Model:
        I_p = Y_pp V_p + Y_ph V_h

    Since this is underdetermined, estimate V_h via regularized complex LS:
        min ||Y_ph V_h - (I_p - Y_pp V_p)||^2
            + lambda_prior ||V_h - V_h_prior||^2
            + lambda_l2 ||V_h||^2

    where:
        V_h_prior = -Y_hh^{-1} Y_hp V_p

    Returns
    -------
    results_df
        Long dataframe with per-hidden-bus, per-timestamp estimates and errors.
    per_bus_summary_df
        Aggregated metrics per hidden bus.
    summary
        Global estimator summary.
    """
    if config is None:
        config = YBusCurrentEstimatorConfig()

    metadata = load_synth_metadata(scenario_id=scenario_id)
    pmu_buses = sorted(int(bus) for bus in metadata["pmu_buses"])
    all_buses = sorted(int(bus) for bus in metadata["output_buses"])

    Y, buses, bus_to_idx = _build_ybus_from_metadata(metadata)

    pmu_idx = [bus_to_idx[bus] for bus in pmu_buses]
    hidden_buses = [bus for bus in buses if bus not in pmu_buses]
    hidden_idx = [bus_to_idx[bus] for bus in hidden_buses]

    Y_pp = Y[np.ix_(pmu_idx, pmu_idx)]
    Y_ph = Y[np.ix_(pmu_idx, hidden_idx)]
    Y_hh = Y[np.ix_(hidden_idx, hidden_idx)]
    Y_hp = Y[np.ix_(hidden_idx, pmu_idx)]

    bus_frames = _load_bus_frames(
        scenario_id=scenario_id,
        representation=config.representation,
        s_base_mva=config.s_base_mva,
    )

    base_bus = all_buses[0]
    timestamps = bus_frames[base_bus]["TIMESTAMP"].to_numpy()
    events = bus_frames[base_bus]["Event"].to_numpy()

    v_pmu_pu = np.column_stack(
        [bus_frames[bus_id]["V_complex_pu"].to_numpy() for bus_id in pmu_buses]
    )
    i_pmu_pu = np.column_stack(
        [bus_frames[bus_id]["I_complex_pu"].to_numpy() for bus_id in pmu_buses]
    )
    data_present = np.column_stack(
        [bus_frames[bus_id]["DATA_PRESENT"].to_numpy() for bus_id in pmu_buses]
    ).astype(float)

    v_pmu_pu = _apply_pmu_imputation(
        values_2d=v_pmu_pu,
        valid_mask_2d=data_present > 0.5,
        mode=config.pmu_imputation,
    )
    i_pmu_pu = _apply_pmu_imputation(
        values_2d=i_pmu_pu,
        valid_mask_2d=data_present > 0.5,
        mode=config.pmu_imputation,
    )

    current_sign_used = _choose_current_sign(
        Y_pp=Y_pp,
        Y_ph=Y_ph,
        Y_hh=Y_hh,
        Y_hp=Y_hp,
        v_pmu_pu=v_pmu_pu,
        i_pmu_pu=i_pmu_pu,
        data_present=data_present,
        config=config,
        events=events,
    )

    i_pmu_pu = current_sign_used * i_pmu_pu

    v_hidden_hat_pu, mean_current_residual = _estimate_hidden_voltage_pu(
        Y_pp=Y_pp,
        Y_ph=Y_ph,
        Y_hh=Y_hh,
        Y_hp=Y_hp,
        v_pmu_pu=v_pmu_pu,
        i_pmu_pu=i_pmu_pu,
        data_present=data_present,
        config=config,
    )

    records: list[dict[str, Any]] = []

    for hidden_col_idx, bus_id in enumerate(hidden_buses):
        v_true_actual = bus_frames[bus_id]["V_complex_actual"].to_numpy()

        v_base_ln = _bus_base_voltage_ln_volts(bus_id)
        v_hat_actual = v_hidden_hat_pu[:, hidden_col_idx] * v_base_ln

        mag_true = np.abs(v_true_actual)
        mag_hat = np.abs(v_hat_actual)

        ang_true = np.rad2deg(np.angle(v_true_actual))
        ang_hat = np.rad2deg(np.angle(v_hat_actual))
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
                    "current_sign_used": float(current_sign_used),
                    "mean_current_residual_norm_pu": float(mean_current_residual),
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
        current_sign_used=current_sign_used,
        n_total_buses=len(all_buses),
        n_pmu_buses=len(pmu_buses),
        n_hidden_buses=len(hidden_buses),
        n_timestamps=len(timestamps),
    )

    return results_df, per_bus_summary_df, summary


def summary_to_dict(summary: YBusCurrentEstimatorSummary) -> dict[str, Any]:
    return asdict(summary)



def diagnose_ybus_current_consistency(
    scenario_id: str = "SIM_0001",
    config: YBusCurrentEstimatorConfig | None = None,
) -> dict[str, float]:
    """
    Check whether the measured PMU currents are consistent with the Ybus surrogate
    when using the true voltages of all buses.

    If this mismatch is large, then the current-informed estimator is using an
    invalid current model, wrong sign convention, wrong base conversion, or an
    oversimplified Ybus.
    """
    if config is None:
        config = YBusCurrentEstimatorConfig()

    metadata = load_synth_metadata(scenario_id=scenario_id)
    pmu_buses = sorted(int(bus) for bus in metadata["pmu_buses"])
    all_buses = sorted(int(bus) for bus in metadata["output_buses"])

    Y, buses, bus_to_idx = _build_ybus_from_metadata(metadata)

    bus_frames = _load_bus_frames(
        scenario_id=scenario_id,
        representation=config.representation,
        s_base_mva=config.s_base_mva,
    )

    # Full true voltage matrix in pu
    V_all = np.column_stack(
        [bus_frames[bus_id]["V_complex_pu"].to_numpy() for bus_id in all_buses]
    )

    # Measured PMU currents in pu
    I_pmu_meas = np.column_stack(
        [bus_frames[bus_id]["I_complex_pu"].to_numpy() for bus_id in pmu_buses]
    )

    data_present = np.column_stack(
        [bus_frames[bus_id]["DATA_PRESENT"].to_numpy() for bus_id in pmu_buses]
    ).astype(float)

    I_pmu_meas = _apply_pmu_imputation(
        values_2d=I_pmu_meas,
        valid_mask_2d=data_present > 0.5,
        mode=config.pmu_imputation,
    )

    pmu_idx = [bus_to_idx[bus] for bus in pmu_buses]

    # Predicted PMU currents from full true voltages
    I_all_pred = (Y @ V_all.T).T
    I_pmu_pred = I_all_pred[:, pmu_idx]

    events = bus_frames[all_buses[0]]["Event"].to_numpy()

    results = {}

    for sign in (+1.0, -1.0):
        residual = sign * I_pmu_meas - I_pmu_pred
        residual_norm = np.linalg.norm(residual, axis=1)
        meas_norm = np.maximum(np.linalg.norm(sign * I_pmu_meas, axis=1), 1e-12)
        rel_residual = residual_norm / meas_norm

        normal_mask = events == 0
        event_mask = events != 0

        results[f"sign_{int(sign)}_mean_rel_residual_overall"] = float(np.mean(rel_residual))
        results[f"sign_{int(sign)}_mean_rel_residual_normal"] = float(np.mean(rel_residual[normal_mask]))
        results[f"sign_{int(sign)}_mean_rel_residual_event"] = float(np.mean(rel_residual[event_mask]))

    return results