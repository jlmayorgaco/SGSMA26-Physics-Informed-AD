from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.utils.data_loader import load_synth_bus_data, load_synth_metadata


VoltageRepresentation = Literal["A", "B", "C", "positive_sequence"]


@dataclass(frozen=True)
class YBusBaselineSummary:
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


def _build_susceptance_matrix_from_metadata(
    metadata: dict,
) -> tuple[np.ndarray, list[int], dict[int, int]]:
    """
    Build a susceptance/Laplacian matrix from branch reactances.

    This is not the full complex Ybus.
    It is a topology-only surrogate using b_ij = 1 / x_ij.
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
            raise KeyError(
                f"Missing reactance for branch ({from_bus}, {to_bus}) in metadata"
            )

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


def _extract_bus_complex_voltage(
    df_bus: pd.DataFrame,
    representation: VoltageRepresentation,
) -> np.ndarray:
    """
    Convert single-bus dataframe into a complex voltage series.

    Assumes load_synth_bus_data() returns local columns like:
        VA_MAG, VA_ANG, VB_MAG, VB_ANG, VC_MAG, VC_ANG, ...
    """
    if representation == "A":
        return _phasor_from_mag_angle(df_bus["VA_MAG"].to_numpy(), df_bus["VA_ANG"].to_numpy())

    if representation == "B":
        return _phasor_from_mag_angle(df_bus["VB_MAG"].to_numpy(), df_bus["VB_ANG"].to_numpy())

    if representation == "C":
        return _phasor_from_mag_angle(df_bus["VC_MAG"].to_numpy(), df_bus["VC_ANG"].to_numpy())

    if representation == "positive_sequence":
        va = _phasor_from_mag_angle(df_bus["VA_MAG"].to_numpy(), df_bus["VA_ANG"].to_numpy())
        vb = _phasor_from_mag_angle(df_bus["VB_MAG"].to_numpy(), df_bus["VB_ANG"].to_numpy())
        vc = _phasor_from_mag_angle(df_bus["VC_MAG"].to_numpy(), df_bus["VC_ANG"].to_numpy())

        a = np.exp(1j * 2.0 * np.pi / 3.0)
        return (va + a * vb + (a**2) * vc) / 3.0

    raise ValueError(f"Unsupported representation: {representation}")


def _load_voltage_series_for_all_buses(
    scenario_id: str,
    representation: VoltageRepresentation,
) -> dict[int, pd.DataFrame]:
    """
    Load all buses and return:
        {
            bus_id: DataFrame[TIMESTAMP, Event, V_complex]
        }
    """
    metadata = load_synth_metadata(scenario_id=scenario_id)
    buses = sorted(int(bus) for bus in metadata["output_buses"])

    bus_frames: dict[int, pd.DataFrame] = {}

    for bus_id in buses:
        df_bus = load_synth_bus_data(scenario_id=scenario_id, bus_id=bus_id)
        v_complex = _extract_bus_complex_voltage(df_bus, representation)

        bus_frames[bus_id] = pd.DataFrame(
            {
                "TIMESTAMP": df_bus["TIMESTAMP"].to_numpy(),
                "Event": df_bus["Event"].to_numpy(),
                "V_complex": v_complex,
            }
        )

    return bus_frames


def _estimate_hidden_voltages(
    B: np.ndarray,
    buses: list[int],
    pmu_buses: list[int],
    v_pmu: np.ndarray,
    ridge: float = 1e-9,
) -> tuple[list[int], np.ndarray]:
    """
    Estimate hidden voltages with:
        V_h = -B_hh^{-1} B_hp V_p

    Parameters
    ----------
    B
        Full susceptance/Laplacian matrix.
    buses
        Ordered bus list matching B.
    pmu_buses
        Observed buses.
    v_pmu
        Complex PMU voltages with shape (n_timestamps, n_pmu)

    Returns
    -------
    hidden_buses, v_hidden_hat
        hidden_buses: ordered hidden bus ids
        v_hidden_hat: complex estimates with shape (n_timestamps, n_hidden)
    """
    bus_to_idx = {bus: idx for idx, bus in enumerate(buses)}

    pmu_idx = [bus_to_idx[bus] for bus in pmu_buses]
    hidden_buses = [bus for bus in buses if bus not in pmu_buses]
    hidden_idx = [bus_to_idx[bus] for bus in hidden_buses]

    B_hh = B[np.ix_(hidden_idx, hidden_idx)]
    B_hp = B[np.ix_(hidden_idx, pmu_idx)]

    # Small ridge for numeric stability.
    B_hh_reg = B_hh + ridge * np.eye(B_hh.shape[0])

    rhs = -(B_hp @ v_pmu.T)  # (n_hidden, n_timestamps)
    v_hidden_hat = np.linalg.solve(B_hh_reg, rhs).T  # (n_timestamps, n_hidden)

    return hidden_buses, v_hidden_hat


def _summarize_results(
    results_df: pd.DataFrame,
    scenario_id: str,
    representation: str,
    n_total_buses: int,
    n_pmu_buses: int,
    n_hidden_buses: int,
    n_timestamps: int,
) -> YBusBaselineSummary:
    def _mag_rmse(frame: pd.DataFrame) -> float:
        return float(np.sqrt(np.mean(np.square(frame["mag_error"].to_numpy()))))

    def _ang_mae(frame: pd.DataFrame) -> float:
        return float(np.mean(np.abs(frame["angle_error_deg"].to_numpy())))

    overall = results_df
    normal = results_df[results_df["Event"] == 0]
    event = results_df[results_df["Event"] != 0]

    return YBusBaselineSummary(
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


def run_ybus_voltage_baseline(
    scenario_id: str = "SIM_0001",
    representation: VoltageRepresentation = "positive_sequence",
) -> tuple[pd.DataFrame, pd.DataFrame, YBusBaselineSummary]:
    """
    Run a topology-only hidden-bus voltage estimator and compare against
    synthetic ground truth.

    Returns
    -------
    results_df
        Long dataframe with per-bus, per-timestamp errors.
    per_bus_summary_df
        Aggregated metrics per hidden bus.
    summary
        Global summary dataclass.
    """
    metadata = load_synth_metadata(scenario_id=scenario_id)
    pmu_buses = sorted(int(bus) for bus in metadata["pmu_buses"])
    all_buses = sorted(int(bus) for bus in metadata["output_buses"])

    B, buses, _ = _build_susceptance_matrix_from_metadata(metadata)
    bus_frames = _load_voltage_series_for_all_buses(
        scenario_id=scenario_id,
        representation=representation,
    )

    base_bus = pmu_buses[0]
    timestamps = bus_frames[base_bus]["TIMESTAMP"].to_numpy()
    events = bus_frames[base_bus]["Event"].to_numpy()

    v_pmu = np.column_stack(
        [bus_frames[bus_id]["V_complex"].to_numpy() for bus_id in pmu_buses]
    )

    hidden_buses, v_hidden_hat = _estimate_hidden_voltages(
        B=B,
        buses=buses,
        pmu_buses=pmu_buses,
        v_pmu=v_pmu,
    )

    records: list[dict] = []

    for hidden_col_idx, bus_id in enumerate(hidden_buses):
        v_true = bus_frames[bus_id]["V_complex"].to_numpy()
        v_hat = v_hidden_hat[:, hidden_col_idx]

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
                    "v_true_real": float(np.real(v_true[t_idx])),
                    "v_true_imag": float(np.imag(v_true[t_idx])),
                    "v_hat_real": float(np.real(v_hat[t_idx])),
                    "v_hat_imag": float(np.imag(v_hat[t_idx])),
                    "mag_true": float(mag_true[t_idx]),
                    "mag_hat": float(mag_hat[t_idx]),
                    "mag_error": float(mag_hat[t_idx] - mag_true[t_idx]),
                    "mag_abs_error": float(np.abs(mag_hat[t_idx] - mag_true[t_idx])),
                    "angle_true_deg": float(ang_true[t_idx]),
                    "angle_hat_deg": float(ang_hat[t_idx]),
                    "angle_error_deg": float(ang_err[t_idx]),
                    "angle_abs_error_deg": float(np.abs(ang_err[t_idx])),
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
        representation=representation,
        n_total_buses=len(all_buses),
        n_pmu_buses=len(pmu_buses),
        n_hidden_buses=len(hidden_buses),
        n_timestamps=len(timestamps),
    )

    return results_df, per_bus_summary_df, summary