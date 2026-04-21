"""Use case for hardened M6 topology-aware PMU-only state estimation."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.domain.topology import bus_sort_key, bus_token, canonical_bus_name
from src.estimation.state_estimation.measurement_model import FrameMeasurements
from src.estimation.state_estimation.models import EstimationConfig
from src.estimation.state_estimation.per_unit import current_to_pu, voltage_to_pu
from src.estimation.state_estimation.pmu_state_estimator import PmuStateEstimator
from src.estimation.state_estimation.positive_sequence import positive_sequence_from_mag_angle
from src.estimation.state_estimation.priors import build_loadflow_prior
from src.infrastructure.io.metadata_loader import load_pmu_metadata
from src.infrastructure.io.pmu_csv_loader import PmuCsvLoader, REQUIRED_SUFFIXES
from src.infrastructure.io.raw_network_loader import load_network_model_from_raw
from src.metrics.state_estimation_metrics import compute_state_estimation_metrics, residual_diagnostics
from src.reports.state_estimation_report import write_state_estimation_report
from src.simulation.andes_ieee39_runner import run_andes_ieee39_truth
from src.visualization.state_estimation_plots import (
    generate_diagnostic_plots,
    plot_error_summaries,
    plot_estimator_performance,
    plot_selected_bus_voltage,
)


LOGGER = logging.getLogger(__name__)
PMU_DEFAULT = ["BUS39", "BUS29", "BUS10", "BUS22", "BUS19", "BUS2", "BUS5", "BUS6"]


def _create_output_layout(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "config": output_dir / "config",
        "metadata": output_dir / "metadata",
        "intermediate": output_dir / "intermediate",
        "estimated": output_dir / "estimated",
        "truth": output_dir / "truth",
        "metrics": output_dir / "metrics",
        "plots": output_dir / "plots",
        "report": output_dir / "report",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _align_truth_to_network(truth_bus_ids: list[str], truth_v: np.ndarray, network_bus_order: list[str]) -> np.ndarray:
    truth_idx = {canonical_bus_name(b): i for i, b in enumerate(truth_bus_ids)}
    out = np.zeros((truth_v.shape[0], len(network_bus_order)), dtype=complex)
    for j, bus in enumerate(network_bus_order):
        out[:, j] = truth_v[:, truth_idx[bus]] if bus in truth_idx else 1.0 + 0.0j
    return out


def _merged_input_from_pmu_tables(pmu_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for bus, df in pmu_tables.items():
        local = df.copy()
        local = local.rename(
            columns={
                "DATA_PRESENT": f"BUS{bus}_DATA_PRESENT",
                "Event": f"BUS{bus}_Event",
            }
        )
        local["TIMESTAMP_KEY"] = np.round(local["TIMESTAMP"].to_numpy(float) * 1000).astype(int)
        if merged is None:
            merged = local
        else:
            merged = pd.merge(merged, local, on=["TIMESTAMP_KEY"], how="outer", suffixes=("", "_dup"))
            if "TIMESTAMP_x" in merged.columns and "TIMESTAMP_y" in merged.columns:
                merged["TIMESTAMP"] = merged["TIMESTAMP_x"].fillna(merged["TIMESTAMP_y"])
                merged = merged.drop(columns=["TIMESTAMP_x", "TIMESTAMP_y"])
    if merged is None:
        return pd.DataFrame(columns=["TIMESTAMP"])
    dup_cols = [c for c in merged.columns if c.endswith("_dup")]
    if dup_cols:
        merged = merged.drop(columns=dup_cols)
    merged["TIMESTAMP"] = pd.to_numeric(merged["TIMESTAMP"], errors="coerce")
    merged = merged.dropna(subset=["TIMESTAMP"]).drop_duplicates(subset=["TIMESTAMP_KEY"]).sort_values("TIMESTAMP")
    return merged.reset_index(drop=True)


def _required_columns_for_bus(bus: str) -> list[str]:
    token = bus_token(bus)
    return [f"BUS{token}_{s}" for s in REQUIRED_SUFFIXES] + ["DATA_PRESENT"]


def _build_frames_and_audits_from_merged(
    merged: pd.DataFrame,
    expected_pmu_buses: list[str],
    bus_kv_map: dict[str, float],
    base_mva: float,
) -> tuple[list[FrameMeasurements], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[FrameMeasurements] = []
    assembly_rows: list[dict] = []
    solver_input_rows: list[dict] = []
    pos_rows: list[dict] = []
    pu_rows: list[dict] = []
    dropped_rows: list[dict] = []

    for _, row in merged.iterrows():
        ts = float(row["TIMESTAMP"])
        v_by_bus: dict[str, complex] = {}
        i_by_bus: dict[str, complex] = {}
        dropped: dict[str, str] = {}
        valid_buses: list[str] = []
        available_buses: list[str] = []

        for bus in expected_pmu_buses:
            token = bus_token(bus)
            prefix = f"BUS{token}_"
            req = [f"{prefix}{s}" for s in REQUIRED_SUFFIXES]
            data_present_col = f"BUS{token}_DATA_PRESENT"
            has_cols = all(c in row.index for c in req) and (data_present_col in row.index)
            if not has_cols:
                dropped[bus] = "missing_columns"
                dropped_rows.append({"TIMESTAMP": ts, "PMU_BUS": bus, "REASON": "missing_columns"})
                continue
            available_buses.append(bus)
            if int(float(row.get(data_present_col, 0))) != 1:
                dropped[bus] = "data_present_0"
                dropped_rows.append({"TIMESTAMP": ts, "PMU_BUS": bus, "REASON": "data_present_0"})
                continue
            values = [row[c] for c in req]
            if any(pd.isna(v) for v in values):
                dropped[bus] = "nan_measurement"
                dropped_rows.append({"TIMESTAMP": ts, "PMU_BUS": bus, "REASON": "nan_measurement"})
                continue

            v1 = positive_sequence_from_mag_angle(
                np.asarray([row[f"{prefix}VA_MAG"]], dtype=float),
                np.asarray([row[f"{prefix}VA_ANG"]], dtype=float),
                np.asarray([row[f"{prefix}VB_MAG"]], dtype=float),
                np.asarray([row[f"{prefix}VB_ANG"]], dtype=float),
                np.asarray([row[f"{prefix}VC_MAG"]], dtype=float),
                np.asarray([row[f"{prefix}VC_ANG"]], dtype=float),
            )[0]
            i1 = positive_sequence_from_mag_angle(
                np.asarray([row[f"{prefix}IA_MAG"]], dtype=float),
                np.asarray([row[f"{prefix}IA_ANG"]], dtype=float),
                np.asarray([row[f"{prefix}IB_MAG"]], dtype=float),
                np.asarray([row[f"{prefix}IB_ANG"]], dtype=float),
                np.asarray([row[f"{prefix}IC_MAG"]], dtype=float),
                np.asarray([row[f"{prefix}IC_ANG"]], dtype=float),
            )[0]
            kv = float(bus_kv_map.get(bus, 345.0))
            v_pu = complex(voltage_to_pu(np.asarray([v1], dtype=complex), kv)[0])
            i_pu = complex(current_to_pu(np.asarray([i1], dtype=complex), kv, base_mva)[0])
            v_by_bus[bus] = v_pu
            i_by_bus[bus] = i_pu
            valid_buses.append(bus)

            pos_rows.append(
                {
                    "TIMESTAMP": ts,
                    "PMU_BUS": bus,
                    "V1_REAL": float(np.real(v1)),
                    "V1_IMAG": float(np.imag(v1)),
                    "I1_REAL": float(np.real(i1)),
                    "I1_IMAG": float(np.imag(i1)),
                    "V1_MAG": float(np.abs(v1)),
                    "V1_ANG_DEG": float(np.rad2deg(np.angle(v1))),
                    "I1_MAG": float(np.abs(i1)),
                    "I1_ANG_DEG": float(np.rad2deg(np.angle(i1))),
                    "INPUT_WAS_VALID": True,
                    "CONVERSION_STATUS": "ok",
                    "NOTES": "",
                }
            )
            pu_rows.append(
                {
                    "PMU_BUS": bus,
                    "KV_BASE": kv,
                    "MVA_BASE": float(base_mva),
                    "I_BASE": float(base_mva * 1e6 / (np.sqrt(3) * kv * 1e3)),
                    "SAMPLE_TIMESTAMP": ts,
                    "V1_MAG_RAW": float(np.abs(v1)),
                    "V1_MAG_PU": float(np.abs(v_pu)),
                    "I1_MAG_RAW": float(np.abs(i1)),
                    "I1_MAG_PU": float(np.abs(i_pu)),
                    "CONVERSION_STATUS": "ok",
                }
            )

        frame = FrameMeasurements(
            timestamp=ts,
            voltage_by_bus=v_by_bus,
            current_by_bus=i_by_bus,
            data_present_count=len(valid_buses),
            expected_pmu_buses=expected_pmu_buses,
            valid_pmu_buses=valid_buses,
            dropped_reasons=dropped,
        )
        frames.append(frame)
        z_dim = len(valid_buses) * 4
        assembly_rows.append(
            {
                "TIMESTAMP": ts,
                "EXPECTED_PMU_COUNT": len(expected_pmu_buses),
                "AVAILABLE_PMU_COUNT": len(available_buses),
                "VALID_PMU_COUNT": len(valid_buses),
                "SOLVER_PMU_COUNT": len(valid_buses),
                "EXPECTED_PMUS": "|".join(sorted(expected_pmu_buses, key=bus_sort_key)),
                "AVAILABLE_PMUS": "|".join(sorted(available_buses, key=bus_sort_key)),
                "VALID_PMUS": "|".join(sorted(valid_buses, key=bus_sort_key)),
                "SOLVER_PMUS": "|".join(sorted(valid_buses, key=bus_sort_key)),
                "DROPPED_PMUS": "|".join(sorted(dropped.keys(), key=bus_sort_key)),
                "DROP_REASONS": "|".join(f"{k}:{v}" for k, v in dropped.items()),
                "Z_VECTOR_DIM": z_dim,
                "EXPECTED_Z_VECTOR_DIM": len(expected_pmu_buses) * 4,
                "STATUS": "ok" if len(valid_buses) > 0 else "no_valid_pmu",
            }
        )
        meas_idx = 0
        for bus in expected_pmu_buses:
            used = bus in valid_buses
            reason = dropped.get(bus, "")
            for meas_type, value in [
                ("V_REAL", np.real(v_by_bus.get(bus, np.nan + 0j))),
                ("V_IMAG", np.imag(v_by_bus.get(bus, np.nan + 0j))),
                ("I_REAL", np.real(i_by_bus.get(bus, np.nan + 0j))),
                ("I_IMAG", np.imag(i_by_bus.get(bus, np.nan + 0j))),
            ]:
                solver_input_rows.append(
                    {
                        "TIMESTAMP": ts,
                        "PMU_BUS": bus,
                        "MEAS_TYPE": meas_type,
                        "RAW_VALUE": value,
                        "USED_VALUE": value if used else np.nan,
                        "IS_AVAILABLE": bus in available_buses,
                        "DATA_PRESENT": 1 if bus in valid_buses else 0,
                        "ENTERED_SOLVER": used,
                        "DROPPED_REASON": reason,
                        "MASK_STATUS": "used" if used else "dropped",
                        "MEAS_INDEX_IN_ZK": meas_idx if used else -1,
                    }
                )
                if used:
                    meas_idx += 1

    return (
        frames,
        pd.DataFrame(assembly_rows),
        pd.DataFrame(solver_input_rows),
        pd.DataFrame(pos_rows),
        pd.DataFrame(pu_rows).drop_duplicates(subset=["PMU_BUS", "SAMPLE_TIMESTAMP"]),
        pd.DataFrame(dropped_rows),
    )


def _result_to_dataframe(
    timestamps: np.ndarray,
    bus_order: list[str],
    v_est_pu: np.ndarray,
    pmu_buses: set[str],
    frame_diag: pd.DataFrame,
) -> pd.DataFrame:
    frame_map = frame_diag.set_index("TIMESTAMP") if not frame_diag.empty else pd.DataFrame()
    rows: list[dict] = []
    for ti, t in enumerate(np.asarray(timestamps, dtype=float)):
        frame = frame_map.loc[t] if (not frame_map.empty and t in frame_map.index) else None
        for bi, bus in enumerate(bus_order):
            val = complex(v_est_pu[ti, bi])
            rows.append(
                {
                    "TIMESTAMP": float(t),
                    "BUS": bus,
                    "IS_PMU_BUS": bool(bus in pmu_buses),
                    "V_EST_REAL_PU": float(np.real(val)),
                    "V_EST_IMAG_PU": float(np.imag(val)),
                    "V_EST_MAG_PU": float(np.abs(val)),
                    "V_EST_ANG_DEG": float(np.rad2deg(np.angle(val))),
                    "N_VALID_PMUS_USED": int(frame["N_PMUS_USED_IN_SOLVER"]) if frame is not None else 0,
                    "PMU_BUSES_USED": str(frame["PMU_BUSES_USED_IN_SOLVER"]) if frame is not None else "",
                    "FRAME_SOLVER_STATUS": str(frame["SOLVER_STATUS"]) if frame is not None else "",
                    "OPTIONAL_CONFIDENCE_OR_RESIDUAL": float(frame["RESIDUAL_NORM"]) if frame is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _build_bus_variability_summary(estimation_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for bus, grp in estimation_df.groupby("BUS"):
        mag = grp["V_EST_MAG_PU"].to_numpy(float)
        ang = grp["V_EST_ANG_DEG"].to_numpy(float)
        rows.append(
            {
                "BUS": bus,
                "IS_PMU_BUS": bool(grp["IS_PMU_BUS"].iloc[0]),
                "STD_V_MAG": float(np.std(mag)),
                "STD_V_ANG_DEG": float(np.std(ang)),
                "RANGE_V_MAG": float(np.max(mag) - np.min(mag)),
                "RANGE_V_ANG_DEG": float(np.max(ang) - np.min(ang)),
                "MEAN_V_MAG": float(np.mean(mag)),
                "MEAN_V_ANG_DEG": float(np.mean(ang)),
                "IS_NEARLY_FROZEN": bool(np.std(mag) < 1e-4 and np.std(ang) < 1e-2),
                "NOTES": "",
            }
        )
    return pd.DataFrame(rows).sort_values("BUS", key=lambda s: s.map(bus_sort_key))


def run_m6_topology_aware_state_estimation_use_case(
    raw_path: str | Path,
    pmu_location_path: str | Path,
    output_dir: str | Path,
    pmu_data_dir: str | Path | None = None,
    use_andes_truth: bool = False,
    diagnostic: bool = True,
    start_time: float | None = None,
    end_time: float | None = None,
    stride: int = 1,
    lambda_reg: float = 5e-2,
    mu_reg: float = 1e-3,
) -> dict[str, Any]:
    """Run M6 state estimation with hardening diagnostics and optional ANDES truth validation."""
    out = Path(output_dir)
    layout = _create_output_layout(out)
    pmu_meta = load_pmu_metadata(pmu_location_path)
    network = load_network_model_from_raw(raw_path=raw_path, rated_frequency_hz=float(pmu_meta.get("rated_frequency_hz", 60.0)))
    expected_pmu_buses = sorted(
        {canonical_bus_name(entry.get("bus_label_canonical", "")) for entry in pmu_meta.get("pmu_map", [])} or set(PMU_DEFAULT),
        key=bus_sort_key,
    )

    normalized_rows: list[dict] = []
    for b in network.bus_order:
        normalized_rows.append({"ORIGINAL_SOURCE": "raw", "ORIGINAL_NAME": b, "CANONICAL_NAME": canonical_bus_name(b), "SOURCE_TYPE": "raw"})
    for entry in pmu_meta.get("pmu_map", []):
        raw = entry.get("bus_label_canonical", "")
        normalized_rows.append({"ORIGINAL_SOURCE": "metadata", "ORIGINAL_NAME": raw, "CANONICAL_NAME": canonical_bus_name(raw), "SOURCE_TYPE": "metadata"})
    pd.DataFrame(normalized_rows).to_csv(layout["metadata"] / "normalized_bus_mapping.csv", index=False)

    prior = build_loadflow_prior(network.bus_order, pmu_meta)
    truth_v = None
    truth_t = None
    merged = pd.DataFrame()
    coverage_rows: list[dict] = []
    merged_nulls = pd.DataFrame()
    frames: list[FrameMeasurements]
    assembly_df = pd.DataFrame()
    solver_inputs_df = pd.DataFrame()
    positive_seq_df = pd.DataFrame()
    pu_audit_df = pd.DataFrame()
    dropped_df = pd.DataFrame(columns=["TIMESTAMP", "PMU_BUS", "REASON"])

    if use_andes_truth:
        truth = run_andes_ieee39_truth(tf=end_time if end_time is not None else 10.0, tstep=1.0 / 30.0, stride=max(1, int(stride)))
        truth_t = truth.timestamps
        truth_v = _align_truth_to_network(truth.bus_ids, truth.voltage_complex_pu, network.bus_order)
        i_truth = (network.ybus @ truth_v.T).T
        frames = []
        for ti, ts in enumerate(truth_t):
            v_by = {}
            i_by = {}
            for bus in expected_pmu_buses:
                idx = network.bus_order.index(bus) if bus in network.bus_order else None
                if idx is None:
                    continue
                v_by[bus] = complex(truth_v[ti, idx])
                i_by[bus] = complex(i_truth[ti, idx])
            frames.append(
                FrameMeasurements(
                    timestamp=float(ts),
                    voltage_by_bus=v_by,
                    current_by_bus=i_by,
                    data_present_count=len(v_by),
                    expected_pmu_buses=expected_pmu_buses,
                    valid_pmu_buses=sorted(v_by.keys(), key=bus_sort_key),
                    dropped_reasons={},
                )
            )
            assembly_df = pd.concat(
                [
                    assembly_df,
                    pd.DataFrame(
                        [
                            {
                                "TIMESTAMP": float(ts),
                                "EXPECTED_PMU_COUNT": len(expected_pmu_buses),
                                "AVAILABLE_PMU_COUNT": len(v_by),
                                "VALID_PMU_COUNT": len(v_by),
                                "SOLVER_PMU_COUNT": len(v_by),
                                "EXPECTED_PMUS": "|".join(expected_pmu_buses),
                                "AVAILABLE_PMUS": "|".join(sorted(v_by.keys(), key=bus_sort_key)),
                                "VALID_PMUS": "|".join(sorted(v_by.keys(), key=bus_sort_key)),
                                "SOLVER_PMUS": "|".join(sorted(v_by.keys(), key=bus_sort_key)),
                                "DROPPED_PMUS": "",
                                "DROP_REASONS": "",
                                "Z_VECTOR_DIM": len(v_by) * 4,
                                "EXPECTED_Z_VECTOR_DIM": len(expected_pmu_buses) * 4,
                                "STATUS": "ok",
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        for bus in expected_pmu_buses:
            coverage_rows.append(
                {
                    "PMU_BUS_EXPECTED": bus,
                    "FOUND_IN_RAW_NETWORK": bus in network.bus_order,
                    "FOUND_IN_BUS_INDEX_MAP": bus in {b for b in network.bus_order},
                    "FOUND_IN_PMU_METADATA": True,
                    "FOUND_IN_MERGED_INPUT": True,
                    "HAS_VA": True,
                    "HAS_VB": True,
                    "HAS_VC": True,
                    "HAS_IA": True,
                    "HAS_IB": True,
                    "HAS_IC": True,
                    "HAS_FREQ": True,
                    "HAS_ROCOF": True,
                    "HAS_DATA_PRESENT": True,
                    "SOURCE_COLUMNS": "ANDES_TRUTH",
                    "COVERAGE_STATUS": "ok",
                    "NOTES": "synthetic PMU from truth",
                }
            )
    else:
        if pmu_data_dir is None:
            raise ValueError("pmu_data_dir is required when use_andes_truth=False")
        loader = PmuCsvLoader(pmu_data_dir)
        pmu_tables: dict[str, pd.DataFrame] = {}
        for bus in expected_pmu_buses:
            token = bus_token(bus)
            try:
                table = loader.load_bus(token)
                pmu_tables[token] = table
                for col in table.columns:
                    normalized_rows.append({"ORIGINAL_SOURCE": "csv", "ORIGINAL_NAME": col, "CANONICAL_NAME": canonical_bus_name(col), "SOURCE_TYPE": "csv"})
            except Exception as exc:
                LOGGER.warning("PMU %s not loaded: %s", bus, exc)
        if not pmu_tables:
            raise RuntimeError(f"No PMU CSVs loaded from {pmu_data_dir}")
        merged = _merged_input_from_pmu_tables(pmu_tables)
        merged_nulls = pd.DataFrame({"COLUMN": merged.columns, "NULL_COUNT": [int(merged[c].isna().sum()) for c in merged.columns]})
        merged_nulls.to_csv(layout["metrics"] / "merged_input_nulls.csv", index=False)
        (layout["metadata"] / "merged_input_audit.json").write_text(
            json.dumps(
                {
                    "row_count": int(len(merged)),
                    "timestamp_count": int(merged["TIMESTAMP"].nunique() if "TIMESTAMP" in merged.columns else 0),
                    "column_count": int(merged.shape[1]),
                    "timestamp_min": float(merged["TIMESTAMP"].min()) if len(merged) else None,
                    "timestamp_max": float(merged["TIMESTAMP"].max()) if len(merged) else None,
                    "duplicate_timestamps": int(merged["TIMESTAMP"].duplicated().sum()) if "TIMESTAMP" in merged.columns else 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        frames, assembly_df, solver_inputs_df, positive_seq_df, pu_audit_df, dropped_df = _build_frames_and_audits_from_merged(
            merged=merged,
            expected_pmu_buses=expected_pmu_buses,
            bus_kv_map=network.bus_kv_map,
            base_mva=network.base_mva,
        )
        if start_time is not None:
            frames = [f for f in frames if f.timestamp >= start_time]
        if end_time is not None:
            frames = [f for f in frames if f.timestamp <= end_time]
        if stride > 1:
            frames = frames[::stride]
            ts_keep = {f.timestamp for f in frames}
            assembly_df = assembly_df[assembly_df["TIMESTAMP"].isin(ts_keep)]
            solver_inputs_df = solver_inputs_df[solver_inputs_df["TIMESTAMP"].isin(ts_keep)]
            positive_seq_df = positive_seq_df[positive_seq_df["TIMESTAMP"].isin(ts_keep)]
            dropped_df = dropped_df[dropped_df["TIMESTAMP"].isin(ts_keep)]
        for bus in expected_pmu_buses:
            token = bus_token(bus)
            cols = [f"BUS{token}_{s}" for s in REQUIRED_SUFFIXES]
            has = [c in merged.columns for c in cols]
            coverage_rows.append(
                {
                    "PMU_BUS_EXPECTED": bus,
                    "FOUND_IN_RAW_NETWORK": bus in network.bus_order,
                    "FOUND_IN_BUS_INDEX_MAP": bus in network.bus_order,
                    "FOUND_IN_PMU_METADATA": bus in expected_pmu_buses,
                    "FOUND_IN_MERGED_INPUT": all(c in merged.columns for c in cols),
                    "HAS_VA": f"BUS{token}_VA_MAG" in merged.columns and f"BUS{token}_VA_ANG" in merged.columns,
                    "HAS_VB": f"BUS{token}_VB_MAG" in merged.columns and f"BUS{token}_VB_ANG" in merged.columns,
                    "HAS_VC": f"BUS{token}_VC_MAG" in merged.columns and f"BUS{token}_VC_ANG" in merged.columns,
                    "HAS_IA": f"BUS{token}_IA_MAG" in merged.columns and f"BUS{token}_IA_ANG" in merged.columns,
                    "HAS_IB": f"BUS{token}_IB_MAG" in merged.columns and f"BUS{token}_IB_ANG" in merged.columns,
                    "HAS_IC": f"BUS{token}_IC_MAG" in merged.columns and f"BUS{token}_IC_ANG" in merged.columns,
                    "HAS_FREQ": f"BUS{token}_Freq" in merged.columns,
                    "HAS_ROCOF": f"BUS{token}_ROCOF" in merged.columns,
                        "HAS_DATA_PRESENT": f"BUS{token}_DATA_PRESENT" in merged.columns,
                    "SOURCE_COLUMNS": "|".join([c for c in cols if c in merged.columns]),
                    "COVERAGE_STATUS": "ok" if any(has) else "missing",
                    "NOTES": "",
                }
            )

    pd.DataFrame(coverage_rows).to_csv(layout["metadata"] / "pmu_coverage_audit.csv", index=False)
    pd.DataFrame(normalized_rows).drop_duplicates().to_csv(layout["metadata"] / "normalized_bus_mapping.csv", index=False)
    (layout["metadata"] / "pmu_bus_mapping.json").write_text(json.dumps(expected_pmu_buses, indent=2), encoding="utf-8")
    (layout["metadata"] / "bus_index_map.json").write_text(
        json.dumps({bus: i for i, bus in enumerate(network.bus_order)}, indent=2),
        encoding="utf-8",
    )
    (layout["metadata"] / "network_summary.json").write_text(
        json.dumps(
            {"bus_count": len(network.bus_order), "base_mva": network.base_mva, "rated_frequency_hz": network.rated_frequency_hz},
            indent=2,
        ),
        encoding="utf-8",
    )
    if diagnostic:
        assembly_df.to_csv(layout["intermediate"] / "measurement_assembly_audit.csv", index=False)
        solver_inputs_df.to_csv(layout["intermediate"] / "solver_inputs_long.csv", index=False)
        positive_seq_df.to_csv(layout["intermediate"] / "positive_sequence_audit.csv", index=False)
        pu_audit_df.to_csv(layout["intermediate"] / "per_unit_audit.csv", index=False)

    if not frames:
        raise RuntimeError("No estimator frames were built after filtering.")

    estimator = PmuStateEstimator(network=network, config=EstimationConfig(lambda_reg=lambda_reg, mu_reg=mu_reg))
    estimation = estimator.estimate_timeseries(frames=frames, x_prior=prior)
    diag_rows = [asdict(d) for d in estimation.diagnostics]
    frame_diag = pd.DataFrame(diag_rows)
    frame_diag["EXPECTED_PMU_BUSES"] = "|".join(expected_pmu_buses)
    frame_diag["AVAILABLE_PMU_BUSES_FROM_MERGED_INPUT"] = frame_diag["pmu_buses_used"].apply(lambda v: "|".join(v) if isinstance(v, list) else "")
    frame_diag["VALID_PMU_BUSES_AFTER_MASK"] = frame_diag["pmu_buses_used"].apply(lambda v: "|".join(v) if isinstance(v, list) else "")
    frame_diag["PMU_BUSES_USED_IN_SOLVER"] = frame_diag["pmu_buses_used"].apply(lambda v: "|".join(v) if isinstance(v, list) else "")
    frame_diag["PMU_BUSES_EXCLUDED_FROM_SOLVER"] = frame_diag["pmu_buses_excluded"].apply(lambda v: "|".join(v) if isinstance(v, list) else "")
    frame_diag["DROPPED_PMU_BUSES"] = frame_diag["dropped_reasons"].apply(lambda d: "|".join(sorted(d.keys())) if isinstance(d, dict) else "")
    frame_diag["DROPPED_REASONS"] = frame_diag["dropped_reasons"].apply(lambda d: "|".join(f"{k}:{v}" for k, v in d.items()) if isinstance(d, dict) else "")
    frame_diag["N_EXPECTED_PMUS"] = len(expected_pmu_buses)
    frame_diag["N_AVAILABLE_PMUS"] = frame_diag["n_pmus_used"]
    frame_diag["N_VALID_PMUS"] = frame_diag["n_valid_pmus"]
    frame_diag["N_PMUS_USED_IN_SOLVER"] = frame_diag["n_pmus_used"]
    frame_diag["HAS_MISSING_PMUS"] = frame_diag["N_PMUS_USED_IN_SOLVER"] < len(expected_pmu_buses)
    frame_diag["Z_VECTOR_DIM"] = frame_diag["measurement_count"]
    frame_diag["X_VECTOR_DIM"] = len(network.bus_order) * 2
    frame_diag["SOLVER_STATUS"] = frame_diag["solver_status"]
    frame_diag["SOLVER_MESSAGE"] = frame_diag["solver_message"]
    frame_diag["CONDITION_NUMBER_OR_ESTIMATE"] = frame_diag["matrix_condition_number"]
    frame_diag["RESIDUAL_NORM"] = frame_diag["residual_norm"]
    frame_diag["DATA_TERM_VALUE"] = frame_diag["data_term_value"]
    frame_diag["TEMPORAL_PRIOR_TERM_VALUE"] = frame_diag["temporal_prior_term_value"]
    frame_diag["LOADFLOW_PRIOR_TERM_VALUE"] = frame_diag["loadflow_prior_term_value"]
    frame_diag["TOTAL_OBJECTIVE_VALUE"] = frame_diag["total_objective_value"]
    frame_diag = frame_diag.rename(columns={"timestamp": "TIMESTAMP", "frame_index": "FRAME_INDEX"})
    frame_diag.to_csv(layout["metrics"] / "frame_diagnostics.csv", index=False)
    frame_diag.to_json(layout["metrics"] / "frame_diagnostics.jsonl", orient="records", lines=True)
    frame_diag[
        [
            "TIMESTAMP",
            "N_VALID_PMUS",
            "N_PMUS_USED_IN_SOLVER",
            "voltage_measurement_count",
            "current_measurement_count",
            "RESIDUAL_NORM",
            "DATA_TERM_VALUE",
            "TEMPORAL_PRIOR_TERM_VALUE",
            "LOADFLOW_PRIOR_TERM_VALUE",
            "TOTAL_OBJECTIVE_VALUE",
            "CONDITION_NUMBER_OR_ESTIMATE",
            "SOLVER_STATUS",
        ]
    ].rename(
        columns={
            "voltage_measurement_count": "VOLTAGE_MEASUREMENT_COUNT",
            "current_measurement_count": "CURRENT_MEASUREMENT_COUNT",
        }
    ).to_csv(layout["metrics"] / "residual_diagnostics.csv", index=False)

    estimation_df = _result_to_dataframe(
        timestamps=estimation.timestamps,
        bus_order=estimation.bus_order,
        v_est_pu=estimation.voltage_estimates_pu,
        pmu_buses=set(expected_pmu_buses),
        frame_diag=frame_diag,
    )
    estimation_df.to_csv(layout["estimated"] / "estimated_bus_states.csv", index=False)
    try:
        estimation_df.to_parquet(layout["estimated"] / "estimated_bus_states.parquet", index=False)
    except Exception:
        LOGGER.warning("Parquet export skipped (engine unavailable).")

    variability_df = _build_bus_variability_summary(estimation_df)
    variability_df.to_csv(layout["metrics"] / "bus_variability_summary.csv", index=False)

    pmu_fit_rows: list[dict] = []
    for _, rec in estimation_df[estimation_df["IS_PMU_BUS"]].iterrows():
        ts = float(rec["TIMESTAMP"])
        bus = rec["BUS"]
        frame_match = next((f for f in frames if abs(f.timestamp - ts) < 1e-9), None)
        if frame_match is not None:
            raw_meas = frame_match.voltage_by_bus.get(bus, np.nan + 1j * np.nan)
        else:
            raw_meas = np.nan + 1j * np.nan
        meas = complex(raw_meas) if raw_meas is not None else complex(np.nan, np.nan)
        est = complex(rec["V_EST_REAL_PU"] + 1j * rec["V_EST_IMAG_PU"])
        pmu_fit_rows.append(
            {
                "TIMESTAMP": ts,
                "PMU_BUS": bus,
                "USED_IN_SOLVER": bool(frame_match is not None and bus in frame_match.voltage_by_bus),
                "V_MEAS_REAL": float(np.real(meas)),
                "V_MEAS_IMAG": float(np.imag(meas)),
                "V_EST_REAL": float(np.real(est)),
                "V_EST_IMAG": float(np.imag(est)),
                "V_MEAS_MAG": float(np.abs(meas)),
                "V_EST_MAG": float(np.abs(est)),
                "V_MEAS_ANG_DEG": float(np.rad2deg(np.angle(meas))),
                "V_EST_ANG_DEG": float(np.rad2deg(np.angle(est))),
                "ABS_COMPLEX_ERROR": float(np.abs(est - meas)),
                "MAG_ERROR": float(np.abs(np.abs(est) - np.abs(meas))),
                "ANGLE_ERROR_DEG": float(((np.rad2deg(np.angle(est)) - np.rad2deg(np.angle(meas)) + 180.0) % 360.0) - 180.0),
                "IS_VALID_MEAS": bool(np.isfinite(np.real(meas)) and np.isfinite(np.imag(meas))),
                "NOTES": "",
            }
        )
    pmu_fit_df = pd.DataFrame(pmu_fit_rows)
    pmu_fit_df.to_csv(layout["metrics"] / "pmu_fit_diagnostics.csv", index=False)

    global_metrics: dict[str, Any] = {}
    per_bus_metrics = pd.DataFrame()
    truth_csv_path: str | None = None
    if truth_v is not None and truth_t is not None:
        truth_map = {float(t): i for i, t in enumerate(np.asarray(truth_t, dtype=float))}
        idx = np.asarray([truth_map[float(t)] for t in estimation.timestamps if float(t) in truth_map], dtype=int)
        if len(idx) == len(estimation.timestamps):
            truth_for_est = truth_v[idx, :]
            per_bus_metrics, global_metrics = compute_state_estimation_metrics(
                timestamps=estimation.timestamps,
                bus_order=network.bus_order,
                v_est_pu=estimation.voltage_estimates_pu,
                v_true_pu=truth_for_est,
                pmu_buses=set(expected_pmu_buses),
            )
            global_metrics.update(
                {
                    "pmu_bus_count": len(expected_pmu_buses),
                    "run_type": "andes_validation",
                }
            )
            per_bus_metrics.to_csv(layout["metrics"] / "per_bus_metrics.csv", index=False)
            (layout["metrics"] / "global_metrics.json").write_text(json.dumps(global_metrics, indent=2), encoding="utf-8")
            truth_rows: list[dict] = []
            for ti, ts in enumerate(estimation.timestamps):
                for bi, bus in enumerate(network.bus_order):
                    val = truth_for_est[ti, bi]
                    truth_rows.append(
                        {
                            "TIMESTAMP": float(ts),
                            "BUS": bus,
                            "V_TRUE_REAL_PU": float(np.real(val)),
                            "V_TRUE_IMAG_PU": float(np.imag(val)),
                            "V_TRUE_MAG_PU": float(np.abs(val)),
                            "V_TRUE_ANG_DEG": float(np.rad2deg(np.angle(val))),
                        }
                    )
            truth_df = pd.DataFrame(truth_rows)
            truth_path = layout["truth"] / "andes_truth_bus_states.csv"
            truth_df.to_csv(truth_path, index=False)
            truth_csv_path = str(truth_path)

            selected_buses = sorted(expected_pmu_buses, key=bus_sort_key)[:4]
            plot_selected_bus_voltage(estimation.timestamps, network.bus_order, estimation.voltage_estimates_pu, truth_for_est, selected_buses, layout["plots"])
            plot_error_summaries(estimation.timestamps, network.bus_order, estimation.voltage_estimates_pu, truth_for_est, per_bus_metrics, frame_diag, layout["plots"])
            t = estimation.timestamps
            fig, ax = plt.subplots(figsize=(10, 4))
            for bus in selected_buses:
                bi = network.bus_order.index(bus)
                ax.plot(t, np.abs(truth_for_est[:, bi]), label=f"{bus} true")
                ax.plot(t, np.abs(estimation.voltage_estimates_pu[:, bi]), "--", label=f"{bus} est")
            ax.legend(fontsize=8, ncol=2)
            ax.set_title("Selected Bus Voltage Magnitude")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("|V| p.u.")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(layout["plots"] / "selected_bus_voltage_mag.png", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(10, 4))
            for bus in selected_buses:
                bi = network.bus_order.index(bus)
                ax.plot(t, np.rad2deg(np.angle(truth_for_est[:, bi])), label=f"{bus} true")
                ax.plot(t, np.rad2deg(np.angle(estimation.voltage_estimates_pu[:, bi])), "--", label=f"{bus} est")
            ax.legend(fontsize=8, ncol=2)
            ax.set_title("Selected Bus Voltage Angle")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Angle (deg)")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            fig.savefig(layout["plots"] / "selected_bus_voltage_angle.png", dpi=150)
            plt.close(fig)

    performance_summary = {
        "pmu_fit_abs_error_mean": float(pmu_fit_df["ABS_COMPLEX_ERROR"].mean()) if not pmu_fit_df.empty else np.nan,
        "pmu_fit_abs_error_p95": float(pmu_fit_df["ABS_COMPLEX_ERROR"].quantile(0.95)) if not pmu_fit_df.empty else np.nan,
        "frame_residual_mean": float(frame_diag["RESIDUAL_NORM"].mean()) if not frame_diag.empty else np.nan,
        "frame_residual_p95": float(frame_diag["RESIDUAL_NORM"].quantile(0.95)) if not frame_diag.empty else np.nan,
        "mean_pmus_used": float(frame_diag["N_PMUS_USED_IN_SOLVER"].mean()) if not frame_diag.empty else np.nan,
        "min_pmus_used": float(frame_diag["N_PMUS_USED_IN_SOLVER"].min()) if not frame_diag.empty else np.nan,
    }
    performance_df = pd.DataFrame([performance_summary])
    performance_df.to_csv(layout["metrics"] / "estimator_performance_summary.csv", index=False)
    (layout["metrics"] / "estimator_performance_summary.json").write_text(
        json.dumps(performance_summary, indent=2),
        encoding="utf-8",
    )

    diag_plot_paths = generate_diagnostic_plots(
        frame_df=frame_diag,
        pmu_fit_df=pmu_fit_df,
        residual_df=frame_diag,
        variability_df=variability_df,
        dropped_df=dropped_df,
        out_dir=layout["plots"],
    )
    perf_plot_paths = plot_estimator_performance(
        per_bus_metrics=per_bus_metrics,
        global_metrics=global_metrics,
        pmu_fit_df=pmu_fit_df,
        out_dir=layout["plots"],
    )
    diag_plot_paths.update(perf_plot_paths)
    if not (layout["plots"] / "selected_bus_voltage_mag.png").exists():
        chosen = sorted(expected_pmu_buses, key=bus_sort_key)[:4]
        t = estimation.timestamps
        fig, ax = plt.subplots(figsize=(10, 4))
        for bus in chosen:
            bi = network.bus_order.index(bus)
            ax.plot(t, np.abs(estimation.voltage_estimates_pu[:, bi]), label=f"{bus} est")
        ax.legend(fontsize=8, ncol=2)
        ax.set_title("Selected Bus Voltage Magnitude (Estimate)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("|V| p.u.")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(layout["plots"] / "selected_bus_voltage_mag.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        for bus in chosen:
            bi = network.bus_order.index(bus)
            ax.plot(t, np.rad2deg(np.angle(estimation.voltage_estimates_pu[:, bi])), label=f"{bus} est")
        ax.legend(fontsize=8, ncol=2)
        ax.set_title("Selected Bus Voltage Angle (Estimate)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(layout["plots"] / "selected_bus_voltage_angle.png", dpi=150)
        plt.close(fig)

    (layout["report"] / "diagnostic_report.md").write_text(
        "\n".join(
            [
                "# M6 Diagnostic Report",
                f"- expected PMUs: {', '.join(expected_pmu_buses)}",
                f"- frames processed: {len(frames)}",
                f"- mean PMUs used in solver: {frame_diag['N_PMUS_USED_IN_SOLVER'].mean():.2f}",
                f"- min PMUs used in solver: {frame_diag['N_PMUS_USED_IN_SOLVER'].min()}",
                "- Root-cause check: PMU timestamp alignment is now based on rounded TIMESTAMP_KEY merge, avoiding strict float-index dropouts.",
            ]
        ),
        encoding="utf-8",
    )
    (layout["report"] / "validation_report.md").write_text(
        "\n".join(
            [
                "# M6 Validation Report",
                f"- use_andes_truth: {use_andes_truth}",
                f"- pmus used per frame (mean): {frame_diag['N_PMUS_USED_IN_SOLVER'].mean():.2f}",
                f"- pmus used per frame (min): {frame_diag['N_PMUS_USED_IN_SOLVER'].min()}",
                f"- global metrics available: {bool(global_metrics)}",
                f"- non-PMU buses frozen count: {int(variability_df['IS_NEARLY_FROZEN'].sum())}",
            ]
        ),
        encoding="utf-8",
    )

    run_cfg = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(Path(raw_path)),
        "pmu_location_path": str(Path(pmu_location_path)),
        "pmu_data_dir": str(Path(pmu_data_dir)) if pmu_data_dir is not None else None,
        "use_andes_truth": bool(use_andes_truth),
        "diagnostic": bool(diagnostic),
        "start_time": start_time,
        "end_time": end_time,
        "stride": int(stride),
        "lambda_reg": float(lambda_reg),
        "mu_reg": float(mu_reg),
        "expected_pmu_buses": expected_pmu_buses,
    }
    (layout["config"] / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    summary = {
        "output_dir": str(out),
        "timestamp_count": int(len(frames)),
        "bus_count": int(len(network.bus_order)),
        "pmu_bus_count": int(len(expected_pmu_buses)),
        "use_andes_truth": bool(use_andes_truth),
        "min_pmus_used": int(frame_diag["N_PMUS_USED_IN_SOLVER"].min()),
        "mean_pmus_used": float(frame_diag["N_PMUS_USED_IN_SOLVER"].mean()),
        "estimated_csv": str(layout["estimated"] / "estimated_bus_states.csv"),
        "truth_csv": truth_csv_path,
        "global_metrics": global_metrics,
        "diagnostic_plots": diag_plot_paths,
    }
    report_files = write_state_estimation_report(
        output_dir=layout["report"],
        summary=summary,
        global_metrics=global_metrics,
        plot_paths=diag_plot_paths,
    )
    summary.update(report_files)
    return summary
