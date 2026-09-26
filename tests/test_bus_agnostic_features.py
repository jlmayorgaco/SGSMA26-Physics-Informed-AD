from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_factory.dynamic_feature_extractor_v3 import extract_dynamic_features_from_frames
from src.features.bus_agnostic import (
    aggregate_pmu_features,
    graph_feature_block,
    pmu_severity_vector,
    pmu_window_features,
    sequence_components,
    unwrap_degrees,
    zbus_diffusion_scores,
)
from src.features.pmu_discovery import infer_pmu_buses_from_columns, infer_pmu_buses_from_frames
from pipelines.p11_train_bus_agnostic_full import _build_bus_agnostic_table
from pipelines.p17_train_event4_load_ranker import _candidate_features, _pmu_load_severity


def _frame(bus: int, spike: bool = False, missing: bool = False) -> pd.DataFrame:
    t = np.round(np.arange(0.0, 6.0, 1.0 / 30.0), 3)
    step = (t > 3.0).astype(float)
    data = {"TIMESTAMP": t, "DATA_PRESENT": 0 if missing else 1}
    for phase, angle in {"A": 0.0, "B": -120.0, "C": 120.0}.items():
        vmag = 100.0 + 0.05 * np.sin(t)
        imag = 10.0 + 0.02 * np.cos(t)
        if spike and phase == "A":
            vmag = vmag - 20.0 * step
            imag = imag + 5.0 * step
        data[f"BUS{bus}_V{phase}_MAG"] = np.nan if missing else vmag
        data[f"BUS{bus}_V{phase}_ANG"] = np.nan if missing else angle + 0.01 * t
        data[f"BUS{bus}_I{phase}_MAG"] = np.nan if missing else imag
        data[f"BUS{bus}_I{phase}_ANG"] = np.nan if missing else angle - 20.0 + 0.01 * t
    data[f"BUS{bus}_Freq"] = np.nan if missing else 60.0 + 0.01 * step
    data[f"BUS{bus}_ROCOF"] = np.nan if missing else np.gradient(data[f"BUS{bus}_Freq"], 1.0 / 30.0)
    return pd.DataFrame(data)


def test_unwrap_degrees_removes_boundary_jump() -> None:
    values = np.asarray([170.0, 175.0, -179.0, -175.0])
    unwrapped = unwrap_degrees(values)
    assert np.max(np.abs(np.diff(unwrapped))) < 10.0


def test_sequence_components_balanced_positive_sequence() -> None:
    a = np.exp(1j * 2.0 * np.pi / 3.0)
    phasors = {"A": np.ones(4), "B": (a**2) * np.ones(4), "C": a * np.ones(4)}
    x0, x1, x2 = sequence_components(phasors)
    assert np.allclose(np.abs(x1), 1.0)
    assert np.allclose(np.abs(x0), 0.0, atol=1e-12)
    assert np.allclose(np.abs(x2), 0.0, atol=1e-12)


def test_pmu_window_features_detect_spike_and_missing() -> None:
    normal = pmu_window_features(_frame(3), 3)
    event = pmu_window_features(_frame(3, spike=True), 3)
    missing = pmu_window_features(_frame(4, missing=True), 4)
    assert event["VA_MAG__max_abs_robust_z"] > normal["VA_MAG__max_abs_robust_z"]
    assert missing["DQ__data_present_fraction"] == 0.0
    assert missing["DQ__nan_fraction_max"] == 1.0


def test_aggregation_and_severity_are_order_invariant() -> None:
    rows_a = {3: pmu_window_features(_frame(3, spike=True), 3), 8: pmu_window_features(_frame(8), 8)}
    rows_b = {8: rows_a[8], 3: rows_a[3]}
    assert pmu_severity_vector(rows_a)[3] > pmu_severity_vector(rows_a)[8]
    assert aggregate_pmu_features(rows_a)["GLOB__top_pmu_bus"] == aggregate_pmu_features(rows_b)["GLOB__top_pmu_bus"]


def test_zbus_diffusion_scores_rank_near_candidate() -> None:
    distances = pd.DataFrame(
        [
            {"from_bus": 3, "to_bus": 3, "z_eff_abs": 0.001},
            {"from_bus": 8, "to_bus": 3, "z_eff_abs": 0.8},
            {"from_bus": 3, "to_bus": 8, "z_eff_abs": 0.8},
            {"from_bus": 8, "to_bus": 8, "z_eff_abs": 0.001},
        ]
    )
    scores = zbus_diffusion_scores({3: 10.0, 8: 1.0}, distances, [3, 8], taus=(0.2,))
    assert scores["GSP__BUS3__tau0.2__cosine"] > scores["GSP__BUS8__tau0.2__cosine"]


def test_graph_feature_block_exports_global_scores(tmp_path: Path) -> None:
    distances = pd.DataFrame(
        [
            {"from_bus": 3, "to_bus": 3, "z_eff_abs": 0.001},
            {"from_bus": 8, "to_bus": 8, "z_eff_abs": 0.001},
            {"from_bus": 3, "to_bus": 8, "z_eff_abs": 0.5},
            {"from_bus": 8, "to_bus": 3, "z_eff_abs": 0.5},
        ]
    )
    rows = {3: pmu_window_features(_frame(3, spike=True), 3), 8: pmu_window_features(_frame(8), 8)}
    out = graph_feature_block(rows, distances, candidate_buses=[3, 8], lines=[(3, 8)])
    assert "GSP__severity_graph_total_variation" in out
    assert "GSP__line_grid__top1_score" in out


def test_training_table_removes_specific_bus_ids() -> None:
    frame = pd.DataFrame(
        {
            "event_label": [0, 1],
            "abnormal_label": [0, 1],
            "location_label": ["none", "BUS3"],
            "location_type": ["NONE", "BUS"],
            "BUS3__BUS3_VA_MAG__full__max_abs": [0.1, 2.0],
            "BUS8__BUS8_VA_MAG__full__max_abs": [0.2, 1.0],
            "BUS3__BUS3_IA_MAG__max_abs_derivative": [0.0, 4.0],
            "BUS8__BUS8_IA_MAG__max_abs_derivative": [0.0, 2.0],
        }
    )
    agnostic = _build_bus_agnostic_table(frame)
    assert any(col.startswith("PMU_AGG__") for col in agnostic.columns)
    assert not any("BUS3" in col or "BUS8" in col for col in agnostic.columns)
    assert float(agnostic.loc[1, "PMU_AGG__VA_MAG__full__max_abs__max"]) == 2.0


def test_event4_ranker_reads_raw_dynamic_semantics_without_bus_feature_names() -> None:
    row = pd.Series(
        {
            "ROLL__BUS2__VA_MAG__w5__energy": 5.0,
            "HILB__BUS2__Freq__inst_freq__max_abs": 0.2,
            "BUS2__single_signal_score_max": 3.0,
        }
    )
    severity = _pmu_load_severity(row, (2,))
    features = _candidate_features(row, 7, {(2, 7): 0.1}, {7: 2}, observed=(2,), severity_map=severity)
    assert severity[2] > 0.0
    assert not any("BUS2" in key or "BUS7" in key for key in features)


def test_pmu_discovery_infers_raw0002_style_buses_without_legacy_assumption() -> None:
    columns = [
        "ROLL__BUS11__VA_MAG__w5__energy",
        "HILB__BUS17__Freq__inst_freq__max_abs",
        "RESID__BUS29__influence_cosine",
        "BUS31__BUS31_IA_MAG__full__max_abs",
    ]
    assert infer_pmu_buses_from_columns(columns) == (11, 17, 31)


def test_dynamic_extractor_uses_detected_frames_not_fixed_raw001_pmus() -> None:
    frames = {11: _frame(11, spike=True), 17: _frame(17)}
    assert infer_pmu_buses_from_frames(frames) == (11, 17)
    features = extract_dynamic_features_from_frames(frames, include_blocks=("rolling",))
    assert "ROLL__BUS11__VA_MAG__w5__energy" in features
    assert "ROLL__BUS17__VA_MAG__w5__energy" in features
    assert not any("__BUS29__" in key or "__BUS39__" in key for key in features)
