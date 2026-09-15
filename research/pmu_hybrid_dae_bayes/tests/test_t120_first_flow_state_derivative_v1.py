from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
OUT = ROOT / "output" / "t120_first_flow_state_derivative_v1"
RES = OUT / "results"


def test_complete_native_state_maps_are_finite_and_dimensioned():
    rows = pd.read_csv(RES / "full_state_first_flow.csv")
    assert set(rows.op_tag) == {"op_m035", "op_m085", "op_m125"}
    # 3 operating points x (5 self directions x 4 signs/scales +
    # 4 cross directions x 8 sign/scale combinations).
    assert len(rows) == 156
    assert set(rows.n_state) == {192}
    assert rows.finite.astype(bool).all()


def test_first_flow_derivative_orders_and_parity():
    first = pd.read_csv(RES / "first_order_state_derivatives.csv")
    second = pd.read_csv(RES / "second_order_state_derivatives.csv")
    assert len(first) == 15
    assert len(second) == 27
    assert first.relative_error.max() < 1e-5
    assert second.relative_error.max() > 1e-2
    assert np.isfinite(second[["relative_error", "cosine"]].to_numpy()).all()


def test_pilot_is_explicitly_partial_and_does_not_claim_corrected_tail():
    report = (OUT / "reports" / "t120_first_flow_state_derivative_v1.md").read_text(
        encoding="utf-8"
    )
    assert "CORRECTED_OP_DICTIONARY = NOT_BUILT_FULL" in report
    assert "T120_CORRECTED_PHYSICAL_REPLAY = NOT_RUN" in report
    assert "T120_STATE_EVENT_MAP_VALIDATION = PARTIAL" in report
    assert "T120_PHYSICAL_CONTRACT = NOT_FREEZE_READY" in report
    assert "V3_READINESS = NOT_READY" in report
    tail = pd.read_csv(RES / "corrected_eta_tail.csv")
    assert tail.empty
    hom = pd.read_csv(RES / "homogeneous_propagation_closure.csv")
    assert hom.iloc[0].status == "NOT_RUN_PENDING_FULL_STATE_BASIS"


def test_state_partition_is_114_plus_78():
    manifest = pd.read_csv(RES / "corrected_dictionary_manifest.csv")
    assert set(manifest.state_dimension) == {192}
    assert set(manifest.differential) == {114}
    assert set(manifest.algebraic) == {78}
