from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
OUT = ROOT / "output" / "t120_op_event_map_replay_v1"
RES = OUT / "results"


def test_existing_op_banks_and_first_step_rows():
    d = pd.read_csv(RES / "first_step_derivatives_by_op.csv")
    assert set(d.op_tag) == {"op_m035", "op_m085", "op_m125"}
    assert len(d[d.direction == "self_7"]) == 6
    assert len(d[d.direction == "cross_7_12"]) == 3
    assert d[d.derivative_order == 1].relative_error.max() < 1e-5


def test_second_order_map_mismatch_is_explicit_and_tail_is_uncorrected():
    d = pd.read_csv(RES / "first_step_derivatives_by_op.csv")
    assert d[d.derivative_order == 2].relative_error.max() > 1e-2
    report = (OUT / "reports" / "t120_op_event_map_replay_v1.md").read_text(encoding="utf-8")
    assert "OP_EVENT_MAP_CORRECTION_REQUIRED = YES" in report
    assert "T120_PHYSICAL_CONTRACT = NOT_FREEZE_READY" in report
    assert "V3_READINESS = NOT_READY" in report


def test_no_new_tds_manifest_and_historical_tail_reference():
    audit = pd.read_csv(RES / "op_event_map_audit.csv")
    assert (~audit.state_derivative_available).all()
    eta = pd.read_csv(RES / "eta_tail_replay.csv")
    assert (eta.query("model == 'eta_conditioned'").gt_1 == 0).all()
