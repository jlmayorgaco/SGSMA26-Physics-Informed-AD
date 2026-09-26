from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "powerdynamics_ieee39" / "output" / "event_map_alignment_pilot_v1"
RES = OUT / "results"


def test_production_flow_manifest_is_complete_and_finite():
    m = pd.read_csv(OUT / "flow_manifest.csv")
    assert len(m) == 21
    assert m.case.nunique() == 21
    assert m.finite.all()
    assert m.retcode.astype(str).str.contains("Success|RESUMED").all()


def test_first_step_derivatives_and_richardson_uncertainty_are_finite():
    d = pd.read_csv(RES / "production_first_step_derivatives.csv")
    u = pd.read_csv(RES / "numerical_uncertainty.csv")
    assert {"self_bus7", "cross_bus7_12"} <= set(d.direction)
    assert len(d) == 192 * 5
    assert np.isfinite(d.value).all()
    assert np.isfinite(u.select_dtypes("number").to_numpy()).all()
    assert (u.richardson_error_h0025 >= 0).all()


def test_three_way_alignment_contract_and_homogeneous_test():
    d = pd.read_csv(RES / "aligned_vs_existing_vs_tds.csv")
    h = pd.read_csv(RES / "homogeneous_error_dynamics.csv")
    assert len(d) == 2 * 30 * 32
    assert d.direction.nunique() == 2
    assert np.isfinite(d[["aligned", "existing", "tds_richardson"]].to_numpy()).all()
    assert len(h) == 2 * 30
    assert np.isfinite(h.select_dtypes("number").to_numpy()).all()
    assert h.relative_error.median() < 1e-4
    assert h.cosine.min() > 0.999


def test_alignment_reduces_first_sample_and_window_error():
    s = pd.read_csv(RES / "alignment_summary.csv").set_index("direction")
    assert (s.aligned_raw_median_relative < s.existing_raw_median_relative).all()
    assert (s.median_error_reduction_fraction > 0).all()
    assert s.loc["self_bus7", "first_frame_aligned_abs"] < 1e-7
    assert s.loc["cross_bus7_12", "first_frame_aligned_abs"] < 1e-7


def test_report_closes_only_the_audit_and_excludes_future_campaigns():
    text = (OUT / "reports" / "event_map_alignment_pilot_v1.md").read_text(encoding="utf-8")
    for marker in (
        "EVENT_MAP_MISMATCH = CONFIRMED_PRIMARY_CAUSE",
        "SECOND_ORDER_CONTINUUM_PROPAGATION = PASS_AFTER_EVENT_MAP_ALIGNMENT",
        "T30_BACKWARD_COMPATIBILITY = PASS",
    ):
        assert marker in text
    assert "T120" in text and "Optional 1/60 and 1/120" in text
