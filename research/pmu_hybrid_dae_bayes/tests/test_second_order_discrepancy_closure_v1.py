from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "powerdynamics_ieee39" / "output" / "second_order_discrepancy_closure_v1"
RES = EXP / "results"


def test_descriptor_residual_and_mass_inventory():
    d = pd.read_csv(RES / "descriptor_residual_audit.csv").iloc[0]
    assert (int(d.n), int(d.differential), int(d.algebraic), int(d.mass_rank)) == (192, 114, 78, 114)
    assert d.residual_match == "PASS"
    m = pd.read_csv(RES / "mass_matrix_derivative_audit.csv")
    assert len(m) == 8
    assert np.max(np.abs(m.abs_derivative)) == 0.0
    assert (m.status == "PASS_CONSTANT_NETWORK_MASS").all()


def test_residual_hvp_parity_and_finite_values():
    h = pd.read_csv(RES / "residual_hvp_comparison.csv")
    assert len(h) == 24
    numeric = h[["h", "ad_norm", "fd_norm", "abs_error", "relative_error", "cosine"]]
    assert np.isfinite(numeric.to_numpy()).all()
    stable = h[h.h <= 3e-4]
    assert stable.relative_error.max() < 1e-3
    bridge = pd.read_csv(RES / "residual_hvp_analytic_contraction_bridge.csv").iloc[0]
    assert bool(bridge.finite_all)
    assert bridge.status == "PASS"


def test_exact_onset_exposes_production_callback_semantics():
    o = pd.read_csv(RES / "exact_onset_second_order.csv")
    assert set(o["sample"]) == {"event_exact", "first_post_save"}
    exact = o[o["sample"] == "event_exact"]
    assert (exact.status == "PASS_CONTINUOUS_CALLBACK").all()
    assert np.allclose(exact.differential_first_norm.fillna(0), 0.0)
    assert np.allclose(exact.algebraic_first_norm.fillna(0), 0.0)
    post = o[o["sample"] == "first_post_save"]
    assert (post.tau_s > 0).all()


def test_q_error_timing_tolerance_and_coordinate_audits():
    q = pd.read_csv(RES / "q_error_vs_time.csv")
    assert len(q) == 17 * 30
    assert np.isfinite(q.relative_error).all()
    assert q[q.frame == 1].relative_error.median() > 0.0
    t = pd.read_csv(RES / "solver_tolerance_convergence.csv")
    assert len(t) == 6 and (t.retcode == "Success").all()
    assert t.relative_to_richardson.median() < 1e-2
    c = pd.read_csv(RES / "coordinate_pipeline_audit.csv")
    assert (c.status == "PASS").all()
    assert c.loc[c.check == "PMU_voltage_basis_fd", "value"].iloc[0] < 1e-8


def test_short_time_and_physical_impact_artifacts():
    s = pd.read_csv(RES / "short_time_flow_convergence.csv")
    assert len(s) == 27
    assert (s.retcode == "Success").all()
    same_tau = s[np.isclose(s.tau_s, 1 / 30)]
    assert len(same_tau) == 3
    assert same_tau.relative_to_analytic_interp.max() - same_tau.relative_to_analytic_interp.min() < 1e-6
    impact = pd.read_csv(RES / "q_error_physical_impact.csv")
    assert len(impact) == 192
    assert np.isfinite(impact.eta_Q).all()
    margin = pd.read_csv(RES / "robust_margin_impact.csv")
    assert len(margin) == len(impact)


def test_closure_report_excludes_future_campaigns():
    summary = pd.read_json(RES / "summary.json", typ="series")
    assert summary["v3_exclusion"]
    text = (EXP / "reports" / "second_order_discrepancy_closure_v1.md").read_text(encoding="utf-8")
    assert "SECOND_ORDER_CONTINUUM_VALIDATION = BLOCKED" in text
    assert "T120" in text and "no V3" in text
