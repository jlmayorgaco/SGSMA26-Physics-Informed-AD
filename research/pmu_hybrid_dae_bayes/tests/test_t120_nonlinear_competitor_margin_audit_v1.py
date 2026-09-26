"""Contract tests for the audit-only T120 nonlinear margin replay."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "powerdynamics_ieee39" / "output" / "t120_nonlinear_competitor_margin_audit_v1"
RES = OUT / "results"


def test_reoptimized_margin_cardinality_and_horizons():
    d = pd.read_csv(RES / "corrected_profiled_margins.csv")
    assert len(d) == 528 * 5
    assert set(d.horizon) == {30, 45, 60, 90, 120}
    assert d.groupby(["op_tag", "pair", "ai", "aj"]).size().eq(5).all()
    cols = ["J_model", "d_analytic", "second_best_distance", "model_error", "eta_model"]
    assert np.isfinite(d[cols].to_numpy(float)).all()
    assert (d.J_model >= 0).all() and (d.d_analytic >= 0).all()


def test_discrete_tds_reference_is_explicit_upper_bound():
    g = pd.read_csv(RES / "tds_grid_competitor_reference.csv")
    assert len(g) == 528 * 5
    assert set(g.horizon) == {30, 45, 60, 90, 120}
    assert g.reference_is_upper_bound.astype(bool).all()
    assert np.isfinite(g[["tds_grid_distance", "R_grid"]].to_numpy(float)).all()


def test_information_monotonicity_and_gamma4():
    info = pd.read_csv(RES / "corrected_information_monotonicity.csv")
    assert len(info) == 528 * 5
    assert np.isfinite(info.J.to_numpy(float)).all()
    violations = pd.read_csv(RES / "information_monotonicity_violations.csv")
    assert violations.empty
    gamma = pd.read_csv(RES / "gamma4_regression.csv")
    assert len(gamma) == 3
    assert (gamma.gamma4_infinity > 0).all()


def test_audit_is_read_only_and_not_v3():
    manifest = pd.read_csv(RES / "audit_manifest.csv")
    assert len(manifest) == 3
    assert not manifest.new_tds_generated.astype(bool).any()
    report = (OUT / "reports" / "t120_nonlinear_competitor_margin_audit_v1.md").read_text(encoding="utf-8")
    assert "V3_READINESS = NOT_READY" in report
    assert "no new TDS" in report


def test_reference_quality_is_not_overclaimed():
    cov = pd.read_csv(RES / "stored_severity_coverage.csv")
    assert len(cov) == 528
    assert set(cov.quality).issubset(
        {"DENSE_LOCAL_REFERENCE", "SPARSE_LOCAL_REFERENCE", "SUPPORT_ONLY_REFERENCE", "NO_USEFUL_REFERENCE"}
    )
    report = (OUT / "reports" / "t120_nonlinear_competitor_margin_audit_v1.md").read_text(encoding="utf-8")
    assert "INCONCLUSIVE" in report
