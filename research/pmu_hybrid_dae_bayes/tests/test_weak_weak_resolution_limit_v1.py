from pathlib import Path
import ast
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
OUT = ROOT / "output" / "weak_weak_resolution_limit_v1"
RES = OUT / "results"


def test_analytic_tangent_numerical_consistency():
    d = pd.read_csv(RES / "analytic_tangent_validation.csv")
    assert len(d) == 16
    assert (d.status == "PASS").all()
    assert d.relative_l2_error.max() < 1e-3
    assert d.cosine.min() > 0.999


def test_whitened_projection_identity_and_direct_residual():
    rng = np.random.default_rng(12)
    x, y = rng.normal(size=(2, 960)); x /= np.linalg.norm(x); y /= np.linalg.norm(y)
    P = np.outer(x, x)
    assert np.allclose(P @ P, P, atol=1e-12)
    r = y - x * (x @ y)
    c = (r @ y) / (y @ y)
    assert np.isfinite(c)


def test_gamma_subset_enumeration_and_horizon_rows():
    d = pd.read_csv(RES / "gamma_vs_horizon.csv")
    assert set(d.horizon_frames) == {5, 10, 20, 30}
    assert set(d.k) == {1, 2, 3, 4}
    g = d[d.k == 4].sort_values("horizon_frames").gamma_k.to_numpy()
    assert np.all(np.diff(g) >= -1e-9)


def test_prefix_posterior_reproducibility():
    d = pd.read_csv(RES / "resolution_delay.csv")
    assert d.case_id.nunique() > 0
    q = d[d.horizon_frames == 30].sort_values(["case_id"])
    assert np.isfinite(q.p_M0).all() and np.isfinite(q.p_M1).all() and np.isfinite(q.p_M2).all()
    assert np.allclose(q[["p_M0", "p_M1", "p_M2"]].sum(axis=1), 1.0, atol=1e-10)


def test_frozen_global137_posterior_regression_contract():
    p = ROOT / "output" / "global_137_confirmatory_v1" / "results" / "support_per_case.csv"
    d = pd.read_csv(p, nrows=137 * 2)
    assert len(d) == 274
    assert d.groupby(["case_id", "noise_seed"]).size().iloc[0] == 137
    assert (d.groupby(["case_id", "noise_seed"]).size() == 137).all()


def test_structural_claim_is_not_proven():
    s = pd.read_csv(RES / "weak_weak_resolution_limit_summary.csv").iloc[0]
    assert s.frozen_estimator_regression == "PASS"
