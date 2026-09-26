from pathlib import Path

import numpy as np

from research.pmu_hybrid_dae_bayes.scripts.cross_k1_second_order_identity_pilot_v1 import (
    fit_even_cross_coefficients,
    fit_vector_coefficients,
    richardson,
)


def test_mixed_richardson_recovers_bilinear_hessian():
    values = {}
    for level, hi, hj in (("coarse", 1e-4, 2e-4), ("fine", 5e-5, 1e-4)):
        for si in (-1, 1):
            for sj in (-1, 1):
                ai, aj = si * hi, sj * hj
                values[(level, si, sj)] = np.array([[3.0 * ai * aj, -2.0 * ai * aj]])
    estimate, uncertainty, _, _ = richardson(values)
    np.testing.assert_allclose(estimate, [[3.0, -2.0]], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(uncertainty, 0.0, atol=1e-10)


def test_vector_polynomial_recovers_a2():
    lam = np.array([.125, .25, .5, .75, 1.0])
    a2 = np.array([2.0, -1.0]); a3 = np.array([.4, .7]); a4 = np.array([-.2, .1])
    y = lam[:, None] ** 2 * a2 + lam[:, None] ** 3 * a3 + lam[:, None] ** 4 * a4
    coef, se, cond, fitted = fit_vector_coefficients(lam, y)
    np.testing.assert_allclose(coef[0], a2, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fitted, y, rtol=1e-12, atol=1e-12)
    assert np.all(se < 1e-10)
    assert np.isfinite(cond)


def test_even_cross_polynomial_recovers_a2():
    lam = np.array([.5, 1.0, 5.0, 20.0])
    a2 = np.array([2.0, -1.0]); a4 = np.array([.04, .07]); a6 = np.array([-2e-5, 1e-5])
    y = lam[:, None] ** 2 * a2 + lam[:, None] ** 4 * a4 + lam[:, None] ** 6 * a6
    coef, se, cond, fitted = fit_even_cross_coefficients(lam, y)
    np.testing.assert_allclose(coef[0], a2, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(fitted, y, rtol=1e-10, atol=1e-10)
    assert np.all(se < 1e-7)
    assert np.isfinite(cond)


def test_no_output_artifact_claims_v3_execution():
    source = Path(__file__).parents[1] / "research" / "pmu_hybrid_dae_bayes" / "scripts" / "cross_k1_second_order_identity_pilot_v1.py"
    text = source.read_text(encoding="utf-8")
    assert '"V3_READINESS": "NOT_READY"' in text
    assert '"future_v3_excluded": True' in text
