"""Contract tests for the frozen LOAD-BAYES-FD-V2 campaign.

These tests are intentionally lightweight: they validate provenance, split
separation, numerical identities, and the generated artifacts without
rerunning the expensive native PowerDynamics trajectories.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT / "powerdynamics_ieee39"
OUT = PD / "output" / "load_bayes_fd_v2"
RES = OUT / "results"
BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28]
WEAK = {-0.004, -0.002, -0.001, -0.0005, -0.0002, -0.0001,
        0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.004}
FINITE = {-0.09, -0.065, -0.045, -0.03, -0.0125,
          0.0125, 0.03, 0.045, 0.065, 0.09}


def test_v2_artifacts_and_statuses_exist():
    required = [
        "load_bayes_dictionary_manifest.csv", "load_bayes_cal_manifest.csv",
        "load_bayes_dev_manifest.csv", "load_bayes_test_manifest.csv",
        "load_whitening_model.csv", "load_source_posterior.parquet",
        "load_source_summary.csv", "load_amplitude_summary.csv",
        "load_calibration.csv", "load_evi.csv",
        "load_pair_geometry_whitened.csv", "load_pair_confusion.csv",
        "load_bus7_bus12.csv", "load_bus7_summary.csv", "load_bayes_fd_v2_summary.csv",
    ]
    assert all((RES / name).is_file() for name in required)
    s = pd.read_csv(RES / "load_bayes_fd_v2_summary.csv").iloc[0]
    assert s.selected_whitening == "W2_SEPARABLE_AR1"
    assert s.selected_likelihood == "L2"
    assert s.ANALYTIC_DAE_TANGENT == "PENDING"
    assert int(s.cal_normal) == 1000 and int(s.dev_normal) == 500


def test_dictionary_is_frozen_and_hashed():
    dpath = PD / "output" / "load_tangent_v2" / "results" / "load_fd_central_operator.npz"
    manifest = pd.read_csv(RES / "load_bayes_dictionary_manifest.csv").iloc[0]
    assert manifest.v1_test_used_for_fit == False
    assert manifest.epsilon == 0.005
    assert int(manifest.n_sources) == 16 and int(manifest.n_frames) == 30
    assert int(manifest.n_channels) == 32
    assert manifest.sha256 == hashlib.sha256(dpath.read_bytes()).hexdigest()


def test_cal_dev_test_seed_separation_and_counts():
    cal = pd.read_csv(RES / "load_bayes_cal_manifest.csv")
    dev = pd.read_csv(RES / "load_bayes_dev_manifest.csv")
    test = pd.read_csv(RES / "load_bayes_test_manifest.csv")
    assert len(cal) == 1000 and len(dev) == 500
    assert set(cal.noise_seed).isdisjoint(set(dev.noise_seed))
    assert set(cal.noise_seed).isdisjoint(set(test.noise_seed))
    assert set(dev.noise_seed).isdisjoint(set(test.noise_seed))
    assert set(test.query("regime == 'WEAK'").true_amplitude.unique()) == WEAK
    assert set(test.query("regime == 'FINITE'").true_amplitude.unique()) == FINITE
    assert len(test.query("regime == 'WEAK'")) == 16 * 12 * 50
    assert len(test.query("regime == 'WEAK_H0'")) == 16 * 50
    assert len(test.query("regime == 'FINITE'")) == 16 * 10 * 50
    # Historical V1 seed ranges were 200000/300000; V2 uses independent ranges.
    assert set(cal.noise_seed).isdisjoint(set(range(200000, 400000)))
    assert set(dev.noise_seed).isdisjoint(set(range(200000, 400000)))


def test_native_physical_atlas_is_complete():
    m = pd.read_csv(OUT / "physical" / "simulation_manifest_native.csv")
    assert len(m) == 16 * 31
    assert set(m.status) == {"EXECUTED_SUCCESS"}
    assert set(m.candidate_bus) == set(BUSES)
    assert 0.0 in set(m.amplitude)


def test_truncation_order_and_q_stability():
    p = pd.read_csv(RES / "load_truncation_order.csv").iloc[0]
    assert 1.5 < p.slope_p < 2.5
    assert p.ci95_low < 2.0 < p.ci95_high
    q = pd.read_csv(RES / "load_q_stability.csv")
    assert q.relative_to_all_dev_Q.median() < 1e-2


def test_posterior_normalization_and_frozen_geometry():
    p = pd.read_parquet(RES / "load_source_posterior.parquet")
    prob = p[["p_h0"] + [f"p_H{b}" for b in BUSES]].sum(axis=1)
    assert np.allclose(prob.to_numpy(), 1.0, atol=1e-10)
    g = pd.read_csv(RES / "load_pair_geometry_whitened.csv")
    assert len(g) == 16 * 15
    assert np.isfinite(g.J_gj).all()


def test_evi_threshold_formula_and_projected_fisher_identity():
    e = pd.read_csv(RES / "load_evi.csv")
    expected = (2.326347874 + 1.281551566) / np.sqrt(e.EVI.to_numpy())
    assert np.allclose(e.a_min_alpha001_power090, expected, rtol=1e-10)
    # Toy identity J_gj = EVI_g(1-mu^2), including a near-collinear pair.
    rng = np.random.default_rng(11)
    d1, d2 = rng.normal(size=(2, 20))
    e1, e2 = d1 @ d1, d2 @ d2
    mu = (d1 @ d2) / np.sqrt(e1 * e2)
    j = e1 - (d1 @ d2) ** 2 / e2
    assert np.isclose(j, e1 * (1 - mu * mu), rtol=1e-12, atol=1e-12)


def test_toy_second_order_and_a4_scaling():
    d, q, a = 2.0, -3.0, np.array([0.01, 0.02, 0.04])
    r = a * d + a**2 * q
    assert np.allclose(r - a * d, a**2 * q)
    # If deterministic truncation is O(a^2), its squared variance scales O(a^4).
    assert np.allclose((a**2) ** 2, a**4)


def test_l0_quadrature_matches_closed_form_scalar():
    # Normal prior a~N(0,sa²), r|a~N(d*a,sigma²); marginal is N(0,sigma²+sa²d²).
    d, sigma, sa, r = 1.7, 0.8, 0.05, 0.011
    grid = np.linspace(-0.3, 0.3, 20001)
    prior = np.exp(-0.5 * (grid / sa) ** 2) / (sa * np.sqrt(2 * np.pi))
    like = np.exp(-0.5 * ((r - d * grid) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    quad = np.trapz(prior * like, grid)
    closed = np.exp(-0.5 * r * r / (sigma**2 + sa**2 * d**2)) / np.sqrt(2 * np.pi * (sigma**2 + sa**2 * d**2))
    assert np.isclose(quad, closed, rtol=2e-5)
