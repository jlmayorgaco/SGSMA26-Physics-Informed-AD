from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pmu_hybrid.e04a4_colored import acf_values, augmented_state_dimension, fit_ar1, pca_fit, stationary_covariance


def test_pca_fit_and_cal_test_leakage() -> None:
    x = np.array([[1., 0.], [2., 0.], [3., 0.]])
    mean, basis, singular, explained = pca_fit(x, 1)
    assert mean.shape == (2,) and basis.shape == (2, 1) and explained[0] == pytest.approx(1.0)
    root = Path(__file__).parents[1] / "powerdynamics_ieee39" / "output" / "results"
    cal = pd.read_csv(root / "e04a3_cal_split_manifest.csv")
    frozen = pd.read_csv(root / "e04_pd_dataset.csv")
    assert set(cal.seed).isdisjoint(set(frozen[frozen.split == "TEST"].seed))


def test_latent_state_dimension_and_stationary_covariance() -> None:
    assert augmented_state_dimension(114, 4) == 118
    pc = stationary_covariance(np.array([.5]), np.array([.75]))
    assert pc[0, 0] == pytest.approx(1.0)


def test_ar1_synthetic_recovery_and_stability_guard() -> None:
    x = np.zeros(1000)
    rng = np.random.default_rng(3)
    for k in range(1, len(x)):
        x[k] = .8 * x[k - 1] + rng.normal(scale=.1)
    f, q = fit_ar1(x)
    assert f == pytest.approx(.8, abs=.05) and q > 0
    with pytest.raises(ValueError):
        stationary_covariance(np.array([1.0]), np.array([1.0]))


def test_augmented_kalman_toy_and_whiteness_metric() -> None:
    # A stable latent discrepancy leaves the physical dimension untouched.
    assert augmented_state_dimension(2, 1) == 3
    white = np.random.default_rng(4).normal(size=500)
    colored = np.cumsum(white) * .01
    assert abs(acf_values(colored, (1,))[1]) > abs(acf_values(white, (1,))[1])
