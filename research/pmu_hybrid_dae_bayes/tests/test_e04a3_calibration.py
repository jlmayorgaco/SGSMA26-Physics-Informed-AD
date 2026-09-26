from pathlib import Path
import numpy as np
import pandas as pd

from pmu_hybrid.e04a3_calibration import (
    coverage, fit_scalar_temperature, gaussian_nll, innovation_acf,
    low_rank_spectrum, nis_raw_normalized,
)


def test_raw_and_normalized_nis_definition() -> None:
    raw, norm = nis_raw_normalized(np.ones(2), np.eye(2))
    assert raw == 2.0 and norm == 1.0


def test_scalar_temperature_does_not_change_mean_and_matches_reference_nll() -> None:
    e = np.array([[1.0, 0.0], [0.0, 1.0]])
    c = np.broadcast_to(np.eye(2), (2, 2, 2)).copy()
    tau, empirical, nll = fit_scalar_temperature(e, c)
    assert tau == empirical == 0.5
    assert np.isclose(nll, gaussian_nll(e, c, tau=tau))
    posterior_mean = np.array([3.0, -2.0]); posterior_cov = np.eye(2)
    calibrated_cov = tau * posterior_cov
    assert np.array_equal(posterior_mean, posterior_mean.copy())
    assert not np.array_equal(posterior_cov, calibrated_cov)


def test_cal_test_seed_disjointness() -> None:
    root = Path(__file__).parents[1] / "powerdynamics_ieee39" / "output" / "results"
    cal = pd.read_csv(root / "e04a3_cal_split_manifest.csv")
    frozen = pd.read_csv(root / "e04_pd_dataset.csv")
    assert set(cal.seed).isdisjoint(set(frozen[frozen.split.isin(["DEV", "TEST"])].seed))


def test_coverage_and_innovation_acf_toy() -> None:
    e = np.zeros((4, 2)); c = np.broadcast_to(np.eye(2), (4, 2, 2)).copy()
    assert coverage(e, c, .95) == (1.0, 1.0)
    x = np.arange(20, dtype=float); acf = innovation_acf(x, (1, 2))
    assert acf[1] > 0.8 and acf[2] > 0.6


def test_low_rank_pca_reconstruction_spectrum() -> None:
    t = np.arange(100.0); x = np.column_stack([t, 2*t, -t, np.zeros_like(t)])
    eig, explained = low_rank_spectrum(x, (1, 2, 4))
    assert eig[0] > 0 and explained[0] > 0.99 and explained[-1] == 1.0
