from __future__ import annotations

import numpy as np

from src.calibration.residual_sampling import (
    candidate_noise_models,
    fit_residual_distribution,
    sample_block_bootstrap,
    sample_residual_process,
    score_distribution_fit,
)


def _noise_stats() -> dict:
    return {
        "fitted_distribution_type": "gaussian",
        "kurtosis": 1.0,
        "skewness": 0.2,
        "residual_samples": list(np.random.default_rng(0).normal(size=300)),
        "std_dev_abs": 0.1,
        "std_dev_raw": 0.1,
        "ar1_rho": 0.2,
        "p_outlier": 0.01,
        "outlier_mag_abs": 0.3,
    }


def test_fit_residual_distribution_respects_requested_mode() -> None:
    assert fit_residual_distribution(_noise_stats(), "current_mag", requested_model="bootstrap") == "bootstrap"


def test_candidate_noise_models_nonempty() -> None:
    assert candidate_noise_models(_noise_stats(), "current_mag")


def test_sample_block_bootstrap_returns_length_n() -> None:
    out = sample_block_bootstrap(np.arange(100, dtype=float), 50, np.random.default_rng(1))
    assert len(out) == 50


def test_sample_residual_process_returns_length_n() -> None:
    out = sample_residual_process(_noise_stats(), 120, np.random.default_rng(2), "gaussian")
    assert len(out) == 120


def test_score_distribution_fit_finite_for_valid_inputs() -> None:
    a = np.random.default_rng(3).normal(size=200)
    b = np.random.default_rng(4).normal(size=200)
    assert np.isfinite(score_distribution_fit(a, b))
