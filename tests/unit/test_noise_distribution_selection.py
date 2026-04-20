from __future__ import annotations

import numpy as np

from src.calibration.residual_models import choose_distribution_type


def test_short_residual_returns_gaussian() -> None:
    assert choose_distribution_type(np.array([0.1, -0.1]), "current_mag") == "gaussian"


def test_heavy_tail_current_mag_prefers_bootstrap() -> None:
    x = np.concatenate([np.random.RandomState(0).normal(0, 1, 1000), np.array([20.0, -25.0])])
    assert choose_distribution_type(x, "current_mag") == "bootstrap"


def test_kurtotic_distribution_prefers_student_t() -> None:
    x = np.random.RandomState(1).standard_t(df=2.5, size=2000)
    out = choose_distribution_type(x, "voltage_mag")
    assert out in {"student_t", "bootstrap"}


def test_skewed_distribution_prefers_gmm_or_valid_nonempty_type() -> None:
    x = np.random.RandomState(2).exponential(scale=1.0, size=1200) - 1.0
    out = choose_distribution_type(x, "other")
    assert out in {"gmm", "student_t", "gaussian", "bootstrap"}


def test_normal_distribution_returns_gaussian() -> None:
    x = np.random.RandomState(3).normal(0, 1, 2000)
    assert choose_distribution_type(x, "other") == "gaussian"
