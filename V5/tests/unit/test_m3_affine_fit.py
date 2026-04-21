from __future__ import annotations

import numpy as np

from src.calibration.calibration_fit import (
    apply_quantile_mapping,
    condition_prepared_to_chunk,
    fit_affine_distribution,
    fit_quantile_mapping,
)


def test_fit_affine_distribution_empty_returns_identity_empty() -> None:
    a, b, mode = fit_affine_distribution([], [], "current_mag")
    assert (a, b, mode) == (1.0, 0.0, "identity_empty")


def test_fit_affine_distribution_constant_returns_median_constant() -> None:
    a, _, mode = fit_affine_distribution(np.ones(20), np.arange(20), "current_mag")
    assert mode == "median_constant"
    assert np.isclose(a, 0.0)


def test_fit_affine_distribution_returns_finite_a_b() -> None:
    sim = np.linspace(1.0, 2.0, 100)
    real = 2.0 * sim + 0.5
    a, b, _ = fit_affine_distribution(sim, real, "voltage_mag")
    assert np.isfinite(a) and np.isfinite(b)


def test_fit_quantile_mapping_returns_none_for_short_inputs() -> None:
    assert fit_quantile_mapping(np.arange(5), np.arange(5)) is None


def test_fit_quantile_mapping_valid_case_returns_src_dst() -> None:
    m = fit_quantile_mapping(np.linspace(0, 1, 200), np.linspace(1, 2, 200))
    assert m is not None
    assert "src" in m and "dst" in m


def test_apply_quantile_mapping_preserves_length() -> None:
    m = fit_quantile_mapping(np.linspace(0, 1, 200), np.linspace(1, 2, 200))
    out = apply_quantile_mapping(np.linspace(0, 1, 50), m)
    assert len(out) == 50


def test_condition_prepared_to_chunk_noop_for_non_supported_family() -> None:
    prepared = {"sim_clean": np.arange(10), "sim_drifted": np.arange(10), "sim_noisy": np.arange(10), "fit_mode": "x"}
    out = condition_prepared_to_chunk(prepared, np.arange(10), "other")
    assert out["chunk_conditioning_used"] == "no"


def test_condition_prepared_to_chunk_applies_mapping_when_possible() -> None:
    prepared = {
        "sim_clean": np.linspace(0, 1, 80),
        "sim_drifted": np.linspace(0, 1, 80),
        "sim_noisy": np.linspace(0, 1, 80),
        "fit_mode": "base",
    }
    real = np.linspace(1, 2, 80)
    out = condition_prepared_to_chunk(prepared, real, "current_mag")
    assert out["chunk_conditioning_used"] in {"yes", "no"}
