from __future__ import annotations

import numpy as np

from src.calibration.calibration_fit import apply_quantile_mapping, fit_quantile_mapping


def test_fit_quantile_mapping_valid_case_returns_mapping() -> None:
    m = fit_quantile_mapping(np.linspace(0, 1, 100), np.linspace(1, 2, 100))
    assert m is not None


def test_apply_quantile_mapping_monotonic_output() -> None:
    m = fit_quantile_mapping(np.linspace(0, 1, 100), np.linspace(1, 2, 100))
    out = apply_quantile_mapping(np.linspace(0, 1, 50), m)
    assert np.all(np.diff(out) >= -1e-12)
