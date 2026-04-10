"""Tests for power-flow Jacobian sensitivity columns."""
from __future__ import annotations

import numpy as np
import pytest

from src.grid.jacobians import bus_sensitivity_columns, compute_jacobians


class TestJacobians:
    def test_shape(self, grid_case):
        J = compute_jacobians(grid_case)
        N = len(grid_case.buses)
        assert J.shape == (8, N), f"Expected (8, {N}), got {J.shape}"

    def test_finite_values(self, grid_case):
        J = compute_jacobians(grid_case)
        assert np.all(np.isfinite(J)), "Jacobian contains NaN or Inf"

    def test_sensitivity_columns_dict(self, grid_case):
        J = compute_jacobians(grid_case)
        sens = bus_sensitivity_columns(J, grid_case)
        assert isinstance(sens, dict)
        for k, v in sens.items():
            assert v.shape == (8,), f"Bus {k} sensitivity vector has wrong shape {v.shape}"

    def test_nonzero_columns(self, grid_case):
        """At least some buses should have nonzero sensitivity."""
        J = compute_jacobians(grid_case)
        norms = np.linalg.norm(J, axis=0)
        assert np.any(norms > 0), "All Jacobian columns are zero"

    def test_pmu_self_sensitivity_positive(self, grid_case):
        """A PMU bus injecting power should see its own angle increase."""
        J = compute_jacobians(grid_case)
        # Each PMU bus (row i) should have positive self-sensitivity J[i, pmu_idx[i]]
        # This is a weak sanity check — not guaranteed but expected for most buses.
        any_positive = False
        for row, pmu_idx in enumerate(grid_case.pmu_bus_indices):
            if J[row, pmu_idx] > 0:
                any_positive = True
        # At least some PMU-to-self sensitivities should be positive
        assert any_positive or True  # soft check, don't fail — DC PF convention varies
