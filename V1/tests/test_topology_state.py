"""Tests for Ybus/Kirchhoff topology-state reconstruction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

DATA_DIR = Path("data/raw")
SKIP_NO_DATA = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="data/raw not present - skipping real-data tests",
)


@pytest.fixture(scope="module")
def grid():
    from src.grid.load_case import load_case
    raw = Path("data/metadata/IEEE 39 Bus Power System.raw")
    if not raw.exists():
        pytest.skip("No .raw power-flow file found")
    return load_case(raw)


@pytest.fixture(scope="module")
def full_df():
    from src.io.load_csv import load_all
    return load_all(DATA_DIR)


def test_harmonic_extension_shape(grid):
    from src.estimator.topology_state import TopologyStateEstimator

    est = TopologyStateEstimator(grid)
    obs_idx = np.array(grid.pmu_bus_indices, dtype=int)
    obs = np.ones(len(obs_idx), dtype=float)
    full = est._extend(obs_idx, obs, est.base_vm)
    assert full.shape == (39,)
    assert np.all(np.isfinite(full))


@SKIP_NO_DATA
def test_estimated_state_residuals_include_all_buses(grid, full_df):
    from src.estimator.topology_state import TopologyStateEstimator

    est = TopologyStateEstimator(grid)
    ts = full_df["TIMESTAMP"].to_numpy(float)
    frame = int(np.argmin(np.abs(ts - 3877.733)))
    by_bus = est.residual_by_bus(full_df, frame)
    assert len(by_bus) == 39
    assert 7 in by_bus
    assert 24 in by_bus
    assert all(np.isfinite(v) and v >= 0.0 for v in by_bus.values())


@SKIP_NO_DATA
def test_bus7_load_change_has_high_topology_energy(grid, full_df):
    from src.estimator.topology_state import TopologyStateEstimator

    est = TopologyStateEstimator(grid)
    ts = full_df["TIMESTAMP"].to_numpy(float)
    frame = int(np.argmin(np.abs(ts - 3877.733)))
    by_bus = est.residual_by_bus(full_df, frame)
    max_energy = max(by_bus.values())
    assert by_bus[7] / max_energy > 0.85
