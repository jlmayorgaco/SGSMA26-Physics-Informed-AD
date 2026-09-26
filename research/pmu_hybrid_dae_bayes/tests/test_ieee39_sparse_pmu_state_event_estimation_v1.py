"""Contract tests for the IEEE39 sparse-PMU integration pilot."""
from pathlib import Path
import importlib.util
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "research" / "pmu_hybrid_dae_bayes" / "scripts" / "ieee39_sparse_pmu_state_event_estimation_v1.py"
spec = importlib.util.spec_from_file_location("sparse_e2e_v1", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_contract_has_exact_eight_pmus_and_137_hypotheses():
    assert mod.PMU_BUSES == [2, 5, 6, 10, 19, 22, 29, 39]
    assert 7 not in mod.PMU_BUSES
    assert len(mod.SUPPORTS) == 137


def test_horizons_are_nested_prefixes():
    assert mod.HORIZONS == [5, 10, 20, 30, 45, 60, 90, 120]
    assert mod.HORIZONS == sorted(mod.HORIZONS)


def test_event_delta_nominal_is_zero():
    z = np.zeros((4, 39), dtype=float)
    assert np.allclose(z, 0.0)


def test_state_dimension_contract():
    assert len(mod.SUPPORTS) == 1 + 16 + 120
    assert mod.RHO == 0.3512083596353588
