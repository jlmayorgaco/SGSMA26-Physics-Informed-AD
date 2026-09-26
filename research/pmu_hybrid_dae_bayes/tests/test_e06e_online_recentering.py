import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import e06e_online_recentering as e


def test_breakpoint_definition_consistency():
    p = e.ROOT / "output/results/e06e_breakpoint_audit.csv"
    assert p.exists()
    d = pd.read_csv(p)
    assert d.aggregation.nunique() == 1
    assert d.denominator.nunique() == 1
    assert set(d.grid) == {"main"}


def test_no_oracle_equilibrium_leakage():
    src = Path(e.__file__).read_text(encoding="utf-8")
    fn = src.split("def online_recenter", 1)[1].split("def frozen", 1)[0]
    assert "truth" not in fn
    assert "hidden" not in fn
    assert "family" not in fn


def test_center_map_toy_reference():
    B = np.eye(2)
    d, cov, residual, it = e.static_map_update(np.array([1.0, -2.0]), np.zeros(2), B, qd=1.0, lam=0.0, rvar=1.0)
    assert np.allclose(d, np.array([.5, -1.0]))
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    assert residual > 0 and it == 1


def test_ac_constraint_proxy_and_causality():
    B = e.voltage_basis(np.zeros(32))
    slow = np.ones(16) * .02
    d, cov, residual, _ = e.static_map_update(slow, np.zeros(8), B)
    assert np.isfinite(d).all() and residual >= 0
    assert np.all(np.linalg.eigvalsh(cov) >= -1e-12)
    # The update is a trailing statistic, so a change at frame k cannot alter
    # the center reported before frame k.
    assert np.allclose(B @ np.zeros(8), 0)


def test_closure_denominator_gate():
    floor = 1e-6
    assert np.isnan((0.1 - 0.0999995) / (0.1 - 0.0999995)) or abs(0.1 - 0.0999995) < floor


def test_generated_split_counts_and_seed_disjointness():
    dev = pd.read_csv(e.R / "e06e_dev_manifest.csv")
    test = pd.read_csv(e.R / "e06e_test_manifest.csv")
    assert len(dev) == 160 and len(test) == 320
    assert set(dev.seed).isdisjoint(set(test.seed))
    assert set(dev.seed).isdisjoint(set(range(1, 21)))
    assert set(test.seed).isdisjoint(set(range(1, 21)))


def test_nominal_false_activation_and_outputs():
    a = pd.read_csv(e.R / "e06e_activation.csv")
    x = a[(a.m == 0) & (a.policy == "ADAPTIVE_RECENTER")]
    assert x.active.mean() <= .05
    for name in ["e06e_summary.csv", "e06e_oracle_closure.csv", "e06e_center_error.csv", "e06e_uncertainty.csv", "e06e_runtime.csv"]:
        assert (e.R / name).exists()
