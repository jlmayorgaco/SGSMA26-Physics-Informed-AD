"""Regression tests for the read-only exact weak-regime audit."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39" / "output" / "exact_weak_regime_resolution_v2"
RES = ROOT / "results"


def test_gh31_t30_reproduces_frozen_global_cardinality():
    d = pd.read_csv(RES / "gh31_t30_regression.csv")
    assert d.status.iloc[0] == "PASS"
    assert float(d.max_T30_cardinality_abs_error.iloc[0]) < 1e-7


def test_prefix_contract_complete():
    d = pd.read_csv(RES / "gh31_prefix_summary.csv")
    assert set(d.horizon_frames) == {5, 10, 15, 20, 25, 30}
    assert d.case_id.nunique() == 2400
    assert len(pd.read_csv(RES / "gh31_prefix_posteriors.csv")) == 2400 * 6 * 137


def test_profiled_distance_categories_and_convergence():
    d = pd.read_csv(RES / "profiled_manifold_distances.csv")
    assert set(d.competitor_class) == {"M1", "SHARED_DOUBLE", "DISJOINT_DOUBLE", "GLOBAL"}
    assert d.groupby("case_id").competitor_class.nunique().eq(4).all()
    # A small set of ill-conditioned competing manifolds is explicitly logged
    # as non-converged by the low-dimensional optimizer; distances remain
    # finite and are not silently discarded.
    assert d.delta2.apply(float).map(lambda x: x == x and abs(x) < float("inf")).all()
    assert d.converged.mean() >= 0.95


def test_quotient_geometry_removes_shared_direction():
    d = pd.read_csv(RES / "quotient_geometry.csv")
    assert len(d) == 3360
    assert d.shared_intersection_removed.all()
    assert d.friedrichs_angle_rad.notna().all()


def test_order_atlas_is_explicitly_first_order():
    d = pd.read_csv(RES / "resolvability_order_fits.csv")
    assert len(d) == 480
    assert set(d.classification) == {"FIRST_ORDER_RESOLVABLE"}
    assert d.local_slope_p.between(1.5, 2.5).all()


def test_delay_keeps_right_censored_cases():
    d = pd.read_csv(RES / "exact_resolution_delay.csv")
    assert set(d.q) == {0.5, 0.8, 0.9, 0.95}
    assert d.tau_S_censored.any()
    assert d.tau_M_censored.any()
    assert set(pd.read_csv(RES / "censoring_analysis.csv").horizon_frames) == {5, 10, 15, 20, 25, 30}
