from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "powerdynamics_ieee39" / "output" / "second_order_op_robustness_v1"
RES = EXP / "results"


def test_operating_point_domain_and_feasibility():
    op = pd.read_csv(RES / "operating_points.csv")
    assert len(op) == 4
    assert set(op.op_m) == {0.0, 0.5, 1.0, 1.5}
    assert (op.physical_feasibility == "PASS").all()
    assert np.all(np.isfinite(op.sigma_min_gz))


def test_all_self_terms_and_richardson_rows():
    q = pd.read_csv(RES / "richardson_self.csv")
    assert len(q) == 64
    assert set(q.op_tag) == {"nominal", "lower_load_m05", "interior_load_m10", "higher_load_m15"}
    assert np.isfinite(q.relative_l2_error).all()
    assert (q.richardson_uncertainty < 0.01).all()


def test_cross_subset_has_three_anchor_points():
    q = pd.read_csv(RES / "richardson_cross.csv")
    assert len(q) == 30
    assert set(q.op_tag) == {"nominal", "lower_load_m05", "higher_load_m15"}
    assert set(q.pair) == {"3-4", "3-7", "3-12", "7-12", "7-20", "8-28", "12-15", "20-21", "23-24", "27-28"}


def test_first_order_and_affinity():
    d = pd.read_csv(RES / "first_order_by_operating_point.csv")
    assert len(d) == 4 * 16 * 3
    assert d.relative_l2_error.max() < 1e-3
    a = pd.read_csv(RES / "measurement_affinity.csv")
    assert len(a) == 4 and (a.status == "PASS").all()
    assert np.max(np.abs(a.max_abs_second_directional)) == 0.0


def test_onset_and_conditioning_outputs():
    o = pd.read_csv(RES / "onset_robustness.csv")
    c = pd.read_csv(RES / "algebraic_conditioning.csv")
    assert len(o) == 64 and (o.status == "PASS_SAMPLED_ONSET_PROXY").all()
    assert len(c) == 64
    assert np.isfinite(c.kappa_gz).all()


def test_v3_exclusion_and_summary():
    m = pd.read_csv(RES / "v3_exclusion_manifest_delta.csv")
    assert len(m) == 4 and m.future_v3_excluded.all()
    s = pd.read_json(RES / "summary.json", typ="series")
    assert int(s["operating_points"]) == 4
    assert s["affinity"] == "PASS"
