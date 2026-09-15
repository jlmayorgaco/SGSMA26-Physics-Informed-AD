from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "powerdynamics_ieee39" / "output" / "analytic_second_order_dae_v1"
RES = EXP / "results"


def test_native_descriptor_inventory_and_hessian_symmetry():
    state = pd.read_csv(EXP / "metadata" / "state_order.csv")
    assert len(state) == 192
    assert int((state.kind == "differential").sum()) == 114
    assert int((state.kind == "algebraic").sum()) == 78
    h = pd.read_csv(RES / "hessian_contraction_checks.csv")
    assert h.finite.all()
    # The first row stores the independent centered contraction check; the
    # remaining rows are the explicit mixed-partial symmetry checks.
    assert float(h.symmetry_max.iloc[1:].max()) < 1e-10


def test_first_and_second_order_contracts():
    first = pd.read_csv(RES / "first_order_regression.csv")
    self_q = pd.read_csv(RES / "analytic_vs_frozen_Qself.csv")
    cross_q = pd.read_csv(RES / "analytic_vs_frozen_Qcross.csv")
    assert len(first) == 16 and first.relative_l2_error.max() < 1e-3
    assert len(self_q) == 16 and self_q.cosine.min() > 0.995
    assert len(cross_q) == 120 and cross_q.cosine.min() > 0.995


def test_onset_and_parameter_semantics():
    onset = pd.read_csv(RES / "algebraic_onset_checks.csv")
    assert len(onset) == 16 and (onset.second_order_status == "COMPUTED").all()
    pmap = pd.read_csv(RES / "event_parameter_map.csv")
    assert len(pmap) == 16 and np.allclose(pmap.p_second_derivative, 0) and np.allclose(pmap.q_second_derivative, 0)


def test_fresh_fd_reference_is_separate_and_excluded():
    fd = pd.read_csv(RES / "analytic_vs_fresh_fd.csv")
    manifest = pd.read_csv(RES / "v3_exclusion_manifest_delta.csv")
    assert len(fd) == 12 and (fd.fresh_fd_status == "PASS").all()
    assert bool(manifest.future_v3_excluded.iloc[0])


def test_ac_hessian_and_measurement_curvature():
    ac = pd.read_csv(RES / "ac_network_hessian_audit.csv")
    assert ac.relative_error.max() < 1e-3
    meas = pd.read_csv(RES / "measurement_second_order_audit.csv")
    assert np.max(np.abs(meas.second_directional_value)) == 0.0
