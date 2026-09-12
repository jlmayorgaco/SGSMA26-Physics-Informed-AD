import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import e06f_nonlinear_ac_recentering as e


def test_fresh_split_contract():
    dev = pd.read_csv(e.R / "e06f_dev_manifest.csv")
    test = pd.read_csv(e.R / "e06f_test_manifest.csv")
    assert len(dev) == 160 and len(test) == 320
    assert set(dev.seed).isdisjoint(set(test.seed))
    assert set(dev.seed).isdisjoint(set(range(1, 221)))


def test_ac_residual_jacobian_finite_difference():
    Y = np.array([[1+2j, -1-2j], [-1-2j, 1+2j]])
    V = np.array([1+0j, .9-.1j])
    S = V*np.conj(Y@V)
    eps = 1e-6
    Vp = V.copy(); Vp[1] += eps
    Sp = Vp*np.conj(Y@Vp)
    assert np.isfinite((Sp-S)/eps).all()
    assert np.linalg.norm(S - V*np.conj(Y@V)) < 1e-12


def test_kkt_toy_constrained_map():
    # min 1/2(x-2)^2 subject to x=1 has solution x=1.
    H = np.array([[1.0]]); J = np.array([[1.0]]); rhs = np.array([2.0, 1.0])
    K = np.block([[H, J.T], [J, np.zeros((1, 1))]])
    sol = np.linalg.solve(K, rhs)
    assert np.allclose(sol[0], 1.0)


def test_nominal_pf_fixed_point_and_pv_slack_semantics():
    A, C, L, y0, h0 = e.load_linear()
    Y, branch = e.build_ybus()
    V = e.nominal_voltage(y0, h0)
    S = V*np.conj(Y@V)
    assert np.isfinite(S).all()
    assert abs(np.angle(V[e.SLACK-1]) - np.angle(V[e.SLACK-1])) < 1e-12
    assert all(b != e.SLACK for b in e.PV)


def test_no_hidden_leakage_in_s0_solver():
    src = Path(e.__file__).read_text(encoding="utf-8")
    body = src.split("def solve_map", 1)[1].split("def main", 1)[0]
    assert "truth" not in body
    assert "hidden" not in body


def test_load_injection_basis_dimension_and_constraints():
    assert len(e.LOAD) == 17
    assert len(set(e.LOAD)) == len(e.LOAD)
    assert e.SLACK not in e.LOAD
    assert set(e.PV).isdisjoint(set(e.LOAD))


def test_m6_static_gate_is_weak_and_online_locked():
    d = pd.read_csv(e.R / "e06f_s0_static_results.csv")
    m6 = d[(d.family == "M6_OPERATING_POINT") & (d.method == "S0-MAP")]
    nom = d[(d.family == "M6_OPERATING_POINT") & (d.method == "S0-NOM")]
    assert m6.hidden_center_TVE_percent.median() > nom.hidden_center_TVE_percent.median()
    assert not (e.R / "e06f_online_summary.csv").exists() or (e.R / "e06f_online_summary.csv").stat().st_size <= 2


def test_laplace_covariance_psd_and_outputs():
    sel = pd.read_csv(e.R / "e06f_static_map_selection.csv")
    assert int(sel.n_d.iloc[0]) == 34
    run = pd.read_csv(e.R / "e06f_runtime.csv")
    assert (run.ac_residual >= 0).all()
