from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from scripts.e06d_oracle_relinearization import load_case, score_case, CASES, R, N, NY, NH

NOM = load_case("NOMINAL_m0_s1")

def test_same_trajectory_supplied_to_all_methods(monkeypatch):
    seen=[]
    import scripts.e06d_oracle_relinearization as e
    real=e.run_filter
    def wrapped(A,C,y0,measurements):
        seen.append(np.asarray(measurements).copy()); return real(A,C,y0,measurements)
    monkeypatch.setattr(e,"run_filter",wrapped)
    score_case(load_case("M7-COUPLED_m1p0_s1"), NOM)
    assert len(seen)==3 and np.array_equal(seen[0],seen[1]) and np.array_equal(seen[1],seen[2])

def test_o1_changes_offsets_only(monkeypatch):
    import scripts.e06d_oracle_relinearization as e
    captured=[]; real=e.run_filter
    def wrapped(A,C,y0,measurements): captured.append((A.copy(),C.copy(),y0.copy())); return real(A,C,y0,measurements)
    monkeypatch.setattr(e,"run_filter",wrapped)
    case=load_case("M1-NETWORK_m1p0_s1"); score_case(case,NOM)
    assert np.array_equal(captured[0][0],captured[1][0]) and np.array_equal(captured[0][1],captured[1][1])
    assert np.allclose(captured[1][2],case["y0"])

def test_o2_uses_physically_rebuilt_matrices():
    case=load_case("M7-COUPLED_m1p5_s1")
    assert case["A"].shape==(N,N) and case["C"].shape==(NY,N) and case["L"].shape==(NH,N)
    assert np.linalg.norm(case["A"]-NOM["A"])>1e-8 or np.linalg.norm(case["C"]-NOM["C"])>1e-8

def test_nominal_controls_equivalent():
    p=pd.read_csv(R/"e06d_per_case.csv"); g=p[p.case_id.str.startswith("NOMINAL")].groupby("method").TVE_fraction.median()
    assert max(g)-min(g)<1e-5

def test_state_coordinate_mapping_identity():
    m=pd.read_csv(R/"e06d_state_coordinate_map.csv")
    assert len(m)>0 and m.same.all() and (m.mapping=="identity").all()

def test_no_true_parameter_leakage_into_n0(monkeypatch):
    import scripts.e06d_oracle_relinearization as e
    seen=[]; real=e.run_filter
    def wrapped(A,C,y0,measurements): seen.append((A.copy(),C.copy(),y0.copy())); return real(A,C,y0,measurements)
    monkeypatch.setattr(e,"run_filter",wrapped)
    case=load_case("M1-NETWORK_m1p0_s1"); score_case(case,NOM)
    assert np.array_equal(seen[0][0],NOM["A"]) and np.array_equal(seen[0][1],NOM["C"]) and np.array_equal(seen[0][2],NOM["y0"])

def test_recovery_fraction_calculation():
    p=pd.read_csv(R/"e06d_recovery_fractions.csv").dropna(subset=["Excess_N0_TVE_fraction"])
    q=p[p.Excess_N0_TVE_fraction>0].iloc[0]
    expected=1-q.Excess_O2_TVE_fraction/q.Excess_N0_TVE_fraction
    assert np.isclose(q.Recovery_Relinearize_TVE_fraction,expected)

def test_oracle_local_convergence_checks():
    c=pd.read_csv(R/"e06d_oracle_linearization_checks.csv")
    assert set(c.family)=={"M1_NETWORK","M5_LOAD_MODEL","M6_OPERATING_POINT","M7_COUPLED"}
    assert (c.order_assessment=="O(eps^2) consistent").all()
