from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/"powerdynamics_ieee39"; R=ROOT/"output/results"

def test_standard_nominal_baseline_reproducibility():
    s=pd.read_csv(R/"e06_standard_summary.csv"); x=s[(s.method=="B2")&(s.m==0)]
    assert len(x)==7 and x.n.eq(20).all() and np.isfinite(x.median_TVE_percent).all()

def test_twenty_case_cell_and_seed_determinism():
    m=pd.read_csv(R/"e06_standard_manifest.csv")
    assert len(m)==980 and m.groupby(["family","m"]).size().eq(20).all()
    assert m.case_id.is_unique and m.groupby(["family","m"]).seed.apply(lambda x: sorted(x.tolist())==list(range(1,21))).all()

def test_checkpoint_resume_artifacts_complete():
    m=pd.read_csv(R/"e06_standard_manifest.csv"); d=R/"e06_standard_cases"
    assert sum((d/(c+"_meta.csv")).exists() and (d/(c+"_trajectory.csv")).exists() for c in m.case_id)==980

def test_estimator_hash_unchanged():
    for name,shape in [("e04_A.csv",(114,114)),("e04_C_pmu.csv",(32,114)),("e04_C_hidden.csv",(62,114))]:
        x=pd.read_csv(R/name); assert x.shape==shape

def test_cal_detector_leakage_audit():
    a=pd.read_csv(R/"e06_standard_adequacy.csv")
    assert len(a)==49 and a.cal_mu_norm.notna().all() and a.cal_cov_trace.notna().all()

def test_recovery_denominator_gate_and_oracle_subset():
    o=pd.read_csv(R/"e06_standard_oracle_subset.csv")
    assert len(o)>=42 and "Excess_N0_TVE_fraction" in o.columns

def test_excitation_stratification():
    m=pd.read_csv(R/"e06_standard_manifest.csv")
    for _,g in m.groupby(["family","m"]): assert set(g.excitation)=={"E-A","E-B","E-C","E-D"}

def test_confidence_interval_reproducibility_columns():
    s=pd.read_csv(R/"e06_standard_summary.csv")
    assert {"bootstrap95_low_percent","bootstrap95_high_percent"}.issubset(s.columns)
    assert (s.bootstrap95_high_percent>=s.bootstrap95_low_percent).all()
