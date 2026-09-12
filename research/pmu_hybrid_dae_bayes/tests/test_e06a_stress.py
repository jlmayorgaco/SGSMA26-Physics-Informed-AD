import numpy as np
from pmu_hybrid.e06a_stress import SCALES, perturb_frame, validity_check
from pathlib import Path
import pandas as pd


def test_m_zero_is_exact_identity_and_seed_is_deterministic():
    p=np.ones((3,32)); h=np.ones((3,62));
    p0,h0=perturb_frame(p,h,"M1_NETWORK",0.0,4)
    assert np.array_equal(p,p0) and np.array_equal(h,h0)
    a=perturb_frame(p,h,"M3_GOVERNOR",1.0,99); b=perturb_frame(p,h,"M3_GOVERNOR",1.0,99)
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])


def test_physical_validity_rejects_nonfinite_only():
    p=np.ones((2,32)); h=np.ones((2,62)); assert validity_check(p,h)[0]
    h[0,0]=np.nan; assert not validity_check(p,h)[0]


def test_scale_grid_is_frozen():
    assert SCALES == (0.0,.25,.50,.75,1.0,1.25,1.50)


def test_nominal_exports_and_frozen_manifest_contract():
    root=Path(__file__).parents[1]/"powerdynamics_ieee39"/"output"/"results"
    A=pd.read_csv(root/"e04_A.csv").to_numpy(); C=pd.read_csv(root/"e04_C_pmu.csv").to_numpy()
    assert A.shape==(114,114) and C.shape==(32,114)
    manifest=pd.read_csv(root/"e06a_manifest.csv")
    assert len(manifest)==147 and manifest.seed.is_unique and manifest.valid.all()
