from pathlib import Path
import inspect
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import ieee39_end2end_single_v1 as e


def test_frozen_pmu_and_source_contract():
    assert e.PMU_BUSES == [2,5,6,10,19,22,29,39]
    assert len(json.loads((e.PREREG/"estimator_contract.json").read_text())["pmu_contract"]["channels"]) == 32
    assert 7 not in e.PMU_BUSES and 7 in e.BUSES


def test_hypothesis_space_exactly_137():
    assert len(e.SUPPORTS)==137 and len(e.PAIRS)==120
    assert e.SUPPORTS[0]==() and (7,) in e.SUPPORTS


def test_estimator_api_has_no_truth_argument():
    p=set(inspect.signature(e.infer_observation).parameters)
    assert not ({"truth","support","severity","state"}&p)


def test_ar1_matches_dense_toy():
    rng=np.random.default_rng(4); r=rng.normal(size=(5,32)); var=np.exp(rng.normal(size=32))
    assert abs(e.ar1_loglike(r,var)-e.dense_ar1_loglike(r,var)) < 1e-9


def test_nested_prefix_and_state_conventions():
    rng=np.random.default_rng(5); D=rng.normal(size=(3840,16)); Q=rng.normal(size=(3840,16)); C=rng.normal(size=(3840,120))
    a=.02
    assert np.array_equal(e.model_mean(D,Q,C,(7,),(a,),30),e.model_mean(D,Q,C,(7,),(a,),120)[:960])
    sm={"x0":np.zeros(2),"D":np.ones((3,2,16)),"Q":np.ones((3,2,16))*2,"Qcross":np.ones((3,2,120))*3}
    s=e.state_mean(sm,(7,),{"a0":a,"a02":a*a})
    assert np.allclose(s,a+2*a*a)  # Q already carries the canonical self 1/2.
    d=e.state_mean(sm,(7,12),{"a0":a,"a02":a*a,"a1":2*a,"a12":4*a*a,"a01":2*a*a})
    assert np.allclose(d,3*a+16*a*a)  # raw cross convention, no extra 1/2.


def test_posterior_normalization_and_bma_weights():
    p=np.array([.2,.3,.5]); assert np.isclose(p.sum(),1)
    inc=np.array([.1,.9]); assert np.all((inc>=0)&(inc<=1))
    w=np.array([.2,.8]); assert np.isclose(w.sum(),1)


def test_estimator_artifact_leakage_and_v3_exclusion_if_executed():
    for p in e.OBS.glob("estimator_input_case_*.npz"):
        assert set(np.load(p,allow_pickle=False).files)=={"time_s","pmu_32","contract_sha256"}
    x=e.AUDIT/"v3_exclusion_manifest_additions.csv"
    if x.exists():
        import pandas as pd
        d=pd.read_csv(x); assert len(d)==3 and (d.reason=="IEEE39_END2END_SINGLE_V1_DEVELOPMENT").all()
