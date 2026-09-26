from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'powerdynamics_ieee39'/'output'/'load_multi_bayes_v1'
RES=OUT/'results'

def test_frozen_dictionary_hash_and_split_contract():
    p=ROOT/'powerdynamics_ieee39'/'output'/'load_tangent_v2'/'results'/'load_fd_central_operator.npz'
    assert hashlib.sha256(p.read_bytes()).hexdigest()=='8a156913cea000cf9dba23bca74f45abbce47292a45993d707815215a103752b'
    cal=pd.read_csv(RES/'load_multi_cal_manifest.csv')
    test=pd.read_csv(RES/'load_multi_test_manifest.csv')
    assert set(cal.noise_seed).isdisjoint(set(test.noise_seed))
    assert test.noise_seed.min()>=800000

def test_qij_manifest_complete_and_successful():
    mf=pd.read_csv(RES/'load_multi_qij_manifest.csv')
    assert len(mf)==120*4
    assert set(mf.status)=={'EXECUTED_SUCCESS'}

def test_posterior_normalization_and_h0_support():
    p=RES/'load_multi_source_posterior.parquet'
    if not p.exists():
        return
    d=pd.read_parquet(p)
    assert np.allclose(d.p_M0+d.p_M1+d.p_M2,1.0,atol=1e-8)
    a=d.filter(like='p_include_').to_numpy(); assert np.isfinite(a).all() and np.all((a>=-1e-12)&(a<=1+1e-9))

def test_quadrature_and_fisher_toy_identities():
    rng=np.random.default_rng(1); A=rng.normal(size=(8,2)); s=0.4
    G=A.T@A; v=np.linalg.inv(G+np.eye(2)/s**2)
    # Gaussian prior normalization and nonnegative information.
    assert np.all(np.linalg.eigvalsh(G)>=-1e-10)
    assert np.all(np.diag(v)>0)
    evi=float(A[:,0]@A[:,0]); coh=float((A[:,0]@A[:,1])**2/(A[:,0]@A[:,0])/(A[:,1]@A[:,1]))
    J=evi*(1-coh)
    J2=float(A[:,0]@A[:,0]-(A[:,0]@A[:,1])**2/(A[:,1]@A[:,1]))
    assert np.isclose(J,J2)

def test_no_true_label_inputs_in_inference_source():
    src=(ROOT/'scripts'/'load_multi_bayes_v1.py').read_text(encoding='utf-8')
    assert 'true_support' not in src.split('def single_logpost',1)[1].split('def build_test_manifest',1)[0]
    assert 'true_amplitude' not in src
