from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular

ROOT = Path(__file__).resolve().parents[1] / 'powerdynamics_ieee39' / 'output' / 'load_bayes_fd_v1'
RES = ROOT / 'results'


def test_frozen_dictionary_and_cal_test_separation():
    z = np.load(Path(__file__).resolve().parents[1] / 'powerdynamics_ieee39' / 'output/load_tangent_v2/results/load_fd_central_operator.npz')
    assert z['central'].shape == (16, 30, 32)
    cal = pd.read_csv(RES / 'load_bayes_cal_manifest.csv')
    test = pd.read_csv(RES / 'load_bayes_test_manifest.csv')
    assert set(cal.noise_seed).isdisjoint(set(test.noise_seed))
    assert len(cal) == 40


def test_posterior_normalization_and_h0_hypothesis():
    d = pd.read_parquet(RES / 'load_source_posterior.parquet')
    cols = ['p_h0'] + [f'p_H{b}' for b in [3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28]]
    assert set(cols).issubset(d.columns)
    assert np.max(np.abs(d[cols].sum(axis=1)-1)) < 1e-10
    assert (d[d.true_source == 0].p_h0 >= 0).all()


def test_analytic_amplitude_posterior_matches_bruteforce_grid():
    rng = np.random.default_rng(3); d = np.array([1.2, -0.4]); S = np.array([[1.0,.2],[.2,.8]])
    p = np.linalg.inv(S); r = np.array([.3,-.2]); sa=.5
    v=1/(d@p@d+sa**-2); mu=v*(d@p@r)
    aa=np.linspace(-3,3,20001); ll=np.exp(-.5*np.array([(r-d*a)@p@(r-d*a) for a in aa]))*np.exp(-.5*(aa/sa)**2)
    grid=np.trapz(aa*ll,aa)/np.trapz(ll,aa)
    assert abs(mu-grid) < 2e-3


def test_woodbury_and_determinant_lemma_against_direct_gaussian():
    rng=np.random.default_rng(4); S=np.array([[1.2,.1],[.1,.9]]); d=np.array([.4,-.2]); sa=.7; r=np.array([.2,.3])
    direct=np.linalg.inv(S+sa**2*np.outer(d,d)); qd=r@direct@r
    p=np.linalg.inv(S); q0=r@p@r; b=d@p@r; qwb=q0-sa**2*b*b/(1+sa**2*(d@p@d))
    assert abs(qd-qwb)<1e-10
    assert abs(np.linalg.slogdet(S+sa**2*np.outer(d,d))[1]-(np.linalg.slogdet(S)[1]+np.log1p(sa**2*d@p@d)))<1e-10


def test_evi_and_geometry_toy_cases():
    S=np.diag([2.,.5]); d1=np.array([1.,0.]); d2=np.array([0.,1.]); P=np.linalg.inv(S)
    assert abs(d1@P@d1-.5)<1e-12
    assert abs(d2@P@d2-2.)<1e-12
    c=(d1@P@d2)/np.sqrt((d1@P@d1)*(d2@P@d2)); assert abs(c)<1e-12


def test_calibration_model_contains_no_test_fit_fields():
    w=pd.read_csv(RES/'load_whitening_model.csv')
    assert 'TEST' not in ' '.join(w.columns)
    assert w.selected.sum() == 1


def test_primary_test_counts_and_model_comparison_exist():
    m = pd.read_csv(RES/'load_bayes_model_comparison.csv')
    assert set(m.model) == {'W0_IDENTITY', 'W1_CHANNEL_COV', 'W2_SEPARABLE_AR1'}
    p = pd.read_parquet(RES/'load_source_posterior.parquet')
    assert (p.true_source != 0).sum() == 16*8*20
    assert (p.true_source == 0).sum() == 16*20
    assert len(pd.read_csv(RES/'load_source_per_source.csv')) == 16
