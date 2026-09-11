import numpy as np
from pmu_hybrid.e04a_estimator import LinearGaussianModel, snapshot_wls, kalman_filter, fixed_lag_filter

def model():
    A=np.array([[.9,.1],[0,.8]]); C=np.array([[1.,0.],[0.,1.]])
    return LinearGaussianModel(A,C,np.eye(2)*.01,np.eye(2)*.04,np.eye(2))

def test_kalman_update_matches_reference_equation():
    m=model(); y=np.array([1.,-2.]); xp=np.zeros(2); Pp=m.P0
    x,P,_,_=m.update(xp,Pp,y); S=m.C@Pp@m.C.T+m.R; K=np.linalg.solve(S,m.C@Pp).T
    assert np.allclose(x,K@y); assert np.allclose(P,P.T); assert np.linalg.eigvalsh(P).min() >= -1e-12

def test_snapshot_batch_map_equals_closed_form():
    m=model(); y=np.array([.3,-.4]); x,P=snapshot_wls(m,y); K=np.linalg.solve(m.C@m.P0@m.C.T+m.R,m.C@m.P0).T
    assert np.allclose(x,K@y); assert np.allclose(P,P.T)

def test_fixed_lag_causality_and_lag_indexing():
    m=model(); ys=np.array([[0.,0.],[1.,0.],[2.,0.],[3.,0.]])
    a=fixed_lag_filter(m,ys,1); b=fixed_lag_filter(m,np.vstack([ys,[[100.,100.]]]),1)
    assert a[0] is not None and b[0] is not None and np.allclose(a[0][0],b[0][0])

def test_no_hidden_state_import_and_observed_api_shape():
    import inspect, pmu_hybrid.e04a_estimator as est
    assert 'evaluation_ground_truth' not in inspect.getsource(est)
