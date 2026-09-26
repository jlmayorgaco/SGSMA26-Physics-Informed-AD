import numpy as np
from pmu_hybrid.e04a_estimator import (
    LinearGaussianModel, snapshot_wls, kalman_filter, fixed_lag_filter, fixed_lag_filter_fast, fixed_lag_filter_fast_multi,
    interleaved_to_complex, complex_to_interleaved, wrapped_angle_error,
    best_global_rotation, phasor_metrics,
)

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


def test_absolute_voltage_reconstruction_and_complex_convention():
    v0 = np.array([1.0 + 0.2j, 0.9 - 0.1j])
    dv = np.array([0.01 - 0.02j, -0.03 + 0.01j])
    encoded = complex_to_interleaved(v0 + dv)
    assert np.allclose(interleaved_to_complex(encoded), v0 + dv)
    # Treating dv itself as an absolute phasor is intentionally not equivalent.
    assert not np.allclose(np.angle(interleaved_to_complex(complex_to_interleaved(dv))), np.angle(v0 + dv))


def test_gauge_rotation_invariance_and_reference_angle_metric():
    truth = np.array([[1.0 + 0j, 0.9 + 0.1j]])
    estimate = truth * np.exp(-1j * 0.37)
    alpha = best_global_rotation(truth, estimate)
    assert np.allclose(estimate * alpha[:, None], truth)
    rel_t = np.angle(truth[:, 1] / truth[:, 0]); rel_e = np.angle(estimate[:, 1] / estimate[:, 0])
    assert np.allclose(wrapped_angle_error(rel_e, rel_t), 0.0)


def test_angle_wrap_at_pi_and_radian_degree_conversion():
    e = wrapped_angle_error(np.array([-np.pi + 1e-6]), np.array([np.pi - 1e-6]))
    assert abs(e[0]) < 3e-6
    assert np.allclose(np.rad2deg(np.pi), 180.0)


def test_tve_fraction_and_percent_are_explicit():
    t = np.array([[1.0 + 0j]])
    p = np.array([[1.01 + 0j]])
    m = phasor_metrics(p, t)
    assert np.isclose(m['TVE_fraction'], 0.01)
    assert np.isclose(m['TVE_percent'], 1.0)


def test_noiseless_snapshot_wls_exact_same_model():
    m = LinearGaussianModel(np.eye(2), np.array([[1., 0.], [0., 1.]]),
                            np.eye(2) * 1e-9, np.eye(2) * 1e-12, np.eye(2) * 1e6)
    x = np.array([0.3, -0.7]); xhat, _ = snapshot_wls(m, m.C @ x)
    assert np.allclose(xhat, x, atol=1e-7)


def test_linear_kalman_sanity_finite_and_not_pathological():
    m = model(); x = np.array([0.4, -0.2]); ys = []
    rng = np.random.default_rng(4)
    for _ in range(20):
        x = m.A @ x
        ys.append(m.C @ x)
    out = kalman_filter(m, np.asarray(ys)); est = np.vstack([o[0] for o in out])
    assert np.all(np.isfinite(est))
    assert np.linalg.norm(est[-1]) < 2.0


def test_gauge_mode_covariance_is_conditioned_by_observation():
    A = np.diag([1.0, 0.8]); C = np.array([[1.0, 0.0]])
    m = LinearGaussianModel(A, C, np.eye(2) * 1e-9, np.array([[1e-4]]), np.eye(2))
    prior = m.P0[0, 0]; post = kalman_filter(m, np.zeros((1, 1)))[0][1][0, 0]
    assert post < prior


def test_cached_fixed_lag_matches_dense_oracle():
    m = model(); ys = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]])
    dense = fixed_lag_filter(m, ys, 2); fast = fixed_lag_filter_fast(m, ys, 2)
    for a, b in zip(dense, fast):
        if a is None: assert b is None
        else:
            assert np.allclose(a[0], b[0], atol=1e-10)
            assert np.allclose(a[1], b[1], atol=1e-10)
    assert np.allclose(dense[2][0], fixed_lag_filter_fast_multi(m, ys, [2])[2][2][0])
