"""Leakage-safe linear-Gaussian estimator for E04-A.

Only observed measurement arrays enter this module. Ground truth lives in a
separate evaluation module and is never imported here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def interleaved_to_complex(values):
    """Convert [real_1, imag_1, ...] channels to complex phasors."""
    a = np.asarray(values, dtype=float)
    if a.shape[-1] % 2:
        raise ValueError("interleaved phasor channels must have even width")
    return a[..., 0::2] + 1j * a[..., 1::2]


def complex_to_interleaved(values):
    """Convert complex phasors to [real_1, imag_1, ...] channels."""
    z = np.asarray(values)
    out = np.empty(z.shape[:-1] + (2 * z.shape[-1],), dtype=float)
    out[..., 0::2] = z.real
    out[..., 1::2] = z.imag
    return out


def wrapped_angle_error(angle_hat, angle_true):
    """Shortest signed angular error in radians, including the +/-pi seam."""
    return np.arctan2(np.sin(np.asarray(angle_hat) - np.asarray(angle_true)),
                      np.cos(np.asarray(angle_hat) - np.asarray(angle_true)))


def best_global_rotation(v_true_observed, v_hat_observed, weights=None):
    """Return the per-frame rotation that maps estimate into truth coordinates.

    Only the supplied observed PMU phasors are used.  The convention follows
    arg(sum w V_true conj(V_hat)); the returned complex number has unit modulus.
    """
    vt = np.asarray(v_true_observed, dtype=complex)
    vh = np.asarray(v_hat_observed, dtype=complex)
    if vt.shape != vh.shape:
        raise ValueError("true and estimated observed phasors must have equal shape")
    w = 1.0 if weights is None else np.asarray(weights)
    score = np.sum(w * vt * np.conj(vh), axis=-1)
    return np.exp(1j * np.angle(score))


def phasor_metrics(v_hat_abs, v_true_abs):
    """Coordinate-correct phasor metrics on absolute complex voltages."""
    vh = np.asarray(v_hat_abs, dtype=complex)
    vt = np.asarray(v_true_abs, dtype=complex)
    err = vh - vt
    dtheta = wrapped_angle_error(np.angle(vh), np.angle(vt))
    tve = np.abs(err) / np.maximum(np.abs(vt), 1e-12)
    return {
        "complex_rmse": float(np.sqrt(np.mean(np.abs(err) ** 2))),
        "re_rmse": float(np.sqrt(np.mean(err.real ** 2))),
        "im_rmse": float(np.sqrt(np.mean(err.imag ** 2))),
        "vm_rmse": float(np.sqrt(np.mean((np.abs(vh) - np.abs(vt)) ** 2))),
        "angle_rmse": float(np.sqrt(np.mean(np.rad2deg(dtheta) ** 2))),
        "TVE_fraction": float(np.mean(tve)),
        "TVE_percent": float(100.0 * np.mean(tve)),
    }

@dataclass
class LinearGaussianModel:
    A: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    P0: np.ndarray

    def predict(self, x, P):
        return self.A @ x, self.A @ P @ self.A.T + self.Q

    def update(self, xpred, Ppred, y):
        S = self.C @ Ppred @ self.C.T + self.R
        K = np.linalg.solve(S, self.C @ Ppred).T
        innov = y - self.C @ xpred
        x = xpred + K @ innov
        I = np.eye(Ppred.shape[0])
        M = I - K @ self.C
        P = M @ Ppred @ M.T + K @ self.R @ K.T
        P = (P + P.T) * 0.5
        return x, P, innov, S

def snapshot_wls(model: LinearGaussianModel, y):
    S = model.C @ model.P0 @ model.C.T + model.R
    K = np.linalg.solve(S, model.C @ model.P0).T
    x = K @ y
    P = (model.P0 - K @ model.C @ model.P0)
    return x, (P + P.T) * 0.5

def kalman_filter(model: LinearGaussianModel, measurements):
    x = np.zeros(model.A.shape[0]); P = model.P0.copy(); out=[]
    for y in measurements:
        xp, Pp = model.predict(x, P)
        x, P, innov, S = model.update(xp, Pp, y)
        out.append((x.copy(), P.copy(), innov.copy(), S.copy()))
    return out

def fixed_lag_filter(model: LinearGaussianModel, measurements, lag: int):
    """Exact fixed-lag RTS estimate; output at k uses measurements ≤ k only."""
    filtered=[]; xp_hist=[]; pp_hist=[]; gains=[]; x=np.zeros(model.A.shape[0]); P=model.P0.copy()
    outputs=[None]*len(measurements)
    for k,y in enumerate(measurements):
        xp,Pp=model.predict(x,P); x,P,innov,S=model.update(xp,Pp,y)
        filtered.append((x.copy(),P.copy())); xp_hist.append(xp.copy()); pp_hist.append(Pp.copy())
        gains.append(None if k == 0 else np.linalg.solve(pp_hist[k].T, (filtered[k-1][1] @ model.A.T).T).T)
        j=k-lag
        if j < 0: continue
        xs,Ps=filtered[j]
        for t in range(j,k):
            xf,Pf=filtered[t]; xp1= xp_hist[t+1]; pp1=pp_hist[t+1]
            G=gains[t+1]
            xs=xf + G @ (xs-xp1); Ps=Pf + G @ (Ps-pp1) @ G.T; Ps=(Ps+Ps.T)*0.5
        outputs[j]=(xs,Ps)
    return outputs
