"""Leakage-safe linear-Gaussian estimator for E04-A.

Only observed measurement arrays enter this module. Ground truth lives in a
separate evaluation module and is never imported here.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

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
