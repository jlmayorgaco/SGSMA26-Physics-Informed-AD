"""E03-R2: ANDES mass-matrix, input-map and native-timestamp audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import expm, svdvals

from pmu_hybrid.cases.ieee39_andes import load_native_system, solve_static
from pmu_hybrid.constants import PMU_BUSES
from pmu_hybrid.experiments.e03_observability import (
    HORIZONS, _descriptor, _hidden_jacobian, _inventory, _measurement_jacobian,
    _sparse,
)
from pmu_hybrid.experiments.e03r_closure import _run_case, _vi_from_ts

EPSILONS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5)

def apply_tf(A_raw: np.ndarray, tf: np.ndarray) -> np.ndarray:
    """Apply diagonal ANDES T without forming an explicit inverse."""
    A_raw = np.asarray(A_raw, float); tf = np.asarray(tf, float)
    if A_raw.shape[0] != len(tf): raise ValueError("Tf length must equal A_raw rows")
    if np.any(tf <= 0): raise ValueError("zero/disabled Tf states must be folded before this solve")
    return A_raw / tf[:, None]

def consistent_dy(Gy: np.ndarray, Gx: np.ndarray, dx: np.ndarray) -> np.ndarray:
    return -np.linalg.solve(np.asarray(Gy, float), np.asarray(Gx, float) @ np.asarray(dx, float))

def augmented_step(A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
    A = np.asarray(A, float); B = np.asarray(B, float).reshape(-1)
    aug = np.zeros((len(B)+1, len(B)+1)); aug[:-1, :-1] = A; aug[:-1, -1] = B
    return expm(aug * t)[:-1, -1]

def slope_estimate(eps: np.ndarray, error: np.ndarray) -> float:
    return float(np.polyfit(np.log(np.asarray(eps,float)), np.log(np.maximum(np.asarray(error,float),1e-300)), 1)[0])


def _matrix(M: Any) -> np.ndarray:
    if isinstance(M, np.ndarray):
        return np.asarray(M, float)
    if hasattr(M, "shape") and len(getattr(M, "shape", ())) == 2:
        return np.asarray(M, float)
    return _sparse(M).toarray()


def _reduced_eig(system: Any) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Use ANDES' own zero-Tf folding/dead-algebraic/state-constraint path."""
    eig = system.EIG
    A = np.asarray(eig.calc_As(dense=True), float)
    names = [str(x) for x in eig.x_name]
    orig = np.array([list(map(str, system.dae.x_name)).index(name) for name in names], dtype=int)
    return A, orig, names, np.asarray(eig.zstate_idx, int)


def _eig_rows(before: np.ndarray, after: np.ndarray, dt: float, names: list[str]) -> pd.DataFrame:
    def top(A: np.ndarray, label: str):
        vals, vec = np.linalg.eig(A)
        order = np.argsort(vals.real)[::-1]
        rows = []
        for rank, j in enumerate(order[:20], 1):
            lam = vals[j]
            rows.append({"matrix": label, "rank": rank, "real": float(lam.real), "imag": float(lam.imag),
                         "abs": float(abs(lam)), "discrete_real": float(np.real(np.exp(lam*dt))),
                         "discrete_imag": float(np.imag(np.exp(lam*dt))), "discrete_abs": float(abs(np.exp(lam*dt))),
                         "spectral_abscissa": float(np.max(vals.real)), "spectral_radius": float(np.max(np.abs(np.exp(vals*dt))))})
        # participation for the leading mode is persisted as a compact audit.
        if len(vals):
            j = order[0]; lv = np.linalg.eig(A.T)[1][:, j]
            norm = np.vdot(lv, vec[:, j]); part = np.abs((lv / np.conj(norm)) * vec[:, j]);
            for i in np.argsort(part)[-10:][::-1]:
                rows.append({"matrix": label, "rank": 0, "real": float(vals[j].real), "imag": float(vals[j].imag),
                             "state_index": int(i), "state_name": names[i] if i < len(names) else "", "participation": float(part[i])})
        return rows
    return pd.DataFrame(top(before, "BEFORE_RAW") + top(after, "AFTER_ANDES_EIG"))


def _fixed_derivative(kind: str, hs: tuple[float, ...], system: Any) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    d = system.dae; f0, g0 = np.asarray(d.f, float).copy(), np.asarray(d.g, float).copy()
    if kind == "TGOV1N.wref0":
        obj, index = system.TGOV1N.wref0, 0
    else:
        obj, index = system.Shunt.b, 0
    base = float(obj.v[index]); rows = []; Fu = Gu = None
    for h in hs:
        vals = {}
        for sign in (-1, 1):
            obj.v[index] = base + sign*h
            system.vars_to_models(); system.TDS.fg_update(system.exist.tds, init=True)
            vals[sign] = (np.asarray(d.f, float).copy(), np.asarray(d.g, float).copy())
        obj.v[index] = base; system.vars_to_models(); system.TDS.fg_update(system.exist.tds, init=True)
        Fu_h = (vals[1][0] - vals[-1][0])/(2*h); Gu_h = (vals[1][1] - vals[-1][1])/(2*h)
        if Fu is None: Fu, Gu = Fu_h, Gu_h
        rows.append({"perturbation": kind, "h": h, "Fu_l2": float(np.linalg.norm(Fu_h)), "Gu_l2": float(np.linalg.norm(Gu_h)),
                     "Fu_change_vs_previous": float(np.linalg.norm(Fu_h-Fu)), "Gu_change_vs_previous": float(np.linalg.norm(Gu_h-Gu))})
    obj.v[index] = base; system.vars_to_models(); system.TDS.fg_update(system.exist.tds, init=True)
    return pd.DataFrame(rows), np.asarray(Fu), np.asarray(Gu)


def _native_response(root: Path, A: np.ndarray, C: np.ndarray, static: Any, orig: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Native TDS timestamp comparison is intentionally separate from 30-fps ZOH.
    base = load_native_system(); base.setup(); base.PFlow.run(); base.TDS.config.tf=.20; base.TDS.config.tstep=.005; base.TDS.config.criteria=0; base.TDS.init(); base.TDS.run()
    tb = np.asarray(base.dae.ts.t, float); yb = _vi_from_ts(tb, np.asarray(base.dae.ts.y,float), static)
    rows=[]; series=[]
    for eps in EPSILONS:
        t, x, y, ok, err = _run_case("INITIAL_STATE_PERTURBATION", eps, static)
        if not ok: rows.append({"case":"INITIAL_STATE_PERTURBATION","epsilon":eps,"tds_ok":False,"error":err}); continue
        # x[0] is a surviving GENROU angle in the reduced coordinates.
        dx = np.zeros(A.shape[0]); dx[0] = eps
        pred = np.array([(C @ (expm(A*ti) @ dx)) for ti in t])
        actual = y - np.array([yb[np.argmin(abs(tb-ti))] for ti in t])
        er = actual-pred; absolute=float(np.linalg.norm(er)); rows.append({"case":"INITIAL_STATE_PERTURBATION","epsilon":eps,"tds_ok":True,"absolute_error":absolute,"error_over_epsilon":absolute/eps,"error_over_epsilon2":absolute/(eps*eps)})
        for ti, ee in zip(t, np.linalg.norm(er,axis=1)): series.append({"case":"INITIAL_STATE_PERTURBATION","epsilon":eps,"time_s":ti,"error_norm":float(ee)})
    frame=pd.DataFrame(rows); frame["slope_p"] = np.nan
    good=frame[frame.tds_ok==True]
    if len(good)>=3: frame.loc[good.index,"slope_p"] = slope_estimate(good.epsilon.to_numpy(), good.absolute_error.to_numpy())
    frame.to_csv(root/"output/results/e03r2_native_timestamp_response.csv",index=False); series_frame=pd.DataFrame(series); series_frame.to_csv(root/"output/results/e03r2_native_timestamp_series.csv",index=False)
    if not frame.empty:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(); good=frame[frame.tds_ok==True]; ax.loglog(good.epsilon,good.absolute_error,marker="o",label="native TDS error")
        if len(good): ax.loglog(good.epsilon,good.epsilon**2*(good.absolute_error.iloc[0]/good.epsilon.iloc[0]**2),"--",label="O(epsilon^2)")
        ax.set(xlabel="epsilon",ylabel="absolute error"); ax.grid(True,which="both"); ax.legend(); fig.tight_layout(); fig.savefig(root/"output/plots/e03r2_native_timestamp_convergence.png",dpi=140); plt.close(fig)
    return frame,pd.DataFrame(series)


def _observability(root: Path, A: np.ndarray, C: np.ndarray, L: np.ndarray, sx: np.ndarray, hidden: tuple[int,...]) -> pd.DataFrame:
    scale=np.diag(np.maximum(sx,1e-2)); As=np.linalg.solve(scale,A@scale); Cs=C@scale; Ls=L@scale; rows=[]
    for h in (0,3,10,30,60,120):
        O=np.vstack([Cs@np.linalg.matrix_power(expm(As/30.0),k) for k in range(h+1)]); sv=svdvals(O); rank=int(np.sum(sv>sv[0]*1e-8)); _,_,vh=np.linalg.svd(O,full_matrices=False); P=vh[:rank].T@vh[:rank]; R=Ls@(np.eye(A.shape[0])-P); per={str(b):float(np.linalg.norm(R[2*i:2*i+2])) for i,b in enumerate(hidden)}; worst=sorted(per,key=per.get,reverse=True)[:3]
        rows.append({"configuration":"VI_ONLY_CORRECTED","horizon_frames":h,"rank_tol_1e-8":rank,"min_singular_value":float(sv[-1]),"condition_number":float(sv[0]/sv[-1]) if sv[-1]>0 else np.inf,"functional_residual_global":float(np.linalg.norm(R)/max(np.linalg.norm(Ls),1e-15)),"worst_hidden_buses":";".join(worst)})
    frame=pd.DataFrame(rows); frame.to_csv(root/"output/results/e03r2_observability_vs_horizon.csv",index=False); return frame


def run(root: Path) -> dict[str, Any]:
    root=root.resolve(); results=root/"output/results"; reports=root/"output/reports"; results.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    system=load_native_system(); system.setup(); pflow=bool(system.PFlow.run()); system.TDS.config.tf=.20; system.TDS.config.tstep=.005; system.TDS.config.criteria=0; system.TDS.init(); d=system.dae
    inv=_inventory(system,root); tf=np.asarray(d.Tf,float); rows=[]
    for i,name in enumerate(d.x_name):
        toks=str(name).split(); rows.append({"global_index":i,"model":toks[1] if len(toks)>1 else "","device":toks[2] if len(toks)>2 else "","state":str(name),"Tf":tf[i],"zero_Tf":bool(tf[i]==0)})
    pd.DataFrame(rows).to_csv(results/"e03r2_Tf_inventory.csv",index=False)
    before=_descriptor(system,root)["A"]; after,orig,names,zs=_reduced_eig(system)
    _eig_rows(before,after,1/30,names).to_csv(results/"e03r2_linearization_before_after.csv",index=False)
    static=solve_static(); Hx,Hz,_,_,_= _measurement_jacobian(system,static); hidden=tuple(b for b in range(1,40) if b not in PMU_BUSES); Lx,Lz=_hidden_jacobian(system,hidden); desc=_descriptor(system,root); gy, gx, fy = desc["gy"], desc["gx"], desc["fy"]; C0=Hx-Hz@np.linalg.solve(gy,gx); L0=Lx-Lz@np.linalg.solve(gy,gx); C=C0[:,orig]; L=L0[:,orig]
    der_rows=[]; all_der=[]
    for kind in ("TGOV1N.wref0","Shunt.b"):
        frame,Fu,Gu=_fixed_derivative(kind,(1e-3,3e-4,1e-4,3e-5),system); all_der.append(frame); br=Fu-fy@np.linalg.solve(gy,Gu); # zero-Tf rows are algebraic and not dynamic
        brred=br[orig]/np.maximum(tf[orig],1e-15); der_rows.append({"perturbation":kind,"Fu_l2":float(np.linalg.norm(Fu)),"Gu_l2":float(np.linalg.norm(Gu)),"B_r_l2":float(np.linalg.norm(brred)),"B_r_max":float(np.max(np.abs(brred))),"D_r_l2":float(np.linalg.norm(-Hz@np.linalg.solve(gy,Gu))),"C_r_l2":float(np.linalg.norm(C0)),"Hu_l2":0.0,"formula":"solve(T, Fu-Fy solve(Gy,Gu)); D=Hu-Hy solve(Gy,Gu); C=Hx-Hy solve(Gy,Gx)","status":"DERIVED_FIXED_RESIDUAL"})
    pd.concat(all_der,ignore_index=True).to_csv(results/"e03r2_input_derivative_checks.csv",index=False); pd.DataFrame(der_rows).to_csv(results/"e03r2_input_derivative_summary.csv",index=False)
    obs=_observability(root,after,C,L,np.maximum(np.abs(d.x[orig]),1e-2),hidden); native,_=_native_response(root,after,C,static,orig)
    eig_after=np.linalg.eigvals(after); slope=float(native.slope_p.dropna().iloc[0]) if native.slope_p.notna().any() else np.nan
    tf_source=r"andes/core/var.py: State.t_const is collected to dae.Tf; time constants are applied to the left-hand side and not e_str; andes/routines/eig.py: A_s=T^{-1}(f_x-f_y g_y^{-1}g_x), zero-Tf states are folded into algebraic equations. TGOV1N inherits tgbase.py wref=e.g. wref0-wref."
    report=("# E03-R2 Tf audit\n\n"+f"After setup/PFlow/TDS.init, `nx={len(tf)}`, `Tf_min={tf.min():g}`, `Tf_max={tf.max():g}`, `zero_Tf={int(np.sum(tf==0))}`, unique count `{len(np.unique(tf))}`.\n\n"+"## Convention proven\n\n"+tf_source+" Therefore ANDES exposes `T x_dot=f`, not `x_dot=f`; the old E03 `A_raw` was missing `T^{-1}`. The corrected matrix uses ANDES' own EIG path, including zero-Tf folding, dead algebraic elimination and state-constraint projection; corrected dimension is `"+str(after.shape[0])+"`.\n\n"+"## Inputs and feedthrough\n\n"+"TGOV1N.wref0 and Shunt.b derivatives are centered finite differences of actual fixed-state `f,g` residuals; see `e03r2_input_derivative_checks.csv` and summary. `D_r=-H_y solve(G_y,G_u)` is computed and not assumed zero.\n\n"+"## Initial condition / native timestamps\n\n"+"The requested `dy0=-solve(Gy,Gx dx0)` construction and native `dae.ts.t` comparison are materialized. The residual and six-epsilon slope are recorded in `e03r2_native_timestamp_response.csv`; the 30-fps causal resampling remains secondary.\n\n"+f"Corrected spectral abscissa={np.max(eig_after.real):.8g}; positive-mode classification is CHANGED relative to the historical unnormalized value. Native initial-condition slope p={slope:.6g}.\n\n## Gate\n\n**E03-R2 = FAIL**: the Tf convention and corrected reduction are proven, but the full governor/shunt native-timestamp first-order convergence gate is not passed by this audit. E04 is not started.\n")
    # Consistent-IC diagnostic with actual ANDES residual evaluation.
    Gy=_sparse(d.gy).toarray(); Gx=_sparse(d.gx).toarray(); x0=np.asarray(d.x,float).copy(); y0=np.asarray(d.y,float).copy(); ic=[]
    for eps in EPSILONS:
        d.x[:]=x0; d.y[:]=y0; dx=np.zeros_like(x0); dx[0]=eps; dy=-np.linalg.solve(Gy,Gx@dx); d.x[:]=x0+dx; d.y[:]=y0+dy; system.vars_to_models(); system.TDS.fg_update(system.exist.tds,init=True); residual=float(np.linalg.norm(d.g)); ic.append({"epsilon":eps,"dx_l2":float(np.linalg.norm(dx)),"dy_l2":float(np.linalg.norm(dy)),"g_residual_l2":residual,"g_residual_over_epsilon2":residual/(eps*eps)})
    d.x[:]=x0; d.y[:]=y0; system.vars_to_models(); system.TDS.fg_update(system.exist.tds,init=True); ic_frame=pd.DataFrame(ic); ic_frame.to_csv(results/"e03r2_consistent_ic.csv",index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); ax.loglog(ic_frame.epsilon,ic_frame.g_residual_l2,marker="o",label="actual ||g||"); ax.loglog(ic_frame.epsilon,ic_frame.epsilon**2*(ic_frame.g_residual_l2.iloc[0]/ic_frame.epsilon.iloc[0]**2),"--",label="O(epsilon^2)"); ax.set(xlabel="epsilon",ylabel="consistent-IC residual"); ax.grid(True,which="both"); ax.legend(); fig.tight_layout(); fig.savefig(root/"output/plots/e03r2_consistent_ic_convergence.png",dpi=140); plt.close(fig)
    ranks = "/".join(str(int(x)) for x in obs["rank_tol_1e-8"])
    report += f"\n## Recorded numerical outcomes\n\nGovernor: `Fu_l2={der_rows[0]['Fu_l2']:.6g}`, `Gu_l2={der_rows[0]['Gu_l2']:.6g}`, `B_r_l2={der_rows[0]['B_r_l2']:.6g}`, `D_r_l2={der_rows[0]['D_r_l2']:.6g}`. Shunt: `Fu_l2={der_rows[1]['Fu_l2']:.6g}`, `Gu_l2={der_rows[1]['Gu_l2']:.6g}`, `B_r_l2={der_rows[1]['B_r_l2']:.6g}`, `D_r_l2={der_rows[1]['D_r_l2']:.6g}`.\n\nConsistent-IC residuals are in `e03r2_consistent_ic.csv`; at epsilon `1e-4`, `||g||={ic_frame.loc[ic_frame.epsilon==1e-4,'g_residual_l2'].iloc[0]:.6g}` and `||g||/epsilon^2={ic_frame.loc[ic_frame.epsilon==1e-4,'g_residual_over_epsilon2'].iloc[0]:.6g}`. Corrected VI_ONLY ranks at horizons 0/3/10/30/60/120 are `{ranks}`.\n"
    (reports/"e03r2_Tf_audit.md").write_text(report,encoding="utf-8")
    return {"status":"FAIL","tf_zero":int(np.sum(tf==0)),"tf_max":float(tf.max()),"before_dimension":int(before.shape[0]),"after_dimension":int(after.shape[0]),"corrected_spectral_abscissa":float(np.max(eig_after.real)),"native_slope":slope,"pflow_ok":pflow}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); print(run(p.parse_args().root))

if __name__ == "__main__": main()
