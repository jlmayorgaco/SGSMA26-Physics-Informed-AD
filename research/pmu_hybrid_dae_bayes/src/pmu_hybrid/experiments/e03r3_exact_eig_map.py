"""E03-R3: exact reproduction of the ANDES EIG coordinate map."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.sparse import csr_matrix, save_npz

from pmu_hybrid.cases.ieee39_andes import load_native_system, solve_static
from pmu_hybrid.constants import PMU_BUSES
from pmu_hybrid.experiments.e03_observability import _hidden_jacobian, _measurement_jacobian, _sparse
from pmu_hybrid.experiments.e03r_closure import _vi_from_ts

EPSILONS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5)


def _dense(M: Any) -> np.ndarray:
    return _sparse(M).toarray() if not isinstance(M, np.ndarray) else np.asarray(M, float)


def _selection(alive: list[int], total: int):
    from andes.shared import spmatrix
    S = spmatrix([1.0] * len(alive), list(range(len(alive))), alive, (len(alive), total))
    ST = spmatrix([1.0] * len(alive), alive, list(range(len(alive))), (total, len(alive)))
    return S, ST


def _constraint_reduction(A: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nc, n = C.shape; dep=[]; used=set()
    for i in range(nc):
        for p in np.argsort(-np.abs(C[i])):
            if int(p) not in used and abs(C[i,p]) > 0:
                dep.append(int(p)); used.add(int(p)); break
    dep=np.asarray(dep,int); free=np.array([i for i in range(n) if i not in set(dep)],int)
    R=np.zeros((n,len(free))); R[free,np.arange(len(free))]=1.0
    R[dep,:]=-np.linalg.solve(C[:,dep],C[:,free])
    return A[free,:]@R,R,free,dep


def reconstruct(system: Any) -> dict[str, Any]:
    """Reproduce EIG's fold/eliminate/reduce operations and the lift map."""
    d=system.dae; e=system.EIG; orig_names=[str(x) for x in d.x_name]; e.x_name=np.array(orig_names); e.x_tex_name=np.array(getattr(d,"x_tex_name",orig_names)); tf=np.asarray(d.Tf,float); zs=np.where(tf==0)[0]; nz=np.where(tf!=0)[0]
    e.zstate_idx=zs.copy(); fx,fy,gx,gy=(getattr(d,k) for k in ("fx","fy","gx","gy"))
    fx,fy,gx,gy,tfnz=e._fold_zstates(fx,fy,gx,gy,tf.copy())
    e.find_dead_algebs(gy,gx); dead=np.asarray(e.dead_algeb_idx,int); Cc=_dense(e._extract_state_constraints(gx)) if len(dead) else np.zeros((0,len(nz)))
    alive=sorted(set(range(gy.size[0]))-set(dead)); S,ST=_selection(alive,gy.size[0]); fx_a,fy_a,gx_a,gy_a=fx,fy*ST,S*gx,S*gy*ST
    # EIG regularizes dead columns after row elimination.
    gy_a=_dense(gy_a); gx_a=_dense(gx_a); fy_a=_dense(fy_a); fx_a=_dense(fx_a)
    for j in np.where(np.linalg.norm(gy_a,axis=0)<e.config.gy_tol)[0]: gy_a[j,j]=1.0
    Apre=(fx_a-fy_a@np.linalg.solve(gy_a,gx_a))/tfnz[:,None]
    Afinal,R,free,dep=_constraint_reduction(Apre,Cc)
    names_nz=[orig_names[i] for i in nz]; names_free=[names_nz[i] for i in free]
    # Full folded algebraic solve uses the alive equations; dead rows are exactly Cc dx=0.
    Lx=np.zeros((len(orig_names),len(free))); Ly=np.zeros((len(d.y_name),len(free)))
    xd_basis=R
    # A least-squares solve of the full folded system is the coordinate lift;
    # it retains the dead rows/columns instead of silently dropping their
    # first-order algebraic contribution.
    eta_full=np.linalg.lstsq(_dense(gy),-_dense(gx)@xd_basis,rcond=None)[0]
    Lx[nz,:]=xd_basis; Lx[zs,:]=eta_full[len(d.y_name):,:]; Ly[:,:]=eta_full[:len(d.y_name),:]
    return {"Apre":Apre,"A":Afinal,"R":R,"free":free,"dep":dep,"constraints":Cc,"dead":dead,"alive":alive,"zs":zs,"nz":nz,"names_nz":names_nz,"names_free":names_free,"Lx":Lx,"Ly":Ly,"folded":(fx,fy,gx,gy,tfnz),"eta_full_basis":eta_full}


def _run_no_input(system: Any, Lx: np.ndarray, Ly: np.ndarray, xi: np.ndarray, static: Any):
    system.TDS.config.tf=.20; system.TDS.config.tstep=.005; system.TDS.config.criteria=0; system.TDS.init(); x0=np.asarray(system.dae.x,float).copy(); y0=np.asarray(system.dae.y,float).copy(); system.TDS._x_t0[:]=x0+Lx@xi; system.TDS._y_t0[:]=y0+Ly@xi; system.dae.x[:]=system.TDS._x_t0; system.dae.y[:]=system.TDS._y_t0; system.vars_to_models(); ok=bool(system.TDS.run()); t=np.asarray(system.dae.ts.t,float); y=_vi_from_ts(t,np.asarray(system.dae.ts.y,float),static); return ok,t,np.asarray(system.dae.ts.x,float),y


def run(root: Path) -> dict[str, Any]:
    root=root.resolve(); out=root/"output"; results=out/"results"; reports=out/"reports"; results.mkdir(parents=True,exist_ok=True); reports.mkdir(parents=True,exist_ok=True)
    # Native reference is run directly, then a separate system is used for reconstruction.
    native=load_native_system(); native.setup(); native.PFlow.run(); native.TDS.config.tf=.20; native.TDS.config.tstep=.005; native.TDS.config.criteria=0; native.TDS.init(); native.EIG.run(); A_native=np.asarray(native.EIG.As,float); mu_native=np.asarray(native.EIG.mu,complex); pf_native=np.asarray(native.EIG.pfactors,float); native_names=[str(x) for x in native.EIG.x_name]
    save_npz(results/"e03r3_native_EIG_As.npz",csr_matrix(A_native)); np.save(results/"e03r3_native_EIG_mu.npy",mu_native); np.save(results/"e03r3_native_EIG_pfactors.npy",pf_native)
    system=load_native_system(); system.setup(); system.PFlow.run(); system.TDS.config.tf=.20; system.TDS.config.tstep=.005; system.TDS.config.criteria=0; system.TDS.init(); rec=reconstruct(system); A=rec["A"]
    vals=np.linalg.eigvals(A); match=np.array([np.min(np.abs(v-mu_native)) for v in vals]); parity={"native_dimension":A_native.shape[0],"custom_dimension":A.shape[0],"frobenius_abs":float(np.linalg.norm(A-A_native)),"frobenius_relative":float(np.linalg.norm(A-A_native)/max(np.linalg.norm(A_native),1e-15)),"max_entry_abs":float(np.max(np.abs(A-A_native))),"eigen_matching_max_abs":float(np.max(match)),"native_spectral_abscissa":float(np.max(mu_native.real)),"custom_spectral_abscissa":float(np.max(vals.real)),"parity":"PASS" if np.linalg.norm(A-A_native)<1e-8 else "FAIL"}
    pd.DataFrame([parity]).to_csv(results/"e03r3_native_eig_parity.csv",index=False)
    pd.DataFrame({"global_index":rec["zs"],"state":[str(system.dae.x_name[i]) for i in rec["zs"]],"Tf":0.0}).to_csv(results/"e03r3_zero_tf_states.csv",index=False)
    Cc=rec["constraints"]; crows=[]
    for i,row in enumerate(Cc):
        inds=np.argsort(np.abs(row))[::-1][:8]; crows.append({"constraint_index":i,"source":"dead algebraic row "+str(rec["dead"][i]),"pivot_state":rec["names_nz"][rec["dep"][i]],"dominant_coefficients":json.dumps({rec["names_nz"][int(j)]:float(row[j]) for j in inds if abs(row[j])>0})})
    pd.DataFrame(crows).to_csv(results/"e03r3_state_constraints.csv",index=False); save_npz(results/"e03r3_R.npz",csr_matrix(rec["R"])); save_npz(results/"e03r3_LIFT_x.npz",csr_matrix(rec["Lx"])); save_npz(results/"e03r3_LIFT_y.npz",csr_matrix(rec["Ly"]))
    lift_constraints=float(np.linalg.norm(Cc@rec["R"])) if len(Cc) else 0.0; folded_res=float(np.linalg.norm(_dense(rec["folded"][2])@rec["R"]+_dense(rec["folded"][3])@rec["eta_full_basis"]))
    pd.DataFrame([{"C_R_l2":lift_constraints,"folded_algebraic_residual_l2":folded_res,"state_dimension":A.shape[0],"original_x":rec["Lx"].shape[0],"original_y":rec["Ly"].shape[0]}]).to_csv(results/"e03r3_lift_residuals.csv",index=False)
    static=solve_static(); Hx,Hy,_,_,_= _measurement_jacobian(system,static); C_red=np.hstack([Hx,Hy])@np.vstack([rec["Lx"],rec["Ly"]]); fd=[]; rng=np.random.default_rng(3)
    for eps in EPSILONS:
        v=rng.normal(size=A.shape[0]); v/=np.linalg.norm(v); z0=np.asarray(system.dae.y,float).copy(); xp=rec["Lx"]@(eps*v); yp=rec["Ly"]@(eps*v); hp=_vi_from_ts(np.array([0.0]),np.array([z0+yp]),static)[0]; hm=_vi_from_ts(np.array([0.0]),np.array([z0-yp]),static)[0]; fd.append({"epsilon":eps,"measurement_fd_error":float(np.linalg.norm((hp-hm)/(2*eps)-C_red@v))})
    pd.DataFrame(fd).to_csv(results/"e03r3_measurement_lift_fd.csv",index=False)
    # Consistent IC and no-input TDS campaign.
    base=load_native_system(); base.setup(); base.PFlow.run(); base.TDS.config.tf=.20; base.TDS.config.tstep=.005; base.TDS.config.criteria=0; base.TDS.init(); base.TDS.run(); tb=np.asarray(base.dae.ts.t,float); yb=_vi_from_ts(tb,np.asarray(base.dae.ts.y,float),static); rows=[]
    for eps in EPSILONS:
        v=np.ones(A.shape[0]); v/=np.linalg.norm(v); xi=eps*v; case=load_native_system(); case.setup(); case.PFlow.run(); ok,t,xx,yy=_run_no_input(case,rec["Lx"],rec["Ly"],xi,static); actual=yy-np.array([yb[np.argmin(abs(tb-ti))] for ti in t]); pred=np.array([C_red@(expm(A*ti)@xi) for ti in t]); err=float(np.linalg.norm(actual-pred)); rows.append({"epsilon":eps,"tds_ok":ok,"absolute_error":err,"error_over_epsilon":err/eps,"error_over_epsilon2":err/eps**2})
    noinput=pd.DataFrame(rows); noinput["slope_p"]=np.polyfit(np.log(noinput.epsilon),np.log(noinput.absolute_error),1)[0]; noinput.to_csv(results/"e03r3_no_input_tds_convergence.csv",index=False)
    # Positive native mode validation with modal coordinate from the left eigenvector.
    j=int(np.argmax(mu_native.real)); lam=mu_native[j]; rv=np.real(np.asarray(native.EIG.N)[:,j]); rv/=np.linalg.norm(rv); lv=np.linalg.eig(A.T)[1][:,np.argmin(abs(np.linalg.eigvals(A.T)-lam))]; modal=[]; eps=1e-5; case=load_native_system(); case.setup(); case.PFlow.run(); ok,t,xx,yy=_run_no_input(case,rec["Lx"],rec["Ly"],eps*rv,static); free=rec["free"]; xi_tr=np.asarray(xx)[:,rec["nz"]][:,free]; a=np.abs(xi_tr@lv); a=np.maximum(a/a[0],1e-300); early=t<=0.15; modal_slope=np.polyfit(t[early],np.log(a[early]),1)[0] if early.sum()>2 else np.nan; pd.DataFrame([{"native_real":lam.real,"native_imag":lam.imag,"tds_ok":ok,"measured_modal_slope":modal_slope,"predicted_real_growth":lam.real,"classification":"NATIVE_MODE_CONFIRMED" if ok and abs(modal_slope-lam.real)<0.5 else "NATIVE_MODE_NOT_REPRODUCED"}]).to_csv(results/"e03r3_positive_mode.csv",index=False)
    report=f"# E03-R3 exact ANDES EIG coordinate map\n\nNative EIG parity: `{parity}`. Zero-Tf states: `{len(rec['zs'])}`; constraints: `{len(rec['dep'])}`; reduced dimension: `{A.shape[0]}`.\n\n`||C R||={lift_constraints:.6g}` and folded algebraic residual `={folded_res:.6g}`. The full lift matrices are stored as `e03r3_LIFT_x.npz` and `e03r3_LIFT_y.npz`; measurement uses `[Hx Hy] LIFT`, not column selection.\n\nNative no-input TDS convergence slope: `{float(noinput.slope_p.iloc[0]):.6g}`. Positive native mode is `{lam.real:.8g}{lam.imag:+.8g}j`; mode validation is recorded in `e03r3_positive_mode.csv`.\n\n**E03-R3 = {'PASS' if parity['parity']=='PASS' and lift_constraints<1e-8 and folded_res<1e-8 and float(noinput.slope_p.iloc[0])>1.5 else 'FAIL'}**. E04 not started.\n"
    (reports/"e03r3_exact_eig_map.md").write_text(report,encoding="utf-8")
    return {"status":"PASS" if parity["parity"]=="PASS" and lift_constraints<1e-8 and folded_res<1e-8 and float(noinput.slope_p.iloc[0])>1.5 else "FAIL",**parity,"C_R_l2":lift_constraints,"folded_residual":folded_res,"no_input_slope":float(noinput.slope_p.iloc[0])}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); print(run(p.parse_args().root))
if __name__=="__main__": main()
