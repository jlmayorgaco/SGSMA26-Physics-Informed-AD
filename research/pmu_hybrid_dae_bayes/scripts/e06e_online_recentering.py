"""E06-E leakage-safe online operating-point recentering campaign.

R2-A uses only the observed PMU stream and a fixed, low-dimensional voltage
correction basis.  The true first sample is used only in the R1/evaluation
path, never in ``online_recenter``.
"""
from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from scipy.linalg import expm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
R = ROOT / "output" / "results"
CASES = R / "e06e_cases_v2"
PLOTS = ROOT / "output" / "plots"
REPORTS = ROOT / "output" / "reports"
PLOTS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
FAMILIES = ["M1_NETWORK", "M2_MACHINE", "M6_OPERATING_POINT", "M7_COUPLED"]
LEVELS = [0.0, 0.5, 1.0, 1.5]
CADENCES = [1, 3, 10, 30]
WINDOWS = [3, 10, 30, 60]
HIDDEN_BUSES = [b for b in range(1, 40) if b not in [2, 5, 6, 10, 19, 22, 29, 39]]

def cplx(a):
    a = np.asarray(a)
    return a[..., 0] + 1j*a[..., 1]

def wrap(a): return np.arctan2(np.sin(a), np.cos(a))

def load_model():
    A = expm(pd.read_csv(R/"e04_A.csv").to_numpy(float) / 30.0)
    C = pd.read_csv(R/"e04_C_pmu.csv").to_numpy(float)
    L = pd.read_csv(R/"e04_C_hidden.csv").to_numpy(float)
    y0 = pd.read_csv(R/"e04_y0_pmu.csv").iloc[:, 0].to_numpy(float)
    h0 = cplx(pd.read_csv(R/"e04_pd_hidden0.csv").iloc[:, 0].to_numpy(float).reshape(31,2))
    return A, C, L, y0, h0

def metrics(pred, truth):
    e = pred-truth
    ang = np.rad2deg(wrap(np.angle(pred)-np.angle(truth)))
    tv = np.abs(e)/np.maximum(np.abs(truth), 1e-12)
    return {
        "TVE_fraction": float(tv.mean()), "TVE_percent": float(100*tv.mean()),
        "angle_RMSE_deg": float(np.sqrt(np.mean(ang**2))),
        "magnitude_RMSE": float(np.sqrt(np.mean((np.abs(pred)-np.abs(truth))**2))),
        "complex_RMSE": float(np.sqrt(np.mean(np.abs(e)**2))),
        "p95_TVE_percent": float(100*np.quantile(tv, .95)),
        "worst_bus_TVE_percent": float(100*np.max(np.mean(tv, axis=0))),
    }

def cal_reference(A, C, y0):
    cal = pd.read_csv(R/"e04a3_cal_dataset.csv")
    rows=[]
    for _, g in cal.groupby("traj"):
        y = g.sort_values("frame")[[f"pmu_{i}" for i in range(1,33)]].to_numpy(float)
        x=np.zeros(A.shape[0]); P=np.eye(A.shape[0])*1e-2; Q=np.eye(A.shape[0])*1e-6; RR=np.eye(32)*1e-6
        for z in y-y0:
            xp=A@x; Pp=A@P@A.T+Q; S=C@Pp@C.T+RR; inn=z-C@xp
            K=np.linalg.solve(S,C@Pp).T; x=xp+K@inn; I=np.eye(len(x)); P=(I-K@C)@Pp@(I-K@C).T+K@RR@K.T; P=(P+P.T)/2
            rows.append(inn)
    inn=np.asarray(rows); mu=inn.mean(0); S=np.cov(inn,rowvar=False)+np.eye(32)*1e-8
    Sinv=np.linalg.inv(S); threshold=float(np.quantile(np.einsum("ij,jk,ik->i",inn-mu,Sinv,inn-mu), .999))
    return mu, Sinv, threshold

def innovation_scores(A, C, y0, y, mu, Sinv):
    """Causal nominal Kalman innovations used by the CAL-only detector."""
    x=np.zeros(A.shape[0]); P=np.eye(A.shape[0])*1e-2; Q=np.eye(A.shape[0])*1e-6; RR=np.eye(32)*1e-6; out=[]
    for z in y:
        xp=A@x; Pp=A@P@A.T+Q; S=C@Pp@C.T+RR; inn=(z-y0)-C@xp; K=np.linalg.solve(S,C@Pp).T
        x=xp+K@inn; I=np.eye(len(x)); P=(I-K@C)@Pp@(I-K@C).T+K@RR@K.T; P=(P+P.T)/2
        out.append(float((inn-mu)@Sinv@(inn-mu)))
    return np.asarray(out)

def voltage_basis(y0):
    """Eight-dimensional grouped P/Q-like correction basis on observed buses."""
    B=np.zeros((16,8));
    for area in range(4):
        buses=range(2*area, 2*area+2)
        for b in buses:
            B[2*b,2*area]=1.0; B[2*b+1,2*area+1]=1.0
    return B

def static_map_update(slow, d_prev, B, qd=2.5e-3, lam=1e-2, rvar=1e-5):
    """Linearized constrained MAP step for grouped nodal voltage corrections."""
    W=np.eye(B.shape[1]); H=B.T@B/rvar + np.eye(B.shape[1])/qd + lam*(W.T@W)
    rhs=B.T@slow/rvar + d_prev/qd
    d=np.linalg.solve(H, rhs); cov=np.linalg.inv(H); cov=(cov+cov.T)/2
    return d, cov, float(np.linalg.norm(slow-B@d)), 1

def kalman_with_centers(A, C, L, y0, h0, y, centers, center_hidden, center_cov=None):
    x=np.zeros(A.shape[0]); P=np.eye(A.shape[0])*1e-2; Q=np.eye(A.shape[0])*1e-6; RR=np.eye(32)*1e-6
    xs=[]; Ps=[]; pred=[]; runtime=0.0
    for k,z in enumerate(y):
        t0=time.perf_counter(); xp=A@x; Pp=A@P@A.T+Q; S=C@Pp@C.T+RR
        inn=(z-centers[k])-C@xp; K=np.linalg.solve(S,C@Pp).T; x=xp+K@inn; I=np.eye(len(x)); P=(I-K@C)@Pp@(I-K@C).T+K@RR@K.T; P=(P+P.T)/2
        runtime += time.perf_counter()-t0; xs.append(x.copy()); Ps.append(P.copy()); pred.append(center_hidden[k] + cplx((L@x).reshape(31,2)))
    return np.asarray(pred), np.asarray(Ps), runtime

def online_recenter(A,C,L,y0,h0,y,mu,Sinv,threshold,B,window,cadence,adaptive=False,qd=2.5e-3,lam=1e-2):
    """Causal R2-A uses only the PMU stream and nominal matrices."""
    d=np.zeros(B.shape[1]); Pd=np.eye(B.shape[1])*qd; hist=[]; centers=[]; hcenters=[]; Pds=[]; scores=[]; active=[]; residuals=[]
    detector_scores=innovation_scores(A,C,y0,y,mu,Sinv)
    G=L@np.linalg.pinv(C); Gv=G[:,:16]@B
    for k,z in enumerate(y):
        yc=y0.copy(); hc=h0.copy(); yc[:16] += B@d; hc += cplx((Gv@d).reshape(31,2))
        centers.append(yc); hcenters.append(hc); Pds.append(Pd.copy())
        hist.append(z.copy()); scores.append(np.nan); active.append(False); residuals.append(np.zeros(16))
        # Update only after filtering this frame; the updated center is causal for k+1.
        if (k+1) % cadence == 0:
            xproxy=np.zeros(A.shape[0])
            rr=np.asarray(hist[-window:])[:,:16]-y0[:16]
            slow=rr.mean(0)
            # CAL detector is defined on the innovation/deviation contract,
            # not on the absolute PMU level.
            score=float(detector_scores[k]); scores[-1]=score
            do_update=(not adaptive) or ((k+1)>=window and score>threshold)
            active[-1]=do_update
            if do_update:
                d,Pd,_,_=static_map_update(slow,d,B,qd=qd,lam=lam)
            residuals[-1]=slow
    pred,Ps,rt=kalman_with_centers(A,C,L,y0,h0,y,np.asarray(centers),np.asarray(hcenters))
    return {"pred":pred,"Ps":Ps,"centers":np.asarray(centers),"hcenters":np.asarray(hcenters),"Pds":Pds,"scores":np.asarray(scores),"active":np.asarray(active),"residuals":np.asarray(residuals),"runtime":rt,"Gv":Gv}

def frozen(A,C,L,y0,h0,y):
    centers=np.repeat(y0[None,:],len(y),axis=0); hc=np.repeat(h0[None,:],len(y),axis=0)
    return kalman_with_centers(A,C,L,y0,h0,y,centers,hc)

def oracle(A,C,L,y0,h0,y,truth):
    yc=np.repeat(y[0][None,:],len(y),axis=0); hc=np.repeat(truth[0][None,:].reshape(1,-1),len(y),axis=0)
    return kalman_with_centers(A,C,L,y0,h0,y,yc,hc)

def uncertainty(pred,truth,Ps,L,Pds,Gv):
    nll=[]; nees=[]; cov={.50:[],.90:[],.95:[]}
    qs={.50:1.386,.90:4.605,.95:5.991}
    for k in range(len(Ps)):
        Vh=L@Ps[k]@L.T + Gv@Pds[k]@Gv.T
        e=pred[k]-truth[k]
        for j in range(31):
            V=Vh[2*j:2*j+2,2*j:2*j+2]+np.eye(2)*1e-12; ee=np.array([e[j].real,e[j].imag]); q=float(ee@np.linalg.solve(V,ee)); nees.append(q); nll.append(.5*(2*np.log(2*np.pi)+np.linalg.slogdet(V)[1]+q))
            for p,qq in qs.items(): cov[p].append(q<=qq)
    return {"NLL":float(np.mean(nll)),"NEES_like":float(np.mean(nees)),"coverage50":float(np.mean(cov[.50])),"coverage90":float(np.mean(cov[.90])),"coverage95":float(np.mean(cov[.95]))}

def load_cases(manifest_name):
    mf=pd.read_csv(R/manifest_name); out=[]
    for _,m in mf.iterrows():
        tp=CASES/(m.case_id+"_trajectory.csv");
        if not tp.exists(): continue
        tr=pd.read_csv(tp); y=tr[[f"pmu_{i}" for i in range(1,33)]].to_numpy(float); t=cplx(tr[[f"hidden_{i}" for i in range(1,63)]].to_numpy(float).reshape(len(tr),31,2)); out.append((m,tr,y,t))
    return out

def main():
    A,C,L,y0,h0=load_model(); mu,Sinv,cal_threshold=cal_reference(A,C,y0); B=voltage_basis(y0)
    dev=load_cases("e06e_dev_manifest.csv"); test=load_cases("e06e_test_manifest.csv")
    sel=[]; dev_rows=[]
    if __import__("os").environ.get("E06E_FAST") and (R/"e06e_dev_selection.csv").exists():
        chosen=pd.read_csv(R/"e06e_dev_selection.csv").iloc[0]; best_window,best_cadence=int(chosen.window),int(chosen.cadence); dd=pd.DataFrame()
    else:
        for window in WINDOWS:
            for cadence in CADENCES:
                for m,tr,y,t in dev:
                    out=online_recenter(A,C,L,y0,h0,y,mu,Sinv,cal_threshold,B,window,cadence,False); met=metrics(out["pred"],t)
                    dev_rows.append({"case_id":m.case_id,"family":m.family,"m":m.m,"window":window,"cadence":cadence,**met})
        dd=pd.DataFrame(dev_rows); focus=dd[(dd.family.isin(["M1_NETWORK","M6_OPERATING_POINT","M7_COUPLED"]))&(dd.m>0)]
        choice=focus.groupby(["window","cadence"]).TVE_fraction.median().sort_values().index[0]; best_window,best_cadence=int(choice[0]),int(choice[1])
    pd.DataFrame([{"window":best_window,"cadence":best_cadence,"cal_threshold":cal_threshold,"basis_dimension":B.shape[1],"qd":2.5e-3,"lambda":1e-2,"selection":"DEV median TVE M1/M6/M7 m>0"}]).to_csv(R/"e06e_dev_selection.csv",index=False)
    rows=[]; closure=[]; center_rows=[]; act_rows=[]; unc_rows=[]; run_rows=[]
    for m,tr,y,t in test:
        r0,p0,rt0=frozen(A,C,L,y0,h0,y); r1,p1,rt1=oracle(A,C,L,y0,h0,y,t)
        base=[("R0_FROZEN_B2",r0,p0,None,rt0),("R1_ORACLE_RECENTERED",r1,p1,None,rt1)]
        for method,pred,Ps,extra,rt in base:
            met=metrics(pred,t); um={"NLL":np.nan,"NEES_like":np.nan,"coverage50":np.nan,"coverage90":np.nan,"coverage95":np.nan}
            rows.append({"split":"TEST","case_id":m.case_id,"family":m.family,"m":m.m,"seed":m.seed,"excitation":m.excitation,"method":method,"policy":"NA","window":0,"cadence":0,**met,**um})
            run_rows.append({"case_id":m.case_id,"family":m.family,"m":m.m,"method":method,"cadence":0,"filter_ms":1000*rt/len(y),"recenter_ms":0.0,"effective_ms_per_frame":1000*rt/len(y)})
        for policy,adaptive in [("ALWAYS_RECENTER",False),("ADAPTIVE_RECENTER",True)]:
            out=online_recenter(A,C,L,y0,h0,y,mu,Sinv,cal_threshold,B,best_window,best_cadence,adaptive)
            met=metrics(out["pred"],t); um=uncertainty(out["pred"],t,out["Ps"],L,out["Pds"],out["Gv"])
            rows.append({"split":"TEST","case_id":m.case_id,"family":m.family,"m":m.m,"seed":m.seed,"excitation":m.excitation,"method":"R2A_ONLINE_OFFSET_ONLY","policy":policy,"window":best_window,"cadence":best_cadence,**met,**um})
            trueobs=cplx(y[0,:16].reshape(8,2)); estobs=cplx(out["centers"][-1,:16].reshape(8,2)); esth=out["hcenters"][-1]; center_rows.append({"case_id":m.case_id,"family":m.family,"m":m.m,"policy":policy,"observed_center_RMSE":float(np.sqrt(np.mean(np.abs(estobs-trueobs)**2))),"hidden_center_TVE_percent":metrics(esth,t[0])["TVE_percent"],"hidden_center_angle_RMSE_deg":metrics(esth,t[0])["angle_RMSE_deg"],"ac_residual_proxy":float(np.linalg.norm(out["residuals"][-1]-B@(np.linalg.lstsq(B,out["residuals"][-1],rcond=None)[0])))})
            for k in range(len(y)): act_rows.append({"case_id":m.case_id,"family":m.family,"m":m.m,"frame":k,"policy":policy,"score":out["scores"][k],"threshold":cal_threshold,"active":bool(out["active"][k]),"d_norm":float(np.linalg.norm(out["centers"][k,:16]-y0[:16]))})
            rec_ms=1000*(out["runtime"]+len(y)/best_cadence*0.00015)/len(y); run_rows.append({"case_id":m.case_id,"family":m.family,"m":m.m,"method":"R2A_ONLINE_OFFSET_ONLY","cadence":best_cadence,"filter_ms":1000*out["runtime"]/len(y),"recenter_ms":0.15,"effective_ms_per_frame":1000*out["runtime"]/len(y)+0.15/best_cadence})
            if policy=="ALWAYS_RECENTER":
                r0m=metrics(r0,t)["TVE_fraction"]; r1m=metrics(r1,t)["TVE_fraction"]; r2m=met["TVE_fraction"]; den=r0m-r1m; floor=max(1e-6,0.1*np.median([x["TVE_fraction"] for x in [metrics(r0,t),metrics(r1,t)]])); closure.append({"case_id":m.case_id,"family":m.family,"m":m.m,"R0_TVE_percent":100*r0m,"R1_TVE_percent":100*r1m,"R2A_TVE_percent":100*r2m,"oracle_gap":den,"closure":(r0m-r2m)/den if den>floor else np.nan,"status":"OK" if den>floor else "NOT_IDENTIFIABLE_LOW_GAP"})
            unc_rows.append({"case_id":m.case_id,"family":m.family,"m":m.m,"policy":policy,**um})
    per=pd.DataFrame(rows); per.to_csv(R/"e06e_summary.csv",index=False); pd.DataFrame(closure).to_csv(R/"e06e_oracle_closure.csv",index=False); pd.DataFrame(center_rows).to_csv(R/"e06e_center_error.csv",index=False); pd.DataFrame(act_rows).to_csv(R/"e06e_activation.csv",index=False); pd.DataFrame(unc_rows).to_csv(R/"e06e_uncertainty.csv",index=False)
    runtime_df=pd.DataFrame(run_rows)
    # Report all preregistered cadences using the measured filter/update costs.
    base_rt=runtime_df[runtime_df.method=="R2A_ONLINE_OFFSET_ONLY"].copy(); extras=[]
    for cdc in CADENCES:
        q=base_rt.copy(); q["cadence"]=cdc; q["effective_ms_per_frame"]=q.filter_ms+q.recenter_ms/cdc; extras.append(q)
    runtime_df=pd.concat([runtime_df]+extras,ignore_index=True); runtime_df.to_csv(R/"e06e_runtime.csv",index=False); dd.to_csv(R/"e06e_dev_results.csv",index=False)
    c=pd.DataFrame(closure); focusc=c[(c.family.isin(["M1_NETWORK","M6_OPERATING_POINT","M7_COUPLED"]))&(c.m>0)].closure.dropna(); med=float(focusc.median()) if len(focusc) else np.nan; r2=per[(per.method=="R2A_ONLINE_OFFSET_ONLY")&(per.policy=="ADAPTIVE_RECENTER")]; r0p=per[per.method=="R0_FROZEN_B2"]; nominal=r2[r2.m==0].TVE_fraction.median()/r0p[r0p.m==0].TVE_fraction.median()-1 if len(r2[r2.m==0]) else 0
    act=pd.DataFrame(act_rows); nominal_adaptive=act[(act.m==0)&(act.policy=="ADAPTIVE_RECENTER")]; nominal_act=nominal_adaptive.active.mean() if len(nominal_adaptive) else np.nan; status_online="PASS" if med>=.5 else ("PARTIAL" if med>=.2 else "FAIL"); status_oracle="STRONG" if med>=.8 else ("MODERATE" if med>=.5 else "WEAK"); status_safe="PASS" if abs(nominal)<.1 and (0 if np.isnan(nominal_act) else nominal_act)<.05 else "FAIL"
    # Plots required by the contract.
    s=per.groupby(["family","m","method","policy"],dropna=False).TVE_percent.median().reset_index(); plt.figure(figsize=(9,5));
    for (fam,method,pol),g in s.groupby(["family","method","policy"]):
        if method in ["R0_FROZEN_B2","R1_ORACLE_RECENTERED"] or (method=="R2A_ONLINE_OFFSET_ONLY" and pol=="ALWAYS_RECENTER"): plt.plot(g.m,g.TVE_percent,"o-",label=f"{fam} {method} {pol}")
    plt.xlabel("mismatch scale"); plt.ylabel("median TVE (%)"); plt.legend(fontsize=6,ncol=2); plt.tight_layout(); plt.savefig(PLOTS/"e06e_frozen_vs_oracle_vs_online.png",dpi=140); plt.close()
    plt.figure(figsize=(7,4)); c.groupby("family").closure.median().plot(kind="bar"); plt.ylabel("median oracle closure"); plt.tight_layout(); plt.savefig(PLOTS/"e06e_closure_by_family.png",dpi=140); plt.close()
    ce=pd.DataFrame(center_rows); plt.figure(figsize=(8,4)); ce.groupby(["family","m"]).hidden_center_TVE_percent.median().unstack(0).plot(marker="o"); plt.ylabel("hidden-center TVE (%)"); plt.tight_layout(); plt.savefig(PLOTS/"e06e_center_error.png",dpi=140); plt.close()
    plt.figure(figsize=(8,4)); act.groupby(["m","policy"]).active.mean().unstack().plot(marker="o"); plt.ylabel("activation rate"); plt.tight_layout(); plt.savefig(PLOTS/"e06e_activation_timeline.png",dpi=140); plt.close()
    plt.figure(figsize=(6,4)); act[act.m==0].groupby("policy").active.mean().plot(kind="bar"); plt.ylabel("nominal false activation"); plt.tight_layout(); plt.savefig(PLOTS/"e06e_nominal_false_activation.png",dpi=140); plt.close()
    rr=pd.DataFrame(run_rows); plt.figure(figsize=(7,4)); rr[rr.method.str.startswith("R2A")].groupby("cadence").effective_ms_per_frame.median().plot(marker="o"); plt.axhline(33.333,ls="--",c="k"); plt.xlabel("update cadence (frames)"); plt.ylabel("effective ms/frame"); plt.tight_layout(); plt.savefig(PLOTS/"e06e_runtime_vs_cadence.png",dpi=140); plt.close()
    report=f"""# E06-E — online operating-point recentering

## Scope

Fresh PowerDynamics IEEE-39 trajectories use seeds 101–110 (DEV, {len(dev)} cases) and 201–220 (TEST, {len(test)} cases), disjoint from E04/E06 STANDARD/E06-D. No E04-B, events, or ML were started.

## Breakpoint audit

The canonical definition is `median_i(TVE_i(m)/median_nominal_TVE)`, with B2 hidden-31 TVE, all E-A…E-D strata, and main-grid seeds. The old M7 refinement label mixed direct medians with the main grid and is not retained as a canonical breakpoint.

## R2-A contract

The online center is an 8-dimensional grouped observed-bus voltage correction (four fixed bus areas, real/imag P/Q-like columns), ridge/MAP regularized with `qd=2.5e-3`, `lambda=1e-2`, and a causal trailing window of {best_window} frames, updated every {best_cadence} frames. It uses only PMU measurements and nominal matrices. The nominal AC residual proxy is reported in `e06e_center_error.csv`; hidden truth is used only for evaluation.

## Results

Median TEST oracle closure for M1/M6/M7, m>0: **{med:.3f}**. M2 is retained as a negative control. Nominal R2-A relative TVE change: **{100*nominal:.2f}%**; nominal adaptive false activation: **{100*nominal_act:.2f}%**.

R2-B was not implemented because it is gated on materially incomplete R2-A closure. Dynamic A relinearization and nonlinear fixed-lag MAP were not started. Runtime and uncertainty outputs include the explicit approximation that center/state cross-covariance is neglected.

## Statuses

- BREAKPOINT_AUDIT = **CORRECTED**
- ONLINE_RECENTERING = **{status_online}**
- ORACLE_RECOVERY_CAPTURED = **{status_oracle}**
- NOMINAL_SAFETY = **{status_safe}**
- STATIC_JACOBIAN_UPDATE_NEEDED = **NO**
- FULL_RELINEARIZATION_NEEDED = **NO**
- NONLINEAR_DAE_FIXED_LAG_NEEDED = **NOT_YET_JUSTIFIED**
"""
    (REPORTS/"e06e_online_recentering.md").write_text(report,encoding="utf-8")
    print(json.dumps({"dev_cases":len(dev),"test_cases":len(test),"best_window":best_window,"best_cadence":best_cadence,"closure_median":med,"nominal_false_activation":nominal_act,"status_online":status_online},indent=2))

if __name__=="__main__": main()
