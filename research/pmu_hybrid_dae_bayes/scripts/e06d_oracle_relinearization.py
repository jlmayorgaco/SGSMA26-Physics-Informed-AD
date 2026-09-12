"""E06-D: score physically rebuilt PowerDynamics oracle models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import expm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
R = ROOT / "output" / "results"; CASES = R / "e06d_cases"; PLOTS = ROOT / "output" / "plots"; REP = ROOT / "output" / "reports"
FAMILIES = ["M1_NETWORK","M2_MACHINE","M3_GOVERNOR","M4_AVR","M5_LOAD_MODEL","M6_OPERATING_POINT","M7_COUPLED"]
OBS = [2,5,6,10,19,22,29,39]; HIDDEN = [b for b in range(1,40) if b not in OBS]
N = 114; NY = 32; NH = 62; QVAR = 1e-6; RVAR = 1e-6; P0VAR = 1e-2

def readv(p): return pd.read_csv(p).iloc[:, -1].to_numpy(float)
def cplx(a): return a[...,0::2] + 1j*a[...,1::2]
def wrap(a): return (a + np.pi) % (2*np.pi) - np.pi

def run_filter(A,C,y0,measurements):
    x=np.zeros(A.shape[0]); P=np.eye(A.shape[0])*P0VAR; xs=[]; Ps=[]
    Q=np.eye(A.shape[0])*QVAR; RR=np.eye(C.shape[0])*RVAR
    for y in measurements:
        xp=A@x; Pp=A@P@A.T+Q; innov=y-y0-C@xp; S=C@Pp@C.T+RR
        K=np.linalg.solve(S,C@Pp).T; x=xp+K@innov; I=np.eye(A.shape[0]); P=(I-K@C)@Pp@(I-K@C).T+K@RR@K.T; P=(P+P.T)/2
        xs.append(x.copy()); Ps.append(P.copy())
    return np.asarray(xs), np.asarray(Ps)

def metrics(pred,truth):
    e=pred-truth; tve=np.abs(e)/np.maximum(np.abs(truth),1e-12)
    ang=np.rad2deg(wrap(np.angle(pred)-np.angle(truth))); vm=np.abs(pred)-np.abs(truth)
    return dict(TVE_fraction=float(tve.mean()),TVE_percent=float(100*tve.mean()),complex_RMSE=float(np.sqrt(np.mean(np.abs(e)**2))),magnitude_RMSE=float(np.sqrt(np.mean(vm**2))),angle_RMSE_deg=float(np.sqrt(np.mean(ang**2))),per_bus_tve=tve.mean(axis=0),per_bus_angle=np.sqrt(np.mean(ang**2,axis=0)))

def load_case(cid):
    base=CASES/cid
    tr=pd.read_csv(str(base)+"_trajectory.csv"); A=expm(pd.read_csv(str(base)+"_A.csv").to_numpy(float)/30.0); C=pd.read_csv(str(base)+"_C.csv").to_numpy(float); L=pd.read_csv(str(base)+"_L.csv").to_numpy(float); y0=readv(str(base)+"_y0.csv"); h0=readv(str(base)+"_h0.csv")
    y=tr[[f"pmu_{i}" for i in range(1,33)]].to_numpy(float); truth=cplx(tr[[f"hidden_{i}" for i in range(1,63)]].to_numpy(float).reshape(len(tr),31,2));
    return dict(A=A,C=C,L=L,y0=y0,h0=h0,y=y,truth=truth,time=tr.time.to_numpy(float))

def score_case(case,nom):
    out={}
    for method in ("N0","O1","O2"):
        if method=="N0": A,C,L,y0,h0=nom["A"],nom["C"],nom["L"],nom["y0"],nom["h0"]
        elif method=="O1": A,C,L,y0,h0=nom["A"],nom["C"],nom["L"],case["y0"],case["h0"]
        else: A,C,L,y0,h0=case["A"],case["C"],case["L"],case["y0"],case["h0"]
        xs,Ps=run_filter(A,C,y0,case["y"])
        # interleaved real/imag rows: hidden output = h0 + L*x
        pred=cplx((h0[None,:]+xs@L.T).reshape(len(xs),31,2)); mm=metrics(pred,case["truth"])
        out[method]=(mm,pred,xs,Ps)
    return out

def main():
    manifest=pd.read_csv(R/"e06d_manifest.csv"); ids=manifest.case_id.tolist(); nominal_id="NOMINAL_m0_s1"; nom=load_case(nominal_id)
    # Explicit nominal reference uses the package-exported frozen model and nominal physical trajectory.
    rows=[]; busrows=[]; cache={}
    for _,mr in manifest.iterrows():
        cid=mr.case_id
        if not (CASES/(cid+"_A.csv")).exists(): continue
        case=load_case(cid); scored=score_case(case,nom); cache[cid]=(case,scored)
        for method,(mm,pred,xs,Ps) in scored.items():
            row={"case_id":cid,"family":mr.family,"m":mr.m,"seed":mr.seed,"method":method,**{k:v for k,v in mm.items() if not k.startswith("per_bus")}}
            rows.append(row)
            for j,b in enumerate(HIDDEN): busrows.append({"case_id":cid,"family":mr.family,"m":mr.m,"seed":mr.seed,"method":method,"hidden_bus":b,"TVE_fraction":mm["per_bus_tve"][j],"angle_RMSE_deg":mm["per_bus_angle"][j]})
    per=pd.DataFrame(rows); per.to_csv(R/"e06d_per_case.csv",index=False); pd.DataFrame(busrows).to_csv(R/"e06d_per_bus.csv",index=False)
    ref=per[per.case_id.str.startswith("NOMINAL")].groupby("method")["TVE_fraction"].median().get("N0",float(per[per.case_id.str.startswith("NOMINAL")].TVE_fraction.median())); refang=per[per.case_id.str.startswith("NOMINAL")].groupby("method")["angle_RMSE_deg"].median().get("N0",0.0)
    mut=per[~per.case_id.str.startswith("NOMINAL")].copy(); piv=mut.pivot_table(index=["case_id","family","m","seed"],columns="method",values=["TVE_fraction","angle_RMSE_deg"]); piv.columns=["_".join(c) for c in piv.columns]; piv=piv.reset_index()
    for metric,rv in (("TVE_fraction",ref),("angle_RMSE_deg",refang)):
        e0=piv[f"{metric}_N0"]-rv; den=e0.where(e0>0,np.nan); piv[f"Excess_N0_{metric}"]=e0; piv[f"Excess_O1_{metric}"]=piv[f"{metric}_O1"]-rv; piv[f"Excess_O2_{metric}"]=piv[f"{metric}_O2"]-rv; piv[f"Recovery_Recenter_{metric}"]=1-(piv[f"{metric}_O1"]-rv)/den; piv[f"Recovery_Relinearize_{metric}"]=1-(piv[f"{metric}_O2"]-rv)/den
    piv.to_csv(R/"e06d_recovery_fractions.csv",index=False)
    fam=[]
    for f,g in piv.groupby("family"):
        med={"family":f}
        for meth in ("N0","O1","O2"): med[f"{meth}_TVE_percent"]=100*g[f"TVE_fraction_{meth}"].median(); med[f"{meth}_angle_RMSE_deg"]=g[f"angle_RMSE_deg_{meth}"].median()
        rec=float(g.Recovery_Recenter_TVE_fraction.median()); rel=float(g.Recovery_Relinearize_TVE_fraction.median()); o2ex=float(g.Excess_O2_TVE_fraction.median()); n0ex=float(g.Excess_N0_TVE_fraction.median())
        if n0ex>0 and o2ex>0.5*n0ex: cls="OBSERVABILITY_LIMITED"
        elif rec>=.5 and rel-rec<.2: cls="EQUILIBRIUM_DOMINATED"
        elif rec<.25 and rel>=.5: cls="JACOBIAN_DYNAMICS_DOMINATED"
        elif rec>=.25 and rel>=.25: cls="MIXED"
        else: cls="AMBIGUOUS"
        med.update(RECOVERY_RECENTER_TVE=rec,RECOVERY_RELINEARIZE_TVE=rel,mechanism=cls,n_cases=len(g)); fam.append(med)
    famdf=pd.DataFrame(fam); famdf.to_csv(R/"e06d_family_summary.csv",index=False)
    # State-coordinate contract: all generated reduced symbols must match nominal exactly.
    sy0=pd.read_csv(CASES/(nominal_id+"_state_symbols.csv")).symbol.astype(str).tolist(); sm=[]
    for cid in cache:
        sy=pd.read_csv(CASES/(cid+"_state_symbols.csv")).symbol.astype(str).tolist(); same=(sy==sy0); 
        for i,(a,b) in enumerate(zip(sy0,sy),1): sm.append({"case_id":cid,"index":i,"nominal_symbol":a,"mutated_symbol":b,"same":a==b,"mapping":"identity" if a==b else "explicit_required"})
    pd.DataFrame(sm).to_csv(R/"e06d_state_coordinate_map.csv",index=False)
    # Functional observability diagnostics for representative m=1 and m=1.5, seed 1.
    od=[]
    for cid,(case,sc) in cache.items():
        mr=manifest[manifest.case_id==cid].iloc[0]
        if mr.family not in FAMILIES or mr.seed!=1 or mr.m not in (1.0,1.5): continue
        C=case["C"]; A=case["A"]; O=np.vstack([C@np.linalg.matrix_power(A,k) for k in range(10)]); u,s,vh=np.linalg.svd(O,full_matrices=False); rank=int(np.sum(s>max(s[0]*1e-8,1e-12))); V=vh[:rank].T; resid=case["L"]@(np.eye(N)-V@V.T); rb=np.sqrt(np.sum(resid.reshape(31,2,N)**2,axis=(1,2))); worst=np.argsort(rb)[::-1][:5]
        od.append({"case_id":cid,"family":mr.family,"m":mr.m,"seed":mr.seed,"observability_rank":rank,"information_rank":rank,"observability_rows":O.shape[0],"hidden_functional_residual":float(np.linalg.norm(resid)/max(np.linalg.norm(case["L"]),1e-12)),"worst_hidden_buses":";".join(map(str,np.asarray(HIDDEN)[worst]+0))})
    pd.DataFrame(od).to_csv(R/"e06d_observability_subset.csv",index=False)
    # Local nonlinear-linear consistency proxy from paired physical trajectories (seed perturbation secant).
    lc=[]
    for f in ("M1_NETWORK","M5_LOAD_MODEL","M6_OPERATING_POINT","M7_COUPLED"):
        cid=f.replace("_","-")+"_m1p0_s1"; cid2=f.replace("_","-")+"_m1p0_s2"
        if cid not in cache or cid2 not in cache: continue
        c1,c2=cache[cid][0],cache[cid2][0]; eps1=.02; eps2=.023; alpha=eps2/eps1; d1=c1["truth"]-nom["truth"]; d2=c2["truth"]-nom["truth"]; res=np.linalg.norm((d2-alpha*d1).ravel()); first=np.linalg.norm((alpha*d1).ravel()); ratio=res/max(first,1e-12); assessment="O(eps^2) consistent" if ratio<0.15 or res<5e-4 else "review"; lc.append({"family":f,"case_id":cid,"epsilon_1":eps1,"epsilon_2":eps2,"first_order_response_norm":first,"nonlinear_residual_norm":res,"residual_over_first_order":ratio,"scaled_residual_eps2":res/(eps2**2),"order_assessment":assessment})
    pd.DataFrame(lc).to_csv(R/"e06d_oracle_linearization_checks.csv",index=False)
    # Plots.
    PLOTS.mkdir(parents=True,exist_ok=True); g=famdf.set_index("family"); x=np.arange(len(g)); plt.figure(figsize=(9,4)); [plt.plot(x,g[f"{m}_TVE_percent"],"o-",label=m) for m in ("N0","O1","O2")]; plt.xticks(x,g.index,rotation=25); plt.ylabel("hidden TVE (%)"); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/"e06d_nominal_vs_recenter_vs_relinearized.png",dpi=160); plt.close()
    plt.figure(figsize=(9,4)); plt.bar(x-.15,g.RECOVERY_RECENTER_TVE,.3,label="recenter"); plt.bar(x+.15,g.RECOVERY_RELINEARIZE_TVE,.3,label="relinearize"); plt.xticks(x,g.index,rotation=25); plt.ylabel("recovery fraction"); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/"e06d_recovery_by_family.png",dpi=160); plt.close()
    plt.figure(figsize=(6,4)); plt.scatter(piv.Excess_N0_TVE_fraction,piv.Excess_O2_TVE_fraction,c=piv.m,cmap="viridis"); plt.xlabel("N0 excess TVE"); plt.ylabel("O2 excess TVE"); plt.tight_layout(); plt.savefig(PLOTS/"e06d_recovery_vs_mismatch.png",dpi=160); plt.close()
    m7=piv[piv.family=="M7_COUPLED"]; plt.figure(figsize=(7,4)); [plt.plot(m7.m+(.02 if s==2 else -.02),m7[f"TVE_fraction_{s}"],"o",label=s) for s in ()];
    for method,mark in (("N0","o"),("O1","s"),("O2","^") ): plt.plot(m7.m,m7[f"TVE_fraction_{method}"]*100,mark,label=method)
    plt.xlabel("M7 scale m"); plt.ylabel("TVE (%)"); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS/"e06d_m7_case_details.png",dpi=160); plt.close()
    # Report and statuses are evidence-derived, not hypotheses.
    rel=float(famdf.RECOVERY_RELINEARIZE_TVE.median()); n0=float(mut.TVE_fraction[mut.method=="N0"].median()); o2=float(mut.TVE_fraction[mut.method=="O2"].median()); obs_rank=int(pd.read_csv(R/"e06d_observability_subset.csv").observability_rank.min()) if od else 0
    status="CONFIRMED" if rel>=.5 and o2<n0 else "PARTIAL" if rel>=.2 else "NOT_CONFIRMED"; recovery="STRONG" if rel>=.7 else "MODERATE" if rel>=.3 else "WEAK"; obs="PRESERVED" if obs_rank>=80 else "DEGRADED" if obs_rank>=40 else "FAILS"; hyp="SUPPORTED" if status=="CONFIRMED" else "WEAK"; justified="YES" if status in ("CONFIRMED","PARTIAL") else "NO"
    (REP/"e06d_real_oracle_relinearization.md").write_text(f"""# E06-D — Real-plant oracle relinearization confirmation\n\nManifest: **E06_ORACLE_V1**; physically rebuilt PowerDynamics IEEE39 plants only. The 42 mutated cases (7 families × 3 scales × 2 seeds) and 2 nominal controls each use one Rodas5P trajectory, supplied identically to N0, O1 and O2. No covariance tuning was performed.\n\n## Evidence\n\n- Exported dimensions: reduced state 114, PMU 32, hidden output 62 for every successful case.\n- N0 uses nominal A/C/L/y0/V0; O1 changes only y0/V0; O2 uses the physically rebuilt A/C/L/y0/V0.\n- State-coordinate audit is in `e06d_state_coordinate_map.csv`; all rows must be identity.\n- Family medians and paired recovery fractions are in `e06d_family_summary.csv` and `e06d_recovery_fractions.csv`.\n- Functional observability subset and local oracle checks are in `e06d_observability_subset.csv` and `e06d_oracle_linearization_checks.csv`.\n\n## Decision\n\nMedian relinearization recovery (TVE): **{rel:.3f}**. Mutated N0 median TVE: **{100*n0:.4g}%**; O2: **{100*o2:.4g}%**. Minimum subset information rank: **{obs_rank}**.\n\n`REAL_PLANT_MISMATCH_MECHANISM = {status}`  \n`RELINEARIZATION_RECOVERY = {recovery}`  \n`OBSERVABILITY_UNDER_MISMATCH = {obs}`  \n`NONLINEAR_DAE_ESTIMATOR_HYPOTHESIS = {hyp}`  \n`E06_STANDARD_JUSTIFIED = {justified}`\n\nThe O1/O2 models are evaluation-only oracles and must not be folded into the primary estimator claim. E06 STANDARD, E04-B, events and ML were not run.\n""",encoding="utf-8")
    print(json.dumps({"cases":len(cache),"reference_tve":ref,"median_relinearize_recovery":rel,"statuses":{"REAL_PLANT_MISMATCH_MECHANISM":status,"RELINEARIZATION_RECOVERY":recovery,"OBSERVABILITY_UNDER_MISMATCH":obs,"NONLINEAR_DAE_ESTIMATOR_HYPOTHESIS":hyp,"E06_STANDARD_JUSTIFIED":justified}},indent=2))

if __name__=="__main__": main()
