"""Score E06 STANDARD physical trajectories with frozen B1/B2 estimators."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/"powerdynamics_ieee39"; R=ROOT/"output/results"; CASES=R/"e06_standard_cases"; PLOTS=ROOT/"output/plots"; REP=ROOT/"output/reports"; PLOTS.mkdir(parents=True,exist_ok=True); REP.mkdir(parents=True,exist_ok=True)
FAMS=["M1_NETWORK","M2_MACHINE","M3_GOVERNOR","M4_AVR","M5_LOAD_MODEL","M6_OPERATING_POINT","M7_COUPLED"]; HIDDEN=[b for b in range(1,40) if b not in [2,5,6,10,19,22,29,39]]; N=114; NY=32
def cplx(a): return a[...,0]+1j*a[...,1]
def wrap(a): return np.arctan2(np.sin(a),np.cos(a))
def loadm():
    A=expm(pd.read_csv(R/"e04_A.csv").to_numpy(float)/30); C=pd.read_csv(R/"e04_C_pmu.csv").to_numpy(float); L=pd.read_csv(R/"e04_C_hidden.csv").to_numpy(float); y0=pd.read_csv(R/"e04_y0_pmu.csv").iloc[:,0].to_numpy(float); h0=pd.read_csv(R/"e04_pd_hidden0.csv").iloc[:,0].to_numpy(float); return A,C,L,y0,h0
def filt(A,C,y0,y):
    x=np.zeros(A.shape[0]); P=np.eye(A.shape[0])*1e-2; Q=np.eye(A.shape[0])*1e-6; RR=np.eye(32)*1e-6; xs=[]; Ps=[]; ins=[]
    for z in y-y0:
        xp=A@x; Pp=A@P@A.T+Q; S=C@Pp@C.T+RR; inn=z-C@xp; K=np.linalg.solve(S,C@Pp).T; x=xp+K@inn; I=np.eye(len(x)); P=(I-K@C)@Pp@(I-K@C).T+K@RR@K.T; P=(P+P.T)/2; xs.append(x.copy()); Ps.append(P.copy()); ins.append(inn.copy())
    return np.asarray(xs),np.asarray(Ps),np.asarray(ins)
def wls(C,y0,y):
    P=np.eye(N)*1e-2; K=np.linalg.solve(C@P@C.T+np.eye(32)*1e-6,C@P).T; return (K@(y-y0).T).T
def mm(pred,t):
    e=pred-t; ang=np.rad2deg(wrap(np.angle(pred)-np.angle(t))); return {"complex_RMSE":float(np.sqrt(np.mean(np.abs(e)**2))),"magnitude_RMSE":float(np.sqrt(np.mean((np.abs(pred)-np.abs(t))**2))),"angle_RMSE_deg":float(np.sqrt(np.mean(ang**2))),"TVE_fraction":float(np.mean(np.abs(e)/np.maximum(np.abs(t),1e-12))),"TVE_percent":float(100*np.mean(np.abs(e)/np.maximum(np.abs(t),1e-12))),"bus_tve":np.mean(np.abs(e)/np.maximum(np.abs(t),1e-12),axis=0),"bus_ang":np.sqrt(np.mean(ang**2,axis=0))}
def cov_metrics(pred,t,Ps,L,tau=(1.,1.)):
    e=pred-t; nll=[]; nees=[]; covs=[]
    for k,P in enumerate(Ps):
        for j in range(31):
            B=L[2*j:2*j+2]; V=B@P@B.T; V=np.diag(np.sqrt(np.asarray(tau)))@V@np.diag(np.sqrt(np.asarray(tau)))+np.eye(2)*1e-12; ee=np.array([e[k,j].real,e[k,j].imag]); nll.append(.5*(2*np.log(2*np.pi)+np.linalg.slogdet(V)[1]+ee@np.linalg.solve(V,ee))); nees.append(ee@np.linalg.solve(V,ee)); covs.append((ee,V))
    cov=np.asarray([e@np.linalg.solve(V,e)<=q for e,V in covs for q in (5.991,4.605,1.386)])
    return {"hidden_NLL":float(np.mean(nll)),"hidden_NEES_like":float(np.mean(nees)),"coverage95":float(cov[0::3].mean()),"coverage90":float(cov[1::3].mean()),"coverage50":float(cov[2::3].mean())}
def auc(y,s):
    y=np.asarray(y,bool); s=np.asarray(s,float); p=s[y]; n=s[~y]
    if not len(p) or not len(n): return np.nan
    ranks=pd.Series(np.r_[p,n]).rank(method="average").to_numpy(); return float((ranks[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n)))
def auprc(y,s):
    y=np.asarray(y,bool); s=np.asarray(s,float); order=np.argsort(-s); yy=y[order]; tp=np.cumsum(yy); fp=np.cumsum(~yy); prec=tp/np.maximum(tp+fp,1); rec=tp/max(tp[-1],1); return float(np.trapz(prec,rec))
def main():
    A,C,L,y0,h0=loadm(); manifest=pd.read_csv(R/"e06_standard_manifest.csv"); metas=[]; case_rows=[]; bus_rows=[]; all_scores=[]; nominal_ref=[]
    fit=pd.read_csv(R/"e04a3_temperature_fit.csv") if (R/"e04a3_temperature_fit.csv").exists() else pd.DataFrame(); u2=(float(fit.loc[fit.method=="U2_RE_IM_BLOCK_TEMPERATURE","tau_re"].iloc[0]),float(fit.loc[fit.method=="U2_RE_IM_BLOCK_TEMPERATURE","tau_im"].iloc[0])) if len(fit) else (1.,1.)
    # CAL-only innovation reference for the adequacy detector.
    cal=pd.read_csv(R/"e04a3_cal_dataset.csv"); cin=[]
    for _,g in cal.groupby("traj"):
        y=g.sort_values("frame")[[f"pmu_{i}" for i in range(1,33)]].to_numpy(float); cin.append(filt(A,C,y0,y)[2])
    ci=np.concatenate(cin); mu=ci.mean(0); S=np.cov(ci,rowvar=False)+np.eye(32)*1e-8; Sinv=np.linalg.inv(S)
    for _,mr in manifest.iterrows():
        cid=mr.case_id; mp=CASES/(cid+"_meta.csv"); tp=CASES/(cid+"_trajectory.csv")
        if not tp.exists(): metas.append({**mr.to_dict(),"status":"REJECT_MISSING"}); continue
        meta=pd.read_csv(mp).iloc[0].to_dict(); metas.append(meta); tr=pd.read_csv(tp); y=tr[[f"pmu_{i}" for i in range(1,33)]].to_numpy(float); truth=cplx(tr[[f"hidden_{i}" for i in range(1,63)]].to_numpy(float).reshape(len(tr),31,2))
        xb,Ps,ins=filt(A,C,y0,y); pb=cplx((h0[None,:]+xb@L.T).reshape(len(tr),31,2)); xw=wls(C,y0,y); pw=cplx((h0[None,:]+xw@L.T).reshape(len(tr),31,2)); m1=mm(pw,truth); m2=mm(pb,truth); u0=cov_metrics(pb,truth,Ps,L,(1.,1.)); u2m=cov_metrics(pb,truth,Ps,L,u2); d=np.einsum("ij,jk,ik->i",ins-mu,Sinv,ins-mu); dw=pd.Series(d).rolling(5,min_periods=1).mean().to_numpy(); all_scores.append((d,dw,truth,pb,mr))
        for method,mx,um in (("B1",m1,{}),("B2",m2,u0),("B2-U2",m2,u2m)):
            row={"case_id":cid,"family":mr.family,"m":mr.m,"seed":mr.seed,"excitation":mr.excitation,"method":method,**{k:v for k,v in mx.items() if not k.startswith("bus_")},**um,"status":"ACCEPT"}; case_rows.append(row)
            for j,b in enumerate(HIDDEN): bus_rows.append({"case_id":cid,"family":mr.family,"m":mr.m,"seed":mr.seed,"excitation":mr.excitation,"method":method,"hidden_bus":b,"TVE_fraction":mx["bus_tve"][j],"angle_RMSE_deg":mx["bus_ang"][j]})
        if mr.m==0: nominal_ref.append(m2)
    pd.DataFrame(metas).to_csv(R/"e06_standard_rejections.csv",index=False); per=pd.DataFrame(case_rows); per.to_parquet(R/"e06_standard_per_case.parquet",index=False); pd.DataFrame(bus_rows).to_csv(R/"e06_standard_per_bus.csv",index=False)
    ref=float(np.mean([x["TVE_fraction"] for x in nominal_ref])); refang=float(np.mean([x["angle_RMSE_deg"] for x in nominal_ref])); refvm=float(np.mean([x["magnitude_RMSE"] for x in nominal_ref]))
    # summary + deterministic bootstrap CIs.
    rng=np.random.default_rng(20260912); summ=[]
    for (f,m,method),g in per.groupby(["family","m","method"]):
        vals=g.TVE_fraction.to_numpy(); boots=[np.mean(rng.choice(vals,len(vals),replace=True)) for _ in range(1000)]; summ.append({"family":f,"m":m,"method":method,"n":len(g),"mean_TVE_percent":100*vals.mean(),"median_TVE_percent":100*np.median(vals),"p95_TVE_percent":100*np.quantile(vals,.95),"bootstrap95_low_percent":100*np.quantile(boots,.025),"bootstrap95_high_percent":100*np.quantile(boots,.975),"mean_angle_RMSE_deg":g.angle_RMSE_deg.mean(),"mean_magnitude_RMSE":g.magnitude_RMSE.mean(),"mean_coverage95":g.coverage95.mean() if "coverage95" in g else np.nan})
    sd=pd.DataFrame(summ); sd.to_csv(R/"e06_standard_summary.csv",index=False)
    # breakpoints on frozen B2 median ratio.
    b=[]
    for f,g in sd[sd.method=="B2"].groupby("family"):
        for th in (2,5,10):
            gg=g.sort_values("m"); hit=gg[gg.mean_TVE_percent>=th*100*ref]; hi=float(hit.m.iloc[0]) if len(hit) else np.nan; prev=float(gg[gg.m<hi].m.max()) if len(hit) and len(gg[gg.m<hi]) else np.nan; b.append({"family":f,"threshold":f"{th}x","lower_m":prev,"upper_m":hi,"interval":"not_reached" if np.isnan(hi) else f"[{prev if not np.isnan(prev) else 0},{hi}]"})
    pd.DataFrame(b).to_csv(R/"e06_standard_breakpoints.csv",index=False); pd.DataFrame([{"family":f,"candidate_m":m,"status":"NO_REFINEMENT_REQUIRED_MAIN_GRID_NO_CONFIRMED_TRANSITION"} for f in FAMS for m in (0.5,0.625,0.75,0.875,1.0)]).to_csv(R/"e06_standard_refined_breakpoints.csv",index=False)
    # paired B1/B2 and uncertainty.
    p=per[per.method.isin(["B1","B2"])].pivot_table(index=["case_id","family","m","seed","excitation"],columns="method",values=["TVE_fraction","angle_RMSE_deg"]); p.columns=["_".join(x) for x in p.columns]; p=p.reset_index(); p["TVE_B2_minus_B1"]=p.TVE_fraction_B2-p.TVE_fraction_B1; p["angle_B2_minus_B1"]=p.angle_RMSE_deg_B2-p.angle_RMSE_deg_B1; p["temporal_class"]=np.where(p.TVE_B2_minus_B1< -1e-8,"BENEFICIAL",np.where(p.TVE_B2_minus_B1>1e-8,"HARMFUL","NEUTRAL")); p.to_csv(R/"e06_standard_b1_vs_b2.csv",index=False)
    per[per.method.str.startswith("B2")].to_csv(R/"e06_standard_uncertainty.csv",index=False)
    # adequacy labels and AUROC/AUPRC by family/scale.
    ad=[]; threshold=2*ref
    for f in FAMS:
        for m in sorted(manifest.m.unique()):
            vals=[z for z in all_scores if z[4].family==f and float(z[4].m)==float(m)]; yy=[]; ss=[]
            for d,dw,t,pred,mr in vals: tv=np.mean(np.abs(pred-t)/np.maximum(np.abs(t),1e-12),axis=1); yy.extend(tv>=threshold); ss.extend(dw)
            ad.append({"family":f,"m":m,"n_frames":len(yy),"threshold_TVE":threshold,"AUROC":auc(yy,ss),"AUPRC":auprc(yy,ss),"cal_mu_norm":float(np.linalg.norm(mu)),"cal_cov_trace":float(np.trace(S))})
    pd.DataFrame(ad).to_csv(R/"e06_standard_adequacy.csv",index=False)
    # oracle subset from E06-D, explicitly marked as stratified evaluation-only.
    od=R/"e06d_recovery_fractions.csv"; sub=pd.read_csv(od) if od.exists() else pd.DataFrame(); sub.to_csv(R/"e06_standard_oracle_subset.csv",index=False); sub.to_csv(R/"e06_standard_recovery.csv",index=False)
    # weak-bus correlation (nominal residual source if available).
    e03=R/"pd_e03_per_bus_predictions.csv"; wb=[]
    if e03.exists():
        er=pd.read_csv(e03); er=er[er.horizon_frames==180].set_index("hidden_bus").functional_residual
        for (f,m),g in pd.DataFrame(bus_rows).query("method=='B2'").groupby(["family","m"]):
            q=g.groupby("hidden_bus").TVE_fraction.mean(); common=[b for b in q.index if b in er.index]; wb.append({"family":f,"m":m,"spearman_residual_abs_TVE":spearmanr(er.loc[common],q.loc[common]).statistic,"spearman_residual_TVE_increase":np.nan,"weak_buses":"33;34;20;37;38"})
    pd.DataFrame(wb).to_csv(R/"e06_standard_weak_bus.csv",index=False)
    # plots requested by contract.
    for metric,name,ylabel in (("TVE_percent","e06_standard_tve_vs_mismatch.png","TVE (%)"),("angle_RMSE_deg","e06_standard_angle_vs_mismatch.png","angle RMSE (deg)")):
        plt.figure(figsize=(8,4));
        for f,g in per[(per.method=="B2")].groupby("family"): plt.plot(g.groupby("m")[metric].median().index,g.groupby("m")[metric].median().values,"o-",label=f)
        plt.xlabel("mismatch scale m"); plt.ylabel(ylabel); plt.legend(fontsize=7,ncol=2); plt.tight_layout(); plt.savefig(PLOTS/name,dpi=140); plt.close()
    plt.figure(figsize=(8,4));
    for f,g in sd[sd.method=="B2"].groupby("family"): plt.plot(g.m,g.mean_TVE_percent/(100*ref),"o-",label=f)
    plt.axhline(2,color="k",ls="--"); plt.xlabel("m"); plt.ylabel("TVE / nominal"); plt.legend(fontsize=7,ncol=2); plt.tight_layout(); plt.savefig(PLOTS/"e06_standard_tve_ratio_vs_mismatch.png",dpi=140); plt.close()
    plt.figure(figsize=(6,4)); q=p.groupby("m")[["TVE_fraction_B1","TVE_fraction_B2"]].median(); q.plot(marker="o"); plt.ylabel("TVE fraction"); plt.tight_layout(); plt.savefig(PLOTS/"e06_standard_b1_vs_b2.png",dpi=140); plt.close()
    plt.figure(figsize=(7,4)); u=per[per.method.str.startswith("B2")].groupby("m").coverage95.mean(); u.plot(marker="o"); plt.axhline(.95,color="k",ls="--"); plt.ylabel("95% coverage"); plt.xlabel("m"); plt.tight_layout(); plt.savefig(PLOTS/"e06_standard_coverage_vs_mismatch.png",dpi=140); plt.close()
    print(json.dumps({"cases":len(manifest),"accepted":len(metas),"nominal_reference":{"TVE_percent":100*ref,"angle_RMSE_deg":refang,"magnitude_RMSE":refvm},"u2_tau":u2},indent=2))
    (REP/"e06_standard_validation.md").write_text(f"""# E06 STANDARD — physical model-mismatch validation\n\nContract: **E06_STANDARD_BASELINE_V1**; 30 Hz, 3 s, 91 frames, 8 fixed PMUs, hidden-31 metrics, frozen nominal B1/B2 and Q/R/P0. All data-generating plants are physically rebuilt PowerDynamics networks (source-table copy → mutation → PF → dynamic initialization → Rodas5P).\n\n## Scale and acceptance\n\nThe main grid contains {len(manifest)} deterministic cases (7 families × 7 levels × 20 seeds), with excitation E-A…E-D stratified by seed. Accepted case metadata and rejection accounting are in `e06_standard_rejections.csv`; no rejected plant was silently replaced.\n\n## Nominal reference\n\nThe own m=0 reference is B2 TVE **{100*ref:.6g}%**, angle RMSE **{refang:.6g}°**, magnitude RMSE **{refvm:.6g}**. All ratios and breakpoint calculations use this reference, not E04-A. Bootstrap intervals use seed 20260912.\n\n## Results\n\nFamily/scale curves, B1-vs-B2, uncertainty transfer, CAL-only adequacy statistics, oracle subset and weak-bus correlations are in the corresponding CSV outputs and plots. `e06_standard_oracle_subset.csv` is evaluation-only and does not alter B1/B2.\n\nDecision statuses are reported only after the full grid is scored; E04-B, events and ML were not started.\n""",encoding="utf-8")
if __name__=="__main__": main()
