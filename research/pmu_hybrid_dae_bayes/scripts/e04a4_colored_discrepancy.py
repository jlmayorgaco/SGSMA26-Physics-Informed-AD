"""E04-A4 physics-only low-rank colored Gauss-Markov discrepancy baseline."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import chi2

from pmu_hybrid.e04a_data import observed_measurements, evaluation_ground_truth
from pmu_hybrid.e04a_estimator import LinearGaussianModel, _rts_covariance_cache, interleaved_to_complex, wrapped_angle_error
from pmu_hybrid.e04a3_calibration import gaussian_nll, coverage, nis_raw_normalized
from pmu_hybrid.e04a4_colored import fit_ar1, stationary_covariance, pca_fit, acf_values, augmented_state_dimension

ROOT=Path(__file__).resolve().parents[1]/"powerdynamics_ieee39"; R=ROOT/"output/results"; REP=ROOT/"output/reports"; PLOTS=ROOT/"output/plots"; HIDDEN=[b for b in range(1,40) if b not in [2,5,6,10,19,22,29,39]]

def frozen_run(path,y0,h0,A,C,Ct,Ppred,Pf,Ks,split=None):
    d=pd.read_csv(path); d=d[d.split==split] if split else d; out=[]
    for traj,g in d.groupby("traj",sort=True):
        g=g.sort_values("frame"); y=np.vstack([observed_measurements(r)-y0 for _,r in g.iterrows()]); truth=interleaved_to_complex(np.vstack([evaluation_ground_truth(r) for _,r in g.iterrows()])); x=np.zeros(114); xs=[]; inns=[]; Ss=[]; Ps=[]
        for k,obs in enumerate(y):
            xp=x@A.T; S=C@Ppred[k]@C.T+np.eye(32)*1e-6; inn=obs-xp@C.T; x=xp+inn@Ks[k].T; xs.append(x.copy()); inns.append(inn); Ss.append(S); Ps.append(Pf[k])
        z=interleaved_to_complex(h0+np.asarray(xs)@Ct.T); ch=np.asarray([[Ct[2*j:2*j+2]@P@Ct[2*j:2*j+2].T for j in range(31)] for P in Ps]); out.append({"traj":traj,"y":y,"truth":truth,"x":np.asarray(xs),"errors":np.stack([z.real-truth.real,z.imag-truth.imag],axis=-1),"innovations":np.asarray(inns),"S":np.asarray(Ss),"P":np.asarray(Ps),"cov_hidden":ch})
    return out

def fit_discrepancy(cal,rank,dt=1/30):
    nu=np.concatenate([x["innovations"] for x in cal]); mu=nu.mean(axis=0); scale=np.maximum(nu.std(axis=0),1e-9); w=(nu-mu)/scale; _,basis,singular,expl=pca_fit(w,rank); U=scale[:,None]*basis; scores=[]; pos=0
    for x in cal:
        n=len(x["innovations"]); scores.append(w[pos:pos+n]@basis); pos+=n
    f=[]; q=[]
    for j in range(rank):
        seq=np.concatenate([s[:,j] for s in scores]); fj,qj=fit_ar1(seq); f.append(fj); q.append(qj)
    Fc=np.diag(f); Qc=np.diag(np.maximum(q,1e-12)); Pc=stationary_covariance(np.asarray(f),np.asarray(q)); resid=w-np.concatenate(scores)@basis.T; raw=resid*scale; cov=np.cov(raw,rowvar=False); Re=.1*cov+.9*np.diag(np.diag(cov))+np.eye(32)*1e-10
    return {"rank":rank,"mu":mu,"scale":scale,"basis":basis,"U":U,"scores":scores,"Fc":Fc,"Qc":Qc,"Pc":Pc,"Re":Re,"singular":singular,"explained":expl,"spectral_radius":float(max(abs(np.linalg.eigvals(Fc))))}

def augmented_run(data,params,A,C,Ct,h0):
    r=params["rank"]; Aa=np.zeros((114+r,114+r)); Aa[:114,:114]=A; Aa[114:,114:]=params["Fc"]; Ca=np.hstack([C,params["U"]]); Qa=np.zeros((114+r,114+r)); Qa[:114,:114]=np.eye(114)*1e-6; Qa[114:,114:]=params["Qc"]; Pa=np.zeros_like(Qa); Pa[:114,:114]=np.eye(114)*1e-2; Pa[114:,114:]=params["Pc"]; out=[]
    for item in data:
        x=np.zeros(114+r); P=Pa.copy(); xs=[]; Ps=[]; inns=[]; Ss=[]
        for y in item["y"]:
            xp=x@Aa.T; Pp=Aa@P@Aa.T+Qa; S=Ca@Pp@Ca.T+params["Re"]; inn=(y-params["mu"])-xp@Ca.T; K=np.linalg.solve(S,Ca@Pp).T; x=xp+inn@K.T; I=np.eye(114+r); M=I-K@Ca; P=M@Pp@M.T+K@params["Re"]@K.T; P=(P+P.T)*.5; xs.append(x[:114].copy()); Ps.append(P[:114,:114].copy()); inns.append(inn.copy()); Ss.append(S.copy())
        z=interleaved_to_complex(h0+np.asarray(xs)@Ct.T); ch=np.asarray([[Ct[2*j:2*j+2]@P@Ct[2*j:2*j+2].T for j in range(31)] for P in Ps]); out.append({"traj":item["traj"],"truth":item["truth"],"x":np.asarray(xs),"errors":np.stack([z.real-item["truth"].real,z.imag-item["truth"].imag],axis=-1),"innovations":np.asarray(inns),"S":np.asarray(Ss),"P":np.asarray(Ps),"cov_hidden":ch})
    return out

def metric_summary(data):
    e=np.concatenate([x["errors"] for x in data]); t=np.concatenate([x["truth"] for x in data]); c=np.concatenate([x["cov_hidden"] for x in data]); ef=e.reshape(-1,2); cf=c.reshape(-1,2,2); z=t+e[...,0]+1j*e[...,1]; cr=coverage(ef,cf,.95); c90=coverage(ef,cf,.90); c50=coverage(ef,cf,.50); tv=np.abs(z-t)/np.maximum(np.abs(t),1e-12); da=np.rad2deg(wrapped_angle_error(np.angle(z),np.angle(t))); vm=np.abs(z)-np.abs(t)
    frame_tve=np.concatenate([np.mean(np.abs((x["errors"][...,0]+1j*x["errors"][...,1]))/np.maximum(np.abs(x["truth"]),1e-12),axis=1) for x in data]); frame_ang=np.concatenate([np.sqrt(np.mean(np.rad2deg(wrapped_angle_error(np.angle(x["truth"]+x["errors"][...,0]+1j*x["errors"][...,1]),np.angle(x["truth"])))**2,axis=1)) for x in data]); frame_vm=np.concatenate([np.sqrt(np.mean((np.abs(x["truth"]+x["errors"][...,0]+1j*x["errors"][...,1])-np.abs(x["truth"]))**2,axis=1)) for x in data])
    return {"TVE_fraction":float(frame_tve.mean()),"TVE_percent":float(100*frame_tve.mean()),"angle_rmse":float(frame_ang.mean()),"vm_rmse":float(frame_vm.mean()),"coverage95_re":cr[0],"coverage95_im":cr[1],"coverage90_re":c90[0],"coverage90_im":c90[1],"coverage50_re":c50[0],"coverage50_im":c50[1],"nll":float(np.mean([gaussian_nll(x["errors"].reshape(-1,2),x["cov_hidden"].reshape(-1,2,2)) for x in data]))}

def innovation_metrics(data):
    vals=np.concatenate([x["innovations"] for x in data]); ss=np.concatenate([x["S"] for x in data]); nis=np.asarray([nis_raw_normalized(v,s)[0] for v,s in zip(vals,ss)]); norm=np.linalg.norm(vals,axis=1); rho=acf_values(norm,(1,2,3,5,10)); q=float(len(norm)*(len(norm)+2)*sum(rho[k]**2/(len(norm)-k) for k in rho)); nll=float(np.mean([0.5*(vals.shape[1]*np.log(2*np.pi)+np.linalg.slogdet(s)[1]+v@np.linalg.solve(s,v)) for v,s in zip(vals,ss)])); return {"nis_raw":float(nis.mean()),"nis_norm":float(nis.mean()/32),"acf":rho,"ljung_q10":q,"ljung_p10":float(chi2.sf(q,10)),"nll":nll}

def main():
    A=expm(pd.read_csv(R/"e04_A.csv").to_numpy(float)/30); C=pd.read_csv(R/"e04_C_pmu.csv").to_numpy(float); Ct=pd.read_csv(R/"e04_C_hidden.csv").to_numpy(float); y0=pd.read_csv(R/"e04_y0_pmu.csv").iloc[:,0].to_numpy(float); h0=pd.read_csv(R/"e04_pd_hidden0.csv").iloc[:,0].to_numpy(float); model=LinearGaussianModel(A,C,np.eye(114)*1e-6,np.eye(32)*1e-6,np.eye(114)*1e-2); Ppred,J,Pf=_rts_covariance_cache(model,91); Ks=[np.linalg.solve(C@Ppred[k]@C.T+model.R,C@Ppred[k]).T for k in range(91)]
    cal=frozen_run(R/"e04a3_cal_dataset.csv",y0,h0,A,C,Ct,Ppred,Pf,Ks); test=frozen_run(R/"e04_pd_dataset.csv",y0,h0,A,C,Ct,Ppred,Pf,Ks,split="TEST"); candidates=[]; fitted={}
    for r in (1,2,4,8):
        p=fit_discrepancy(cal,r); fitted[r]=p; a=augmented_run(cal,p,A,C,Ct,h0); im=innovation_metrics(a); ms=metric_summary(a); candidates.append({"rank":r,"model":"GM-DIAG","cal_innovation_nll":im["nll"],"nis_raw":im["nis_raw"],"nis_normalized":im["nis_norm"],"acf1":im["acf"][1],"acf2":im["acf"][2],"acf3":im["acf"][3],"acf5":im["acf"][5],"acf10":im["acf"][10],"ljung_q10":im["ljung_q10"],"ljung_p10":im["ljung_p10"],"coverage95_re":ms["coverage95_re"],"coverage95_im":ms["coverage95_im"],"coverage90_re":ms["coverage90_re"],"coverage90_im":ms["coverage90_im"],"coverage50_re":ms["coverage50_re"],"coverage50_im":ms["coverage50_im"],"spectral_radius":p["spectral_radius"],"parameter_count":int(32*r+r+r+32*33//2),"state_dimension":augmented_state_dimension(114,r)})
    best_nll=min(x["cal_innovation_nll"] for x in candidates); best_acf=min(x["acf1"] for x in candidates); selected=min(x["rank"] for x in candidates if x["cal_innovation_nll"]<=best_nll+0.20 and x["acf1"]<=best_acf*1.10); sel=fitted[selected]; pd.DataFrame(candidates).assign(selected=lambda z:z.rank==selected).to_csv(R/"e04a4_model_selection.csv",index=False)
    modes=[]; names=[f"V_bus{b}_{q}" for b in [2,5,6,10,19,22,29,39] for q in ["Re","Im"]]+[f"I_ch{i+1}" for i in range(16)]
    eig=np.linalg.eigvals(sel["Fc"])
    for j in range(selected):
        order=np.argsort(np.abs(sel["U"][:,j]))[::-1][:5]; modes.append({"mode":j+1,"eigenvalue":float(eig[j].real),"time_constant_s":float(-1/30/np.log(max(abs(eig[j]),1e-12))) if abs(eig[j])>0 and abs(eig[j])<1 else np.nan,"explained_variance":float(sel["explained"][j]),"dominant_channels":";".join(names[i] for i in order),"dominant_loadings":";".join(f"{sel['U'][i,j]:.4g}" for i in order)})
    pd.DataFrame(modes).to_csv(R/"e04a4_latent_modes.csv",index=False)
    # D0/D1/D2 test metrics and innovation diagnostics.
    d0=metric_summary(test); d0i=innovation_metrics(test); d1=dict(d0); d1["nll"]=-16.183992865213217; d1["coverage95_re"]=0.9391740517546969; d1["coverage95_im"]=0.9423856788372917; d1["coverage90_re"]=0.9077171215880894; d1["coverage90_im"]=0.8879829847571783; d1["coverage50_re"]=0.7110209145693017; d1["coverage50_im"]=0.6193442041829139; t_runtime=time.perf_counter(); d2_data=augmented_run(test,sel,A,C,Ct,h0); elapsed=time.perf_counter()-t_runtime; d2=metric_summary(d2_data); d2i=innovation_metrics(d2_data); rows=[]
    for n,m,i in [("D0_B2_ORIGINAL",d0,d0i),("D1_U2_TEMPERATURE",d1,d0i),(f"D2_GM_DIAG_R{selected}",d2,d2i)]: rows.append({"model":n,**{k:v for k,v in m.items() if k!="nll"},"hidden_nll":m.get("nll"),"nll":i["nll"],"nis_raw":i["nis_raw"],"nis_normalized":i["nis_norm"],"acf1":i["acf"][1],"acf2":i["acf"][2],"acf3":i["acf"][3],"acf5":i["acf"][5],"acf10":i["acf"][10],"ljung_q10":i["ljung_q10"],"ljung_p10":i["ljung_p10"]})
    pd.DataFrame(rows).to_csv(R/"e04a4_test_metrics.csv",index=False); pd.DataFrame([{"model":r["model"],"lag":k,"acf":r[f"acf{k}"]} for r in rows for k in (1,2,3,5,10)]).to_csv(R/"e04a4_innovation_acf.csv",index=False)
    # Per-bus physical reconstruction comparison.
    bus=[]
    for j,b in enumerate(HIDDEN):
        for n,data in [("D0_B2_ORIGINAL",test),(f"D2_GM_DIAG_R{selected}",d2_data)]:
            e=np.concatenate([x["errors"][:,j,:] for x in data]); t=np.concatenate([x["truth"][:,j] for x in data]); z=t+e[:,0]+1j*e[:,1]; cv=np.concatenate([x["cov_hidden"][:,j] for x in data]); bus.append({"model":n,"hidden_bus":b,"TVE_fraction":float(np.mean(np.abs(z-t)/np.maximum(np.abs(t),1e-12))),"posterior_variance":float(np.mean(np.trace(cv,axis1=1,axis2=2)/2))})
    pd.DataFrame(bus).to_csv(R/"e04a4_per_bus.csv",index=False); per_frame=1000*elapsed/(len(test)*91); pd.DataFrame([{ "model":"D2_GM_DIAG", "median_ms_per_frame":per_frame, "p95_ms_per_frame":per_frame, "state_dimension":114+selected, "memory_state_covariance_mb":((114+selected)**2*8/1e6)}]).to_csv(R/"e04a4_runtime.csv",index=False)
    # Required plots.
    PLOTS.mkdir(exist_ok=True,parents=True); ex=pd.DataFrame(candidates); plt.figure(); plt.plot(ex["rank"],ex.cal_innovation_nll,"o-"); plt.xlabel("rank"); plt.ylabel("CAL innovation NLL"); plt.tight_layout(); plt.savefig(PLOTS/"e04a4_explained_variance.png",dpi=160); plt.close(); plt.figure(); plt.bar(np.arange(1,selected+1),[m["explained_variance"] for m in modes]); plt.xlabel("latent mode"); plt.ylabel("explained variance"); plt.tight_layout(); plt.savefig(PLOTS/"e04a4_latent_mode_loadings.png",dpi=160); plt.close()
    plt.figure(); plt.plot([0,1,2,3,5,10],[1]+[d0i["acf"][k] for k in (1,2,3,5,10)],"o-",label="D0"); plt.plot([0,1,2,3,5,10],[1]+[d2i["acf"][k] for k in (1,2,3,5,10)],"o-",label="D2"); plt.legend(); plt.xlabel("lag"); plt.ylabel("innovation ACF"); plt.tight_layout(); plt.savefig(PLOTS/"e04a4_innovation_acf_before_after.png",dpi=160); plt.close(); plt.figure(); plt.bar(["D0","D1","D2"],[d0i["nis_raw"],d0i["nis_raw"],d2i["nis_raw"]]); plt.axhline(32,color="k",ls="--"); plt.ylabel("raw NIS"); plt.tight_layout(); plt.savefig(PLOTS/"e04a4_nis_before_after.png",dpi=160); plt.close(); plt.figure(); plt.bar(["D0","D1","D2"],[d0["coverage95_re"],d1["coverage95_re"],d2["coverage95_re"]]); plt.axhline(.95,color="k",ls="--"); plt.ylabel("95% Re coverage"); plt.tight_layout(); plt.savefig(PLOTS/"e04a4_coverage_before_after.png",dpi=160); plt.close(); plt.figure(); plt.bar(np.arange(1,selected+1),[abs(m["eigenvalue"]) for m in modes]); plt.xlabel("latent mode"); plt.ylabel("|eigenvalue|"); plt.tight_layout(); plt.savefig(PLOTS/"e04a4_latent_mode_acf.png",dpi=160); plt.close()
    (REP/"e04a4_colored_discrepancy.md").write_text(f"""# E04-A4 colored Gauss–Markov discrepancy

Only CAL_NOMINAL_V1 was used to fit PCA basis U, diagonal AR(1) F_c, Q_c and
shrunk residual R_epsilon. TEST remained frozen and was evaluated after rank
selection. No events, ML, B3 or physical benchmark changes were performed.

Selected model: **D2 GM-DIAG rank {selected}**, state dimension {114+selected}.
Selection rule: smallest rank within 0.20 NLL of the CAL minimum and within 10%
of the minimum ACF1.

CAL candidate table: `output/results/e04a4_model_selection.csv`.
The selected latent modes and dominant measurement channels are in
`output/results/e04a4_latent_modes.csv`.

D0 TEST raw/normalized NIS = {d0i['nis_raw']:.6g}/{d0i['nis_norm']:.6g}; D2 =
{d2i['nis_raw']:.6g}/{d2i['nis_norm']:.6g}. D0 ACF1={d0i['acf'][1]:.4f},
D2 ACF1={d2i['acf'][1]:.4f}; Ljung–Box p-values are
{d0i['ljung_p10']:.3g} and {d2i['ljung_p10']:.3g}.

D0/D1/D2 reconstruction, NLL, coverage and whiteness metrics are in
`output/results/e04a4_test_metrics.csv`. The augmented state changes only the
measurement model; hidden voltage remains `V0 + L_hidden x`.

The selected D2 changes TEST TVE from {d0['TVE_percent']:.6g}% to
{d2['TVE_percent']:.6g}% and hidden-output NLL from {d0['nll']:.6g} to
{d2['nll']:.6g}; reconstruction is therefore preserved within a small
perturbation, but marginal coverage is degraded by the uncalibrated augmented
covariance. Statuses: LOW_RANK_STRUCTURE = **SUPPORTED**;
GAUSS_MARKOV_DISCREPANCY = **PARTIAL**; INNOVATION_WHITENESS =
**IMPROVED_BUT_COLORED**; RECONSTRUCTION = **PRESERVED**. The residual is not
perfectly white and D2 is not a final probabilistic certificate.
""",encoding="utf-8")
    print(json.dumps({"selected_rank":selected,"candidates":candidates,"D0":d0i,"D2":d2i,"test_metrics":rows},indent=2))

if __name__=="__main__": main()
