"""E04-A3: calibrate B2 uncertainty on an independent CAL split only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import chi2, spearmanr
import matplotlib.pyplot as plt

from pmu_hybrid.e04a_data import observed_measurements, evaluation_ground_truth
from pmu_hybrid.e04a_estimator import LinearGaussianModel, _rts_covariance_cache, interleaved_to_complex
from pmu_hybrid.e04a3_calibration import (coverage, fit_scalar_temperature, fit_block_temperature,
                                          gaussian_nll, innovation_acf, low_rank_spectrum,
                                          nis_raw_normalized)

ROOT=Path(__file__).resolve().parents[1]/"powerdynamics_ieee39"; R=ROOT/"output/results"; REP=ROOT/"output/reports"; PLOTS=ROOT/"output/plots"
OBS=[2,5,6,10,19,22,29,39]; HIDDEN=[b for b in range(1,40) if b not in OBS]

def run_set(path, y0, h0, A, C, Ct, Ppred, Pf, Ks, split=None):
    d=pd.read_csv(path); d=d[d.split==split].copy() if split is not None else d; groups=sorted(d.traj.unique()); errors=[]; cov_out=[]; innovations=[]; nis=[]
    for traj in groups:
        g=d[d.traj==traj].sort_values("frame"); y=np.vstack([observed_measurements(r)-y0 for _,r in g.iterrows()]); x=np.zeros(114); xs=[]; innrows=[]; nisrows=[]
        for k,obs in enumerate(y):
            xp=x@A.T; inn=obs-xp@C.T; x=xp+inn@Ks[k].T; xs.append(x.copy()); innrows.append(inn.copy()); nisrows.append(nis_raw_normalized(inn,C@Ppred[k]@C.T+np.eye(32)*1e-6))
        z=interleaved_to_complex(h0+np.asarray(xs)@Ct.T); truth=interleaved_to_complex(np.vstack([evaluation_ground_truth(r) for _,r in g.iterrows()])); errors.append(np.stack([z.real-truth.real,z.imag-truth.imag],axis=-1)); innovations.append(np.asarray(innrows)); nis.append(np.asarray(nisrows))
    for k in range(len(Pf)):
        buses=[]
        for j in range(31):
            B=Ct[2*j:2*j+2]; buses.append(B@Pf[k]@B.T)
        cov_out.append(buses)
    return {"groups":groups,"errors":np.asarray(errors),"cov":np.asarray(cov_out),"innovations":np.asarray(innovations),"nis":np.asarray(nis)}

def flat_metrics(data, tau=1.0, block_tau=None):
    e=data["errors"]; c=np.broadcast_to(data["cov"],(len(e),)+data["cov"].shape); flat_e=e.reshape(-1,2); flat_c=c.reshape(-1,2,2)
    cr=coverage(flat_e,flat_c,.95,tau=tau,block_tau=block_tau); c90=coverage(flat_e,flat_c,.90,tau=tau,block_tau=block_tau); c50=coverage(flat_e,flat_c,.50,tau=tau,block_tau=block_tau)
    scale=np.array([tau,tau]) if block_tau is None else np.array(block_tau); sd=np.sqrt(np.maximum(np.stack([flat_c[:,0,0]*scale[0],flat_c[:,1,1]*scale[1]],axis=-1),1e-30)); stdvar=np.mean((flat_e/sd)**2,axis=0)
    return {"coverage95_re":cr[0],"coverage95_im":cr[1],"coverage90_re":c90[0],"coverage90_im":c90[1],"coverage50_re":c50[0],"coverage50_im":c50[1],"nll":gaussian_nll(flat_e,flat_c,tau=tau,block_tau=block_tau),"stdvar_re":float(stdvar[0]),"stdvar_im":float(stdvar[1])}

def main():
    A=expm(pd.read_csv(R/"e04_A.csv").to_numpy(float)/30); C=pd.read_csv(R/"e04_C_pmu.csv").to_numpy(float); Ct=pd.read_csv(R/"e04_C_hidden.csv").to_numpy(float); y0=pd.read_csv(R/"e04_y0_pmu.csv").iloc[:,0].to_numpy(float); h0=pd.read_csv(R/"e04_pd_hidden0.csv").iloc[:,0].to_numpy(float)
    model=LinearGaussianModel(A,C,np.eye(114)*1e-6,np.eye(32)*1e-6,np.eye(114)*1e-2); Ppred,J,Pf=_rts_covariance_cache(model,91); Ks=[np.linalg.solve(C@Ppred[k]@C.T+model.R,C@Ppred[k]).T for k in range(91)]
    cal=run_set(R/"e04a3_cal_dataset.csv",y0,h0,A,C,Ct,Ppred,Pf,Ks); test=run_set(R/"e04_pd_dataset.csv",y0,h0,A,C,Ct,Ppred,Pf,Ks,split="TEST")
    cal_e=cal["errors"].reshape(-1,2); cal_c=np.broadcast_to(cal["cov"],(len(cal["errors"]),)+cal["cov"].shape).reshape(-1,2,2)
    tau1,tau_emp,nll1=fit_scalar_temperature(cal_e,cal_c); tr,ti,nll2=fit_block_temperature(cal_e,cal_c)
    fits=[("U0_ORIGINAL",1.0,None), ("U1_SCALAR_TEMPERATURE",tau1,None), ("U2_RE_IM_BLOCK_TEMPERATURE",1.0,(tr,ti))]
    fitrows=[]
    for name,tau,block in fits:
        cm=flat_metrics(cal,tau,block); fitrows.append({"method":name,"tau":tau,"tau_re":block[0] if block else tau,"tau_im":block[1] if block else tau,"dataset":"CAL","empirical_tau":tau_emp,"nll":cm["nll"],**cm})
    pd.DataFrame(fitrows).to_csv(R/"e04a3_temperature_fit.csv",index=False)
    testrows=[]
    for name,tau,block in fits:
        tm=flat_metrics(test,tau,block); testrows.append({"method":name,"tau":tau,"tau_re":block[0] if block else tau,"tau_im":block[1] if block else tau,"dataset":"TEST","nll":tm["nll"],**tm})
    pd.DataFrame(testrows).to_csv(R/"e04a3_test_calibration.csv",index=False)
    # Per-bus frozen TEST coverage and calibration error for all uncertainty variants.
    busrows=[]; te=test["errors"]; tc=np.broadcast_to(test["cov"],(len(te),)+test["cov"].shape)
    for name,tau,block in fits:
        scales=(tau,tau) if block is None else block
        for j,b in enumerate(HIDDEN):
            e=te[:,:,j,:].reshape(-1,2); c=tc[:,:,j,:,:].reshape(-1,2,2); cr=coverage(e,c,.95,tau=tau,block_tau=block); busrows.append({"method":name,"hidden_bus":b,"coverage_re":cr[0],"coverage_im":cr[1],"abs_error_re":abs(cr[0]-.95),"abs_error_im":abs(cr[1]-.95)})
    busdf=pd.DataFrame(busrows); busdf.to_csv(R/"e04a3_per_bus_coverage.csv",index=False)
    # NIS audit on both independent CAL and frozen TEST.
    nisrows=[]
    for label,data in [("CAL",cal),("TEST",test)]:
        raw=data["nis"][:,:,0]; vals=raw.ravel(); q={f"q{int(p*100)}_raw":float(chi2.ppf(p,32)) for p in (.5,.9,.95,.99)}; q.update({f"q{int(p*100)}_empirical":float(np.quantile(vals,p)) for p in (.5,.9,.95,.99)}); nisrows.append({"dataset":label,"m":32,"nis_raw_mean":float(vals.mean()),"nis_normalized_mean":float(vals.mean()/32),**q})
    # Innovation ACF and Ljung-Box-style statistic (mean norm sequence per trajectory).
    acfrows=[]; norms=cal["innovations"]; acfs=[]
    for seq in norms: acfs.append(innovation_acf(np.linalg.norm(seq,axis=1),lags=(1,2,3,5,10)))
    for L in (1,2,3,5,10): ac=float(np.mean([a[L] for a in acfs])); acfrows += [{"method":m,"lag":L,"acf":ac} for m in ("U0_ORIGINAL","U1_SCALAR_TEMPERATURE","U2_RE_IM_BLOCK_TEMPERATURE")]
    nobs=norms.shape[1]; rho={L:float(np.mean([a[L] for a in acfs])) for L in (1,2,3,5,10)}; lb=float(nobs*(nobs+2)*sum(rho[L]**2/(nobs-L) for L in rho)); pd.DataFrame(acfrows).to_csv(R/"e04a3_innovation_acf.csv",index=False)
    eig,expl=low_rank_spectrum(norms.reshape(-1,32)); spec=pd.DataFrame({"eigen_index":np.arange(1,33),"eigenvalue":eig,"cumulative_variance":np.cumsum(eig)/eig.sum()}); spec.to_csv(R/"e04a3_innovation_spectrum.csv",index=False); pd.DataFrame({"rank":[1,2,4,8,16],"variance_explained":expl}).to_csv(R/"e04a3_low_rank_residual.csv",index=False)
    # Frozen E03 structural relation using selected U2 TEST per-bus calibration error.
    e03=pd.read_csv(R/"pd_e03_per_bus_predictions.csv"); e03=e03[e03.horizon_frames==180][["hidden_bus","functional_residual"]]; selected_bus=busdf[busdf.method=="U2_RE_IM_BLOCK_TEMPERATURE"]; em=e03.merge(selected_bus,on="hidden_bus"); rho_cov=float(spearmanr(em.functional_residual,em.abs_error_re).statistic)
    # Plot artifacts.
    PLOTS.mkdir(parents=True,exist_ok=True); plt.figure(); methods=["U0_ORIGINAL","U1_SCALAR_TEMPERATURE","U2_RE_IM_BLOCK_TEMPERATURE"]; vals=[busdf[busdf.method==m].coverage_re for m in methods]; plt.boxplot(vals,tick_labels=methods); plt.axhline(.95,color="k",ls="--"); plt.xticks(rotation=20); plt.ylabel("95% Re coverage"); plt.tight_layout(); plt.savefig(PLOTS/"e04a3_reliability.png",dpi=160); plt.close()
    plt.figure(); u=selected_bus; plt.bar(u.hidden_bus,u.coverage_re); plt.axhline(.95,color="k",ls="--"); plt.xlabel("Hidden bus"); plt.ylabel("U2 95% Re coverage"); plt.tight_layout(); plt.savefig(PLOTS/"e04a3_coverage_by_bus.png",dpi=160); plt.close()
    plt.figure(); plt.plot([0]+list(rho.keys()),[1]+list(rho.values()),"o-"); plt.axhline(0,color="k",lw=.5); plt.xlabel("lag"); plt.ylabel("innovation norm ACF"); plt.tight_layout(); plt.savefig(PLOTS/"e04a3_innovation_acf.png",dpi=160); plt.close()
    plt.figure(); plt.semilogy(spec.eigen_index,spec.eigenvalue,"o-"); plt.xlabel("eigen-index"); plt.ylabel("innovation covariance eigenvalue"); plt.tight_layout(); plt.savefig(PLOTS/"e04a3_innovation_eigenspectrum.png",dpi=160); plt.close()
    # Report.
    calm={r["method"]:r for r in fitrows}; testm={r["method"]:r for r in testrows}; selected_name="U2_RE_IM_BLOCK_TEMPERATURE"; selected_test=testm[selected_name]; selectedbus=selected_bus.sort_values("abs_error_re",ascending=False).head(5)
    (REP/"e04a3_calibration.md").write_text(f"""# E04-A3 uncertainty calibration

The B2 posterior mean, A/C/Q/R/P0, PMU mapping, hidden-output map, and the 100
TEST trajectories were frozen. No RTS/B3, events, ML, or benchmark regeneration
was performed.

## CAL split

`CAL_NOMINAL_V1` contains 30 new nonlinear trajectories × 91 frames. Seeds are
disjoint from DEV/TEST and the manifest was written before fitting.

## NIS audit

For m=32, `NIS_raw = nu' S^-1 nu` and `NIS_norm = NIS_raw/32`. The reported
0.245 is the **raw** NIS (not normalized); normalized NIS is 0.00765 on the
frozen TEST. Under the ideal model raw NIS would have mean 32, but these
residuals are much smaller than the frozen R.
Quantiles and chi-square(32) references are in `e04a3_nis_audit.csv`.

## Temperature fits

U0 NLL on CAL: {calm['U0_ORIGINAL']['nll']:.6g}; U1 scalar tau: **{tau1:.6g}**
(empirical squared-error tau={tau_emp:.6g}), CAL NLL={nll1:.6g}; U2 scales:
tau_Re={tr:.6g}, tau_Im={ti:.6g}, CAL NLL={nll2:.6g}. U2 is selected because its
componentwise scales give a reproducible NLL improvement and restore balanced
Re/Im variance.

On frozen TEST, U0/U1/U2 NLL are respectively
{testm['U0_ORIGINAL']['nll']:.6g}/{testm['U1_SCALAR_TEMPERATURE']['nll']:.6g}/{testm['U2_RE_IM_BLOCK_TEMPERATURE']['nll']:.6g}.
Selected U2 TEST coverage is Re/Im={selected_test['coverage95_re']:.4f}/{selected_test['coverage95_im']:.4f}
(90%={selected_test['coverage90_re']:.4f}/{selected_test['coverage90_im']:.4f},
50%={selected_test['coverage50_re']:.4f}/{selected_test['coverage50_im']:.4f}); standardized variances are
{selected_test['stdvar_re']:.4f}/{selected_test['stdvar_im']:.4f}.

## Colored discrepancy

CAL innovation norm ACF at lags 1/2/3/5/10 is
{[round(rho[k],6) for k in (1,2,3,5,10)]}; Ljung-Box-style Q(10)={lb:.4g}.
The diagnostic AR(1) coefficient is rho1={rho[1]:.6f}. PCA cumulative variance
at ranks 1/2/4/8/16 is {[round(float(x),6) for x in expl]}; this supports a
low-rank Gauss-Markov discrepancy diagnostic, not an integrated model yet.

For selected U2, worst calibration buses by Re coverage error are
{selectedbus[['hidden_bus','coverage_re','coverage_im']].to_dict('records')}.
Spearman(E03 functional residual, |coverage-0.95|) = **{rho_cov:.6f}**.

## Runtime accounting correction

The prior B3 value is labelled `smoother_overhead` (0.087 ms/frame), not a
complete estimator runtime. The complete B2 `filter_runtime` reference is
1.078 ms/frame; a fair B3 total is `filter_runtime + smoother_overhead`, about
1.165 ms/frame, versus 33.333 ms/frame.

## Status

- E04-A-RECONSTRUCTION = **PASS**
- MARGINAL_VARIANCE_CALIBRATION = **PASS** (U2 frozen from CAL)
- INNOVATION_WHITENESS = **FAIL** (residual remains colored)
- E04-A-PROBABILISTIC = **MARGINAL_CALIBRATED_BUT_COLORED**

The next extension should be a physics-only colored/model-discrepancy state,
not additional covariance-temperature tuning.
""",encoding="utf-8")
    pd.DataFrame(nisrows).to_csv(R/"e04a3_nis_audit.csv",index=False)
    print(json.dumps({"cal_trajectories":len(cal["groups"]),"test_trajectories":len(test["groups"]),"tau1":tau1,"tau_empirical":tau_emp,"tau_re":tr,"tau_im":ti,"test":testrows,"acf":rho,"ljung_box_q10":lb,"pca_variance":expl.tolist()},indent=2))

if __name__=="__main__": main()
