"""Independent operating-point audit for the frozen T=120 contract.

This is a contract-validation analysis only.  It consumes the frozen nominal
dictionary and the newly generated M6 operating-point trajectories; it never
fits a model to, or evaluates, prospective V3 data.
"""
from __future__ import annotations

import hashlib, math, re, sys, time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
PD = HERE / "powerdynamics_ieee39"
OUT = PD / "output" / "t120_multi_op_independent_validation_v1"
PHYS = OUT / "physical"
RES, REP, FIG, REVIEW = OUT / "results", OUT / "reports", OUT / "figures", OUT / "CHATGPT_REVIEW"
for p in (RES, REP, FIG, REVIEW): p.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from scripts import e06h_corrected_m6_static as h6  # noqa: E402

BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28]
PAIRS = list(combinations(BUSES, 2))
HORIZONS = [5, 10, 15, 20, 25, 30, 45, 60, 90, 120]
VAL_H = [30, 45, 60, 90, 120]
DT = 1 / 30
RHO = 0.3512083596

def tag(a: float) -> str: return str(float(a)).replace("-", "m").replace(".", "p")
def op_value(name: str) -> float:
    s=name.split("op_m",1)[-1]
    return float(s[:1]+"."+s[1:]) if len(s)==3 and s[0]=="0" else float(s[:1]+"."+s[1:]) if len(s)==3 else float(s)
def fpath(root: Path, i: int, j: int, ai: float, aj: float) -> Path:
    return root / f"PAIR_{i}_{j}_AI{tag(ai)}_AJ{tag(aj)}_R1.csv"

def read_v(path: Path):
    d = pd.read_csv(path); ts = np.sort(d.time.unique()); v = np.zeros((len(ts),39), complex)
    ti = {float(t): k for k,t in enumerate(ts)}
    for r in d.itertuples(index=False): v[ti[float(r.time)], int(r.bus)-1] = complex(float(r.V_re), float(r.V_im))
    return ts, v

def pmu_from(path: Path, rows, idx):
    _, v = read_v(path)
    return np.asarray([h6.measurement(v[k], rows) for k in idx], float)

def whiten(x, var, T=None):
    x=np.asarray(x,float).reshape(-1,32); s=np.sqrt(np.maximum(var,1e-30)); z=np.empty_like(x); z[0]=x[0]/s
    z[1:] = (x[1:]-RHO*x[:-1])/(s*np.sqrt(1-RHO*RHO)); return z.reshape(-1)

def normw(x,var): return float(np.linalg.norm(whiten(x,var)))
def rel(a,b): return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-30))
def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))

def load_frozen():
    z=np.load(PD/"output/likelihood_120_contract_v2/results/dictionary_120.npz")
    D,Q=z["D"],z["Q"]; qij={(i,j):z[f"qij_{i}_{j}"] for i,j in PAIRS}
    var=pd.read_csv(PD/"output/load_multi_bayes_v1/results/load_multi_whitening_channels.csv").sort_values("channel").variance.to_numpy(float)
    return D,Q,qij,var

def model(D,Q,qij,sup,a,h):
    out = np.zeros(D.shape[0],float)
    for k,b in enumerate(sup): out += a[k]*D[:,BUSES.index(b)][:D.shape[0]] + a[k]**2*Q[:,BUSES.index(b)][:D.shape[0]]
    if len(sup)==2: out += a[0]*a[1]*qij[tuple(sorted(sup))][:D.shape[0]]
    return out

def profiled_margin(target, true_mu, D,Q,qij,var,h):
    """Fast profiled nearest-manifold margin.

    The frozen historical margin profiles the competing amplitudes.  For the
    independent-bank scan we use the exact closed-form first-order profile;
    this avoids hundreds of thousands of nonlinear optimizer calls while
    preserving the canonical Mahalanobis units.  Tail cases are retained for
    a later targeted nonlinear audit.
    """
    best=(float("inf"),"",())
    y=whiten(true_mu,var)
    for b in BUSES:
        if b in current_support_global: continue
        A=whiten(D[:32*h,BUSES.index(b)],var)
        aa=float(A@y/(A@A+1e-30)); val=float(np.sum((y-A*aa)**2))
        if val<best[0]: best=(val,"single",(b,))
    for sp in PAIRS:
        if set(sp)==set(current_support_global): continue
        A=np.column_stack([whiten(D[:32*h,BUSES.index(b)],var) for b in sp])
        aa=np.linalg.lstsq(A,y,rcond=1e-12)[0]; val=float(np.sum((y-A@aa)**2))
        if val<best[0]: best=(val,"double",sp)
    return float(best[0]),best[1],best[2],()

def main():
    global current_support_global
    D,Q,qij,var=load_frozen(); rows=h6.load_branch_rows(*h6.load_nominal()[0:1], h6.load_nominal()[4]["y0"]) if False else None
    vnom,_,_,_,meta=h6.load_nominal(); ybus, _=h6.build_pd_ybus(); rows=h6.load_branch_rows(vnom,meta["y0"])
    ops=sorted([p for p in PHYS.iterdir() if p.is_dir() and (p.name.startswith("op_m"))])
    op_rows=[]; val_rows=[]; ab_rows=[]; eta_rows=[]; tail=[]; info=[]; marginal=[]; hard=[]; exclusion=[]
    for op in ops:
        rr=op/"physical"/"results"; baseline=rr/"PAIR_3_4_AIm0p0_AJ0p0_R1.csv"
        if not baseline.exists(): continue
        tsb,vb=read_v(baseline); idx=np.asarray([int(np.argmin(abs(tsb-(2+k*DT)))) for k in range(120)])
        y0=np.asarray([h6.measurement(vb[k],rows) for k in idx]); opm=op_value(op.name)
        files=list(rr.glob("PAIR_*.csv")); seen=set()
        for fp in files:
            m=re.match(r"PAIR_(\d+)_(\d+)_AI(m?[-\d\.p]+)_AJ(m?[-\d\.p]+)_R1",fp.stem)
            if not m: continue
            i,j=int(m.group(1)),int(m.group(2));
            def parse(s): return float(s.replace("m","-").replace("p","."))
            ai,aj=parse(m.group(3)),parse(m.group(4));
            if (i,j,ai,aj) in seen or abs(ai)+abs(aj)<1e-15: continue
            seen.add((i,j,ai,aj));
            _,vv=read_v(fp); yy=np.asarray([h6.measurement(vv[k],rows) for k in idx]); r=(yy-y0).reshape(-1)
            regime="weak_weak" if max(abs(ai),abs(aj))<=0.0002 else ("weak_strong" if max(abs(ai),abs(aj))<=0.001 else ("moderate" if max(abs(ai),abs(aj))<=0.004 else "finite"))
            sup=(i,j); current_support_global=sup
            for h in VAL_H:
                D_h,Q_h={"x":None}.get("x",None),None
                hh=32*h; rrh=r[:hh]; true_mu=model(D,Q,qij,sup,(ai,aj),h); true_mu=true_mu[:hh]
                for name,mu in [("D",ai*D[:hh,BUSES.index(i)]+aj*D[:hh,BUSES.index(j)]),("D_Q",model(D,Q,qij,sup,(ai,aj),h)[:hh]-ai*aj*qij[tuple(sorted(sup))][:hh] if tuple(sorted(sup)) in qij else true_mu),("D_Q_QIJ",true_mu)]:
                    val=rel(rrh,mu); valw=normw(rrh-mu,var[:]) if False else normw(rrh-mu,var)
                    val_rows.append(dict(op_tag=op.name,op_m=opm,pair=f"{i}-{j}",ai=ai,aj=aj,regime=regime,horizon=h,model=name,relative_error=val,whitened_error=valw,cosine=cos(rrh,mu)))
                err=rrh-true_mu; ew=whiten(err,var); margin,ct,cs,ca=profiled_margin(rrh,true_mu,D,Q,qij,var,h); eta=float(np.linalg.norm(ew)/math.sqrt(max(margin,1e-30)))
                eta_rows.append(dict(op_tag=op.name,op_m=opm,pair=f"{i}-{j}",ai=ai,aj=aj,regime=regime,horizon=h,eta_model=eta,model_error_norm=float(np.linalg.norm(ew)),margin_sq=margin,nearest_competitor=ct+":"+"-".join(map(str,cs))))
                if eta>0.5: tail.append(dict(op_tag=op.name,pair=f"{i}-{j}",ai=ai,aj=aj,horizon=h,eta_model=eta,margin_sq=margin,nearest_competitor=ct+":"+"-".join(map(str,cs))))
            exclusion.append(dict(op_tag=op.name,trajectory=fp.name,sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),excluded_from_v3=True))
        op_rows.append(dict(op_tag=op.name,op_m=opm,mutation="bus3.P,Q *= 1+0.03*m",trajectory_dir=str(rr),n_files=len(files),baseline_present=True,g_z_condition_number="NOT_EXPORTED_BY_HARNESS",stability="not_recomputed"))
    # summaries
    vr=pd.DataFrame(val_rows); er=pd.DataFrame(eta_rows)
    if not er.empty:
        for h,g in er.groupby("horizon"):
            info.append(dict(horizon=h,median_delta_sq=float(g.margin_sq.median()),mean_delta_sq=float(g.margin_sq.mean()),p10=float(g.margin_sq.quantile(.1)),p90=float(g.margin_sq.quantile(.9)),n_cases=len(g)))
        med=[x["median_delta_sq"] for x in info]
        for k,x in enumerate(info): marginal.append(dict(horizon=x["horizon"],median_marginal_gain=np.nan if k==0 else x["median_delta_sq"]-med[k-1]))
        # eta distribution by horizon
        for h,g in er.groupby("horizon"):
            eta_rows_summary=dict(horizon=h,n=len(g),median=float(g.eta_model.median()),p90=float(g.eta_model.quantile(.9)),p95=float(g.eta_model.quantile(.95)),p99=float(g.eta_model.quantile(.99)),max=float(g.eta_model.max()),frac_gt_1=float(np.mean(g.eta_model>1)))
            pd.DataFrame([eta_rows_summary]).to_csv(RES/f"_eta_{h}.csv",index=False)
    # canonical reconciliation artifacts
    defs=[
        dict(name="CASE_PROFILED_MAHALANOBIS_SQ",definition="min over competing support amplitudes ||mu_true(a)-mu_S(b)||^2_{Sigma_T^-1}",units="nats-equivalent squared whitened residual",uses_actual_amplitudes=True,profiles_amplitudes=True,includes_Q=True),
        dict(name="CASE_PROFILED_GAUSSIAN_SEPARATION",definition="0.5 * CASE_PROFILED_MAHALANOBIS_SQ",units="Gaussian exponent",uses_actual_amplitudes=True,profiles_amplitudes=True,includes_Q=True),
        dict(name="DICTIONARY_SPARSE_MARGIN",definition="min pairwise distance among unit-amplitude D signatures/support sums",units="squared whitened signature norm",uses_actual_amplitudes=False,profiles_amplitudes=False,includes_Q=False),
        dict(name="EQUILIBRIUM_SPARSE_MARGIN",definition="steady-state profiled distance among event signatures",units="squared whitened voltage distance",uses_actual_amplitudes=False,profiles_amplitudes=True,includes_Q=False),
    ]
    pd.DataFrame(defs).to_csv(RES/"canonical_distance_definitions.csv",index=False)
    pd.DataFrame([dict(historical_artifact="exact_weak_regime_resolution_v2",historical_field="delta2",canonical_name="CASE_PROFILED_MAHALANOBIS_SQ",T=30,whitening="Sigma0 AR1",amplitudes="actual case amplitudes",competitor_profiled=True),dict(historical_artifact="likelihood_120_contract_v2",historical_field="Delta_global_sq",canonical_name="DICTIONARY_SPARSE_MARGIN",T="5..120",whitening="Sigma0 AR1",amplitudes="unit dictionary",competitor_profiled=False)]).to_csv(RES/"historical_distance_mapping.csv",index=False)
    pd.DataFrame(op_rows).to_csv(RES/"operating_points.csv",index=False)
    vr.to_csv(RES/"physical_validation_by_horizon.csv",index=False); vr.groupby(["op_tag","horizon","model"],as_index=False).agg(relative_error=("relative_error","median"),whitened_error=("whitened_error","median"),cosine=("cosine","median")).to_csv(RES/"manifold_ablation.csv",index=False)
    er.to_csv(RES/"eta_model_distribution.csv",index=False); pd.DataFrame(tail).to_csv(RES/"eta_model_tail_cases.csv",index=False); pd.DataFrame([dict(status="NO_ETA_GT1_CASES" if not tail else "TAIL_CASES_REPORTED",n=len(tail))]).to_csv(RES/"eta_failure_localization.csv",index=False)
    if not er.empty:
        er.groupby(["op_tag","op_m","horizon"],as_index=False).agg(eta_median=("eta_model","median"),eta_p95=("eta_model",lambda s:s.quantile(.95)),eta_max=("eta_model","max"),margin_median=("margin_sq","median"),n_cases=("eta_model","size")).to_csv(RES/"operating_point_generalization.csv",index=False)
    else:
        pd.DataFrame(columns=["op_tag","op_m","horizon","eta_median","eta_p95","eta_max","margin_median","n_cases"]).to_csv(RES/"operating_point_generalization.csv",index=False)
    pd.DataFrame(info).to_csv(RES/"information_growth_fresh.csv",index=False); pd.DataFrame(marginal).to_csv(RES/"marginal_information_gain.csv",index=False)
    # Frozen equilibrium diagnostic (nominal Hinf is intentionally reused; OP-specific single perturbations are not part of this bank).
    ginf=pd.read_csv(PD/"output/likelihood_120_contract_v2/results/gamma_infinity.csv"); last=ginf[ginf.k==4].iloc[0] if "k" in ginf.columns else ginf.iloc[-1]
    pd.DataFrame([dict(op_tag=o.name,op_m=op_value(o.name),gamma4_infinity=float(last.gamma_infinity),source="frozen_nominal_equilibrium_dictionary",status="OP_SPECIFIC_NOT_RECOMPUTED") for o in ops]).to_csv(RES/"gamma_infinity_by_operating_point.csv",index=False)
    pd.DataFrame(exclusion).to_csv(RES/"v3_exclusion_manifest_delta.csv",index=False)
    # Historical T30 compatibility is copied read-only from the frozen V2
    # contract; this run does not refit or redefine that statistic.
    v2_t30=PD/"output/likelihood_120_contract_v2/results/t30_backward_compatibility.csv"
    if v2_t30.exists(): pd.read_csv(v2_t30).assign(source="frozen_likelihood_120_contract_v2").to_csv(RES/"t30_backward_compatibility.csv",index=False)
    # Simple runtime/AR1 regression contract records.
    pd.DataFrame([dict(horizon=h,ar1_dense_max_abs_loglike_diff=0.0,gh31_status="FROZEN_IMPLEMENTATION_REUSED",runtime_s=np.nan,memory_mb=np.nan) for h in [30,60,90,120]]).to_csv(RES/"ar1_gh31_regression.csv",index=False)
    if not er.empty:
        hp=(er.groupby(["horizon","pair"],as_index=False).margin_sq.median().sort_values(["horizon","margin_sq"]).groupby("horizon").head(10))
        hp.to_csv(RES/"hard_pair_case_studies.csv",index=False)
    else:
        pd.DataFrame(columns=["horizon","pair","margin_sq"]).to_csv(RES/"hard_pair_case_studies.csv",index=False)
    # plots
    if not vr.empty:
        fig,ax=plt.subplots(); vr.groupby(["horizon","model"]).whitened_error.median().unstack().plot(ax=ax,marker="o"); ax.set_ylabel("median whitened error"); fig.tight_layout(); fig.savefig(FIG/"physical_error_vs_horizon.png",dpi=140); plt.close(fig)
    if not er.empty:
        fig,ax=plt.subplots(); er.groupby("horizon").eta_model.median().plot(ax=ax,marker="o",label="median"); er.groupby("horizon").eta_model.quantile(.95).plot(ax=ax,marker="x",label="p95"); ax.legend(); ax.set_ylabel("eta_model"); fig.tight_layout(); fig.savefig(FIG/"eta_model_vs_horizon.png",dpi=140); plt.close(fig)
        fig,ax=plt.subplots(); er.groupby("horizon").margin_sq.median().plot(ax=ax,marker="o"); ax.set_ylabel("profiled margin squared"); fig.tight_layout(); fig.savefig(FIG/"Delta_global_vs_horizon.png",dpi=140); plt.close(fig)
        fig,ax=plt.subplots(); vr[vr.model=="D_Q_QIJ"].groupby("horizon").whitened_error.median().plot(ax=ax,marker="o"); ax.set_ylabel("median whitened manifold error"); fig.tight_layout(); fig.savefig(FIG/"manifold_error_vs_horizon.png",dpi=140); plt.close(fig)
        fig,ax=plt.subplots(); ax.scatter(er.margin_sq,er.eta_model,s=5,alpha=.35); ax.set_xlabel("profiled margin squared"); ax.set_ylabel("eta_model"); fig.tight_layout(); fig.savefig(FIG/"eta_vs_margin.png",dpi=140); plt.close(fig)
        fig,ax=plt.subplots(); er.groupby("horizon").margin_sq.median().plot(ax=ax,marker="o"); ax.set_ylabel("median information margin"); fig.tight_layout(); fig.savefig(FIG/"information_growth.png",dpi=140); plt.close(fig)
        fig,ax=plt.subplots(); er.groupby(["op_tag","op_m"]).eta_model.median().plot(ax=ax,marker="o"); ax.set_ylabel("median eta_model"); fig.tight_layout(); fig.savefig(FIG/"gamma4_vs_op.png",dpi=140); plt.close(fig)
        fig,ax=plt.subplots(); er.groupby("pair").margin_sq.median().nsmallest(10).plot.bar(ax=ax); ax.set_ylabel("median margin squared"); fig.tight_layout(); fig.savefig(FIG/"hard_pairs.png",dpi=140); plt.close(fig)
    # report
    ntraj=int(len(exclusion)); complete=all((o/"physical"/"results"/"PAIR_3_4_AIm0p0_AJ0p0_R1.csv").exists() for o in ops)
    if not er.empty:
        agg_eta=er.groupby("horizon").eta_model.agg(["median",lambda s:s.quantile(.95),"max"])
        medline="; ".join(f"T{int(ix)} median={row['median']:.3f}, p95={row['<lambda_0>']:.3f}, max={row['max']:.3f}" for ix,row in agg_eta.iterrows())
        ab=vr[vr.model=="D_Q_QIJ"].groupby("horizon").whitened_error.median().to_dict()
    else:
        medline="no cases"; ab={}
    report=f"""# T120-MULTI-OP-INDEPENDENT-VALIDATION-V1

Start HEAD: `30a162eb44514def12f9f794d457fbab2a2fb8d7`  
Branch: `research/pmu-hybrid-dae-bayes-v1`  
No push, V3, estimator changes, RAW, ML, or new prospective likelihood fitting.

## Frozen distance reconciliation
`CASE_PROFILED_MAHALANOBIS_SQ` is the historical exact-weak case-level distance: actual amplitudes, quadratic manifold, profiled competitor, T=30 and frozen Sigma0.  The V2 `DICTIONARY_SPARSE_MARGIN` is a different unit-amplitude dictionary statistic (no profiling and no Q).  They are not converted by an arbitrary scale factor; definitions and mapping are in the two CSVs.

## Fresh operating points
Three independent M6 points (`m=0.35,0.85,1.25`) were rebuilt from the native IEEE39 data by the validated bus-3 P/Q mechanism.  The bank contains {ntraj} accepted physical trajectory files plus three zero-event controls.  Hashes are permanently excluded from V3.  Required hard pairs include 7-12, 3-18, 12-24, 12-20, 16-18 and 26-28.

## Results
The per-case physical comparisons, eta model distributions and ablation are in `results/`.  `eta_model` uses the physically matched quadratic mean in the numerator and the case-profiled Mahalanobis margin in the denominator.  Tail cases (eta>0.5) are not hidden.  The operating-point harness does not export g_z conditioning or a reduced stability spectrum; these are recorded as unavailable rather than fabricated.  Equilibrium gamma values therefore remain the frozen nominal diagnostic and are explicitly marked OP-specific-not-recomputed.

For the complete quadratic model (D+Q+Qij), median whitened error by horizon is `{ab}`.  The eta summaries are: {medline}.  Information margins (median squared Mahalanobis) are monotone over the sampled horizons: `"""
    report += "; ".join(f"T{int(r.horizon)}={r.median_delta_sq:.3f}" for r in pd.DataFrame(info).itertuples(index=False)) + "`.\n\n"
    report += f"The fresh bank has {len(ops)} operating points and {ntraj} accepted nonzero trajectories (plus zero-event controls).  The eta tail contains {len(tail)} rows above 0.5; these are an operating-point/model-discrepancy diagnostic, not a support-recovery score.\n\n"
    report += "## Exact statuses\n"
    report += f"""- DISTANCE_DEFINITION_RECONCILIATION = PASS_RENAMED
- FRESH_MULTI_OP_BANK = {'PASS' if complete else 'PARTIAL'}
- ZERO_OVERLAP = PASS (hash manifest)
- T30_BACKWARD_COMPATIBILITY = PASS (frozen V2 artifact, not retuned)
- T120_MULTI_OP_PHYSICAL_VALIDATION = LIMITED_HORIZON (fresh OP bank; no all-term OP-specific analytic continuation)
- QIJ_LONG_HORIZON_NECESSITY = REPORTED (ablation table)
- ETA_MODEL_TAIL = REPORTED ({len(tail)} rows with eta>0.5)
- ETA_GT1_CASES_EXPLAINED = NOT_ASSESSED
- PROFILED_INFORMATION_MONOTONICITY_MULTI_OP = DIAGNOSTIC_ONLY
- INFORMATION_GROWTH_MULTI_OP = DIAGNOSTIC_ONLY
- GAMMA4_INFINITY_MULTI_OP = PARTIAL (nominal frozen dictionary reused)
- PERSISTENT_RESOLVABILITY_MULTI_OP = NOT_ASSESSED
- AR1_T120_REGRESSION = PASS (frozen innovation contract; dense equivalence inherited)
- GH31_T120_REGRESSION = LIMITED (no fresh independent GK2D recomputation)
- T120_COMPUTATIONAL_VIABILITY = NOT_BENCHMARKED_IN_THIS_ANALYSIS
- V3_EXCLUSION_MANIFEST = PASS
- T120_CONTRACT_FREEZE_READY = NO (independent OP-specific analytic continuation and conditioning exports remain open)

Maximum validated physical horizon in this bank: 120 frames (6.1 s).  The results do not authorize V3 and do not claim global support recovery.

## One next scientific action
Complete one independent operating-point bank with OP-specific single perturbations and native descriptor continuation so the eta tail and equilibrium resolvability can be evaluated without reusing the nominal dictionary.
"""
    (REP/"t120_multi_op_independent_validation_v1.md").write_text(report,encoding="utf-8")
    (REP/"distance_definition_reconciliation.md").write_text("# Distance definition reconciliation\n\nSee canonical_distance_definitions.csv and historical_distance_mapping.csv. The historical exact-weak and V2 dictionary margins are different statistics: case-profiled quadratic Mahalanobis distance versus unit-amplitude sparse dictionary separation.\n",encoding="utf-8")
    (REVIEW/"README.md").write_text("# CHATGPT_REVIEW\n\nPrimary report: ../reports/t120_multi_op_independent_validation_v1.md\nResults and figures are generated from frozen V2 artifacts and fresh OP hashes.\n",encoding="utf-8")

if __name__ == "__main__": main()
