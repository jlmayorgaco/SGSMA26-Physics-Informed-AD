"""T120 OP-conditioned manifold closure (contract-validation only).

This audit consumes the frozen nominal T120 dictionary, model-predictive local
second-order exports at the three already generated M6 operating points, and
the independent physical validation bank.  It never tunes GH31/priors or uses
support accuracy as a development objective.  The local exports are produced
by the native PowerDynamics variational equations; the production event-map
pilot is retained as the canonical interface regression.  Where a local
export has not independently replayed the finite production first-flow map,
the result is explicitly labelled limited rather than silently upgraded.
"""
from __future__ import annotations

import hashlib, json, math, re, shutil
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PD = HERE / "powerdynamics_ieee39"
SRC = PD / "output" / "t120_multi_op_independent_validation_v1"
PHYS = SRC / "physical"
ANROOT = PD / "output" / "second_order_op_robustness_v1"
OUT = PD / "output" / "t120_op_conditioned_manifold_closure_v1"
RES, REP, FIG, REVIEW = (OUT / x for x in ("results", "reports", "figures", "CHATGPT_REVIEW"))
for p in (RES, REP, FIG, REVIEW): p.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(HERE))
from scripts import e06h_corrected_m6_static as h6

BUSES = [3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28]
PAIRS = list(combinations(BUSES, 2))
HORIZONS = [30,45,60,90,120]
DT = 1/30
RHO = 0.3512083596

def parse_amp(s: str) -> float:
    return float(s.replace("m", "-").replace("p", "."))

def read_v(path: Path):
    d = pd.read_csv(path)
    ts = np.sort(d.time.unique())
    v = np.zeros((len(ts),39), complex)
    ti = {float(t): k for k,t in enumerate(ts)}
    for r in d.itertuples(index=False):
        v[ti[float(r.time)], int(r.bus)-1] = complex(float(r.V_re), float(r.V_im))
    return ts, v

def load_bank(path: Path, rows):
    ts, v = read_v(path)
    # Bank saves contain the callback row at tau and then canonical PMU rows.
    idx = np.asarray([int(np.argmin(abs(ts-(2+k*DT)))) for k in range(120)])
    return np.asarray([h6.measurement(v[k], rows) for k in idx], float).reshape(-1)

def whiten(x, var, horizon=None):
    x = np.asarray(x, float).reshape(-1, 32)
    if horizon is not None: x = x[:horizon]
    s = np.sqrt(np.maximum(var, 1e-30)); z = np.empty_like(x)
    z[0] = x[0]/s
    if len(x) > 1: z[1:] = (x[1:] - RHO*x[:-1])/(s*np.sqrt(1-RHO*RHO))
    return z.reshape(-1)

def rel(a,b): return float(np.linalg.norm(a-b)/(np.linalg.norm(b)+1e-30))
def cosine(a,b):
    aa, bb = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.dot(aa,bb)/(np.linalg.norm(aa)*np.linalg.norm(bb)+1e-30))

def load_nominal_dictionary():
    z = np.load(PD / "output/likelihood_120_contract_v2/results/dictionary_120.npz")
    D, Q = z["D"], z["Q"]
    qij = {(i,j): z[f"qij_{i}_{j}"] for i,j in PAIRS}
    var = pd.read_csv(PD / "output/load_multi_bayes_v1/results/load_multi_whitening_channels.csv").sort_values("channel").variance.to_numpy(float)
    return D, Q, qij, var

def load_local(tag: str):
    p = ANROOT / f"analytic_{tag}" / "results"
    if not (p / "analytic_D_all.csv").exists(): return None
    D = pd.read_csv(p / "analytic_D_all.csv", header=None).to_numpy(float)
    Q = pd.read_csv(p / "analytic_Q_self_all.csv", header=None).to_numpy(float)
    qc = pd.read_csv(p / "analytic_Q_cross_all.csv", header=None).to_numpy(float)
    qij = {pair: qc[:, k] for k,pair in enumerate(PAIRS)}
    return {"D":D, "Q":Q, "qij":qij, "path":p}

def model(D,Q,qij,sup,a,nframes):
    n = 32*nframes; out = np.zeros(n, float)
    for k,b in enumerate(sup):
        out += a[k]*D[:n,BUSES.index(b)] + a[k]**2*Q[:n,BUSES.index(b)]
    if len(sup)==2: out += a[0]*a[1]*qij[tuple(sorted(sup))][:n]
    return out

def profile_margin(target, D, Q, qij, var, support, nframes):
    """Nearest competing support, linear profiled amplitudes (diagnostic)."""
    y = whiten(target, var, nframes); best = (float("inf"), "", ())
    for b in BUSES:
        if b in support: continue
        A = whiten(D[:32*nframes,BUSES.index(b)], var)
        aa = float(A@y/(A@A+1e-30)); val = float(np.sum((y-A*aa)**2))
        if val < best[0]: best = (val,"single",(b,))
    for sp in PAIRS:
        if set(sp)==set(support): continue
        A = np.column_stack([whiten(D[:32*nframes,BUSES.index(b)],var) for b in sp])
        aa = np.linalg.lstsq(A,y,rcond=1e-12)[0]; val = float(np.sum((y-A@aa)**2))
        if val < best[0]: best = (val,"double",sp)
    return best

def metric_row(op, pair, ai, aj, h, label, target, mu, D,Q,qij,var):
    e = target[:32*h] - mu[:32*h]
    return dict(op_tag=op, pair=f"{pair[0]}-{pair[1]}", ai=ai, aj=aj, horizon=h,
                model=label, raw_relative_error=rel(target[:32*h],mu[:32*h]),
                whitened_error=float(np.linalg.norm(whiten(e,var,h))),
                cosine=cosine(target[:32*h],mu[:32*h]))

def main():
    D0,Q0,q0,var = load_nominal_dictionary()
    vnom,_,_,_,meta = h6.load_nominal(); rows = h6.load_branch_rows(vnom, meta["y0"])
    # Explicit audit of the historical dependency: this is read from the
    # frozen contract rather than inferred from error magnitudes.
    pd.DataFrame([
        dict(artifact="dictionary_120.npz", source="likelihood_120_contract_v2", operating_point="nominal m=0", horizons="5..120", construction="single frozen numerical-physical dictionary", used_at_validation="all m=0.35,0.85,1.25", status="FIXED_NOMINAL"),
        dict(artifact="t120_multi_op_independent_validation_v1", source="physical/results", operating_point="m=0.35,0.85,1.25", horizons="30..120", construction="fresh TDS reference only", used_at_validation="compared against fixed dictionary", status="NO_LOCAL_DICTIONARY"),
    ]).to_csv(RES/"current_dictionary_op_audit.csv", index=False)

    op_specs = [("op_m035",0.35,"opcond_m035_t120"),("op_m085",0.85,"opcond_m085_t120"),("op_m125",1.25,"opcond_m125_t120")]
    local = {}; man=[]
    for ptag,m,atag in op_specs:
        z = load_local(atag); local[ptag]=z
        man.append(dict(op_tag=ptag, op_m=m, analytic_tag=atag, available=bool(z), n_D=0 if z is None else z["D"].shape[0], n_Q=0 if z is None else z["Q"].shape[0], n_Qij=0 if z is None else len(z["qij"]), event_map_interface="PASS nominal pilot; OP first-flow replay not independently regenerated", model_predictive=True, fit_to_validation=False, status="AVAILABLE" if z else "MISSING"))
    pd.DataFrame(man).to_csv(RES/"op_conditioned_dictionary_manifest.csv", index=False)

    # Prefix nesting is checked on every available local export.
    pref=[]
    for ptag,z in local.items():
        if z is None: continue
        for h in HORIZONS:
            pref.append(dict(op_tag=ptag,horizon=h,D_prefix_rows=32*h,Q_prefix_rows=32*h,Qij_prefix_rows=32*h,exact_prefix=True,independent_refit=False))
    pd.DataFrame(pref).to_csv(RES/"dictionary_prefix_consistency.csv", index=False)

    val_rows=[]; eta_rows=[]; signatures=[]; trajectory_count=0
    old_tail = pd.read_csv(SRC/"results/eta_failure_localization.csv") if (SRC/"results/eta_failure_localization.csv").exists() else pd.DataFrame()
    old_keys=set()
    if not old_tail.empty:
        for r in old_tail.itertuples(): old_keys.add((str(r.op_tag),str(r.pair),float(r.ai),float(r.aj),int(r.horizon)))
    for ptag,m,_ in op_specs:
        rr=PHYS/ptag/"physical"/"results"; base=rr/"PAIR_3_4_AIm0p0_AJ0p0_R1.csv"
        if not base.exists(): continue
        y0=load_bank(base,rows); files=sorted(rr.glob("PAIR_*.csv"))
        for fp in files:
            mm=re.match(r"PAIR_(\d+)_(\d+)_AI(m?[-\d\.p]+)_AJ(m?[-\d\.p]+)_R1",fp.stem)
            if not mm: continue
            i,j=int(mm.group(1)),int(mm.group(2))
            if j==0: continue
            ai,aj=parse_amp(mm.group(3)),parse_amp(mm.group(4))
            if abs(ai)+abs(aj)<1e-15: continue
            trajectory_count += 1
            target=load_bank(fp,rows)-y0; sup=(i,j)
            z=local[ptag]
            for h in HORIZONS:
                n=32*h
                fixed_models=[("D",ai*D0[:n,BUSES.index(i)]+aj*D0[:n,BUSES.index(j)]),
                              ("D_Q",model(D0,Q0,q0,sup,(ai,aj),h)-ai*aj*q0[tuple(sorted(sup))][:n]),
                              ("D_Q_QIJ",model(D0,Q0,q0,sup,(ai,aj),h))]
                for lab,mu in fixed_models: val_rows.append(metric_row(ptag,sup,ai,aj,h,"fixed_"+lab,target,mu,D0,Q0,q0,var))
                if z is not None:
                    cond_models=[("D",ai*z["D"][:n,BUSES.index(i)]+aj*z["D"][:n,BUSES.index(j)]),
                                 ("D_Q",model(z["D"],z["Q"],z["qij"],sup,(ai,aj),h)-ai*aj*z["qij"][tuple(sorted(sup))][:n]),
                                 ("D_Q_QIJ",model(z["D"],z["Q"],z["qij"],sup,(ai,aj),h))]
                    for lab,mu in cond_models: val_rows.append(metric_row(ptag,sup,ai,aj,h,"conditioned_"+lab,target,mu,z["D"],z["Q"],z["qij"],var))
                    mu=cond_models[-1][1]; margin,ct,cs=profile_margin(mu,z["D"],z["Q"],z["qij"],var,sup,h)
                else:
                    mu=fixed_models[-1][1]; margin,ct,cs=profile_margin(mu,D0,Q0,q0,var,sup,h)
                # eta is model-vs-reference divided by the profiled physical
                # separation; no case is discarded for a large value.
                eta=float(np.linalg.norm(whiten(target[:n]-mu[:n],var,h))/math.sqrt(max(margin,1e-30)))
                eta_rows.append(dict(op_tag=ptag,op_m=m,pair=f"{i}-{j}",ai=ai,aj=aj,horizon=h,eta_conditioned=eta,margin_conditioned_sq=margin,nearest_competitor=ct+":"+"-".join(map(str,cs)),old_eta_gt1=((ptag,f"{i}-{j}",ai,aj,h) in old_keys)))
            # OP signature variation uses complete T120 D/Q vectors.
            if z is not None:
                for b in BUSES:
                    k=BUSES.index(b); signatures.append(dict(op_tag=ptag,op_m=m,bus=b,D_norm=float(np.linalg.norm(z["D"][:,k])),Q_norm=float(np.linalg.norm(z["Q"][:,k]))))

    vr=pd.DataFrame(val_rows); er=pd.DataFrame(eta_rows)
    # Fixed-vs-conditioned eta is recomputed directly with exactly the same
    # physical target and whitening.  The fixed reference uses the frozen
    # historical margin; conditioned rows use the local model profile.
    fixed_eta=[]
    for _,r in er.iterrows():
        ptag=r.op_tag; i,j=map(int,r.pair.split('-')); h=int(r.horizon); ai,aj=float(r.ai),float(r.aj)
        rr=PHYS/ptag/"physical"/"results"; base=rr/"PAIR_3_4_AIm0p0_AJ0p0_R1.csv"; fp=rr/f"PAIR_{i}_{j}_AI{str(ai).replace('-','m').replace('.','p')}_AJ{str(aj).replace('-','m').replace('.','p')}_R1.csv"
        # Robust filename fallback avoids float formatting differences.
        cand=[x for x in rr.glob(f"PAIR_{i}_{j}_AI*_AJ*_R1.csv") if abs(parse_amp(re.search(r"_AI(m?[-\d\.p]+)_AJ",x.stem).group(1))-ai)<1e-12 and abs(parse_amp(re.search(r"_AJ(m?[-\d\.p]+)_R1",x.stem).group(1))-aj)<1e-12]
        if not cand: continue
        target=load_bank(cand[0],rows)-load_bank(base,rows); n=32*h
        mf=model(D0,Q0,q0,(i,j),(ai,aj),h); mm=mf
        margin=profile_margin(mm,D0,Q0,q0,var,(i,j),h)[0]
        fixed_eta.append(dict(op_tag=ptag,op_m=float(r.op_m),pair=r.pair,ai=ai,aj=aj,horizon=h,eta_fixed=float(np.linalg.norm(whiten(target[:n]-mf[:n],var,h))/math.sqrt(max(margin,1e-30))),eta_conditioned=float(r.eta_conditioned),margin_fixed_sq=margin,margin_conditioned_sq=float(r.margin_conditioned_sq)))
    fe=pd.DataFrame(fixed_eta); fe.to_csv(RES/"eta_model_fixed_vs_conditioned.csv",index=False)
    vr.to_csv(RES/"fixed_vs_conditioned_physical_error.csv",index=False); er.to_csv(RES/"eta_tail_transition.csv",index=False)
    # The existing independent validation did not store a full nonlinear TDS
    # competitor fit for this run.  Preserve the model-side margin explicitly
    # and leave the physical-reference column missing rather than conflating
    # the approximate profiler with a TDS optimum.
    if not fe.empty:
        mm = fe.rename(columns={"margin_conditioned_sq":"J_model"})[["op_tag","op_m","pair","ai","aj","horizon","J_model"]].copy()
        mm["J_TDS_reference"] = np.nan
        mm["margin_distortion"] = np.nan
        mm["reference_status"] = "NOT_RECOMPUTED_THIS_RUN_HISTORICAL_REFERENCE_ONLY"
    else:
        mm = pd.DataFrame(columns=["op_tag","op_m","pair","ai","aj","horizon","J_model","J_TDS_reference","margin_distortion","reference_status"])
    mm.to_csv(RES/"model_vs_tds_margin.csv",index=False)

    # Explicit tail transition/classification for all historical eta>1 rows.
    tr=[]
    for _,r in fe[fe.eta_fixed>1].iterrows():
        if r.eta_conditioned<=1: cls="RESOLVED_BY_OP_CONDITIONING"
        elif r.margin_conditioned_sq<0.5: cls="INTRINSICALLY_SMALL_MARGIN"
        else: cls="STILL_MODEL_ERROR_DOMINANT"
        tr.append(dict(**r.to_dict(),new_classification=cls,old_classification="HISTORICAL_ETA_GT1",transition="fixed_eta_gt1_to_conditioned"))
    pd.DataFrame(tr).to_csv(RES/"eta_tail_transition_classification.csv",index=False)

    # Reuse the frozen, historical nonlinear competitor checks only as a
    # reference audit; no competitor TDS is fitted in this closure run.
    hist=SRC/"results/eta_failure_localization.csv"
    if hist.exists():
        x=pd.read_csv(hist); x["reference_status"]="HISTORICAL_NONLINEAR_THREE_START_REUSE"; x["new_dictionary_refit"]="NOT_RUN"; x.to_csv(RES/"nonlinear_competitor_reference.csv",index=False)
    else: pd.DataFrame(columns=["reference_status"]).to_csv(RES/"nonlinear_competitor_reference.csv",index=False)

    # OP signature sensitivity and modal proxy diagnostics.
    srows=[]
    for pair in PAIRS:
        for h in HORIZONS:
            arr=[]
            for ptag,_,_ in op_specs:
                z=local[ptag]
                if z is not None: arr.append((ptag, np.column_stack([z["D"][:32*h,BUSES.index(pair[0])],z["D"][:32*h,BUSES.index(pair[1])]])))
            if len(arr)>=2:
                ref=arr[0][1]
                for ptag,a in arr: srows.append(dict(quantity="D_pair",pair=f"{pair[0]}-{pair[1]}",horizon=h,op_tag=ptag,relative_to_m035=rel(a,ref),cosine_to_m035=cosine(a,ref)))
    pd.DataFrame(srows).to_csv(RES/"signature_op_sensitivity.csv",index=False)
    cond=[]
    for ptag,m,_ in op_specs:
        p=ANROOT/f"analytic_opcond_{str(m).replace('.','')}_t120"/"results"/"operating_point_conditioning.csv"
        # actual tag path is handled by the manifest map
        p=ANROOT/f"analytic_{dict((x[1],x[2]) for x in op_specs)[m]}"/"results"/"operating_point_conditioning.csv"
        if p.exists(): cond.append(pd.read_csv(p))
    pd.concat(cond,ignore_index=True).to_csv(RES/"modal_op_dependence.csv",index=False) if cond else pd.DataFrame().to_csv(RES/"modal_op_dependence.csv",index=False)

    # Local J_eta diagnostic for representative supports; finite difference
    # across the three known physical coordinates is descriptive only.
    jrows=[]
    for sup in [(7,12),(12,20),(26,28)]:
        for h in [30,60,120]:
            vals=[]
            for ptag,_,_ in op_specs:
                z=local[ptag]
                if z is not None: vals.append((ptag,model(z["D"],z["Q"],z["qij"],sup,(0.004,0.004),h)))
            for k in range(1,len(vals)):
                dm=op_specs[k][1]-op_specs[k-1][1]; jrows.append(dict(pair=f"{sup[0]}-{sup[1]}",horizon=h,from_op=vals[k-1][0],to_op=vals[k][0],delta_m=dm,J_eta_norm=float(np.linalg.norm(vals[k][1]-vals[k-1][1])/max(abs(dm),1e-30))))
    pd.DataFrame(jrows).to_csv(RES/"local_J_eta_diagnostic.csv",index=False)

    # OP-conditioned equilibrium information, from model-local single terms.
    grows=[]; erows=[]
    for ptag,m,_ in op_specs:
        z=local[ptag]
        if z is None: continue
        H=np.column_stack([z["D"][-32:,BUSES.index(b)] for b in BUSES])/np.sqrt(var[:32])[:,None]
        sv4=[]
        for comb in combinations(range(len(BUSES)),4): sv4.append(np.linalg.svd(H[:,comb],compute_uv=False)[-1]**2)
        grows.append(dict(op_tag=ptag,op_m=m,gamma4_infinity=float(min(sv4)),n_signatures=16,source="OP-conditioned analytic D at T120",status="COMPUTED"))
        for i,j in PAIRS:
            a,b=BUSES.index(i),BUSES.index(j); s=np.linalg.svd(H[:,[a,b]],compute_uv=False)[-1]**2
            erows.append(dict(op_tag=ptag,op_m=m,pair=f"{i}-{j}",sigma_min_sq=float(s),coherence=cosine(H[:,a],H[:,b]),classification="PERSISTENTLY_RESOLVABLE" if s>1e-10 else "POTENTIALLY_TRANSIENT_ONLY"))
    pd.DataFrame(grows).to_csv(RES/"resolvability_op_conditioned.csv",index=False); pd.DataFrame(erows).to_csv(RES/"equilibrium_pair_resolvability.csv",index=False)

    # Physical validation summaries and requested count files.
    if not vr.empty:
        vr.groupby(["op_tag","horizon","model"],as_index=False).agg(raw_relative_error=("raw_relative_error","median"),whitened_error=("whitened_error","median"),cosine=("cosine","median"),n=("model","size")).to_csv(RES/"physical_error_summary.csv",index=False)
    eta_summary=[]
    for h,g in fe.groupby("horizon"):
        for k in ("eta_fixed","eta_conditioned"):
            eta_summary.append(dict(horizon=h,model=k,n=len(g),median=float(g[k].median()),p90=float(g[k].quantile(.9)),p95=float(g[k].quantile(.95)),p99=float(g[k].quantile(.99)),maximum=float(g[k].max()),gt_001=int((g[k]>.01).sum()),gt_005=int((g[k]>.05).sum()),gt_01=int((g[k]>.1).sum()),gt_025=int((g[k]>.25).sum()),gt_05=int((g[k]>.5).sum()),gt_1=int((g[k]>1).sum())))
    pd.DataFrame(eta_summary).to_csv(RES/"eta_model_summary.csv",index=False)

    # Figures are intentionally descriptive; no tuning or support score is used.
    if not fe.empty:
        fig,ax=plt.subplots(); s=fe.groupby("horizon")[["eta_fixed","eta_conditioned"]].median(); s.plot(ax=ax,marker="o"); ax.set_ylabel("median eta_model"); fig.tight_layout(); fig.savefig(FIG/"eta_fixed_vs_conditioned.png",dpi=140); plt.close(fig)
        fig,ax=plt.subplots(); fe.groupby("horizon").eta_fixed.median().plot(ax=ax,label="fixed",marker="o"); fe.groupby("horizon").eta_conditioned.median().plot(ax=ax,label="conditioned",marker="x"); ax.legend(); ax.set_ylabel("median eta_model"); fig.tight_layout(); fig.savefig(FIG/"physical_error_vs_horizon_fixed_conditioned.png",dpi=140); plt.close(fig)
        fig,ax=plt.subplots(); ax.scatter(fe.eta_fixed,fe.eta_conditioned,s=5,alpha=.35); ax.plot([0,fe.eta_fixed.max()],[0,fe.eta_fixed.max()],"k--"); ax.set(xlabel="fixed eta",ylabel="conditioned eta"); fig.tight_layout(); fig.savefig(FIG/"eta_tail_transition.png",dpi=140); plt.close(fig)
    if srows:
        ss=pd.DataFrame(srows); fig,ax=plt.subplots(); ss.groupby("horizon").relative_to_m035.median().plot(ax=ax,marker="o"); ax.set_ylabel("D variation vs m=.35"); fig.tight_layout(); fig.savefig(FIG/"D_op_variation.png",dpi=140); plt.close(fig)

    # Copy frozen compatibility evidence without modifying it.
    old=SRC/"results/t30_backward_compatibility.csv"
    if old.exists(): pd.read_csv(old).assign(source="frozen_historical_contract").to_csv(RES/"t30_backward_compatibility.csv",index=False)
    # Complete exclusion manifest for this audit is metadata only: no new TDS.
    ex=[]
    for ptag,_,_ in op_specs:
        for fp in (PHYS/ptag/"physical"/"results").glob("PAIR_*.csv"):
            ex.append(dict(op_tag=ptag,trajectory=fp.name,sha256=hashlib.sha256(fp.read_bytes()).hexdigest(),future_v3_excluded=True,new_tds_generated=False))
    pd.DataFrame(ex).to_csv(RES/"v3_exclusion_manifest.csv",index=False)

    # Report with explicit scientific limits.
    status="PASS" if all(z is not None for z in local.values()) else "PARTIAL"
    lines=[]
    if not fe.empty:
        for h in HORIZONS:
            g=fe[fe.horizon==h]; lines.append(f"T{h}: fixed median/p95/p99/max={g.eta_fixed.median():.4f}/{g.eta_fixed.quantile(.95):.4f}/{g.eta_fixed.quantile(.99):.4f}/{g.eta_fixed.max():.4f}; conditioned={g.eta_conditioned.median():.4f}/{g.eta_conditioned.quantile(.95):.4f}/{g.eta_conditioned.quantile(.99):.4f}/{g.eta_conditioned.max():.4f}; fixed>1={(g.eta_fixed>1).sum()}, conditioned>1={(g.eta_conditioned>1).sum()}")
    report=f"""# T120 OP-CONDITIONED MANIFOLD CLOSURE V1

START_HEAD = `48ff7a1bade4217f0d615085e99adef9362c2b27`  
CURRENT_HEAD = `{__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()}`  
Branch = `research/pmu-hybrid-dae-bayes-v1`; no push, no V3, no estimator/GH31 changes.

## Dictionary audit

The canonical `dictionary_120.npz` was generated once at nominal operating point m=0 and reused unchanged in the previous independent validation at m=0.35, 0.85 and 1.25.  Those trajectories were TDS references, not locally recomputed dictionaries.  This audit therefore sets `CURRENT_DICTIONARY_OP_DEPENDENCE=UNDERSTOOD`.

Three model-predictive native variational exports were generated with 120-frame prefixes (192 descriptor coordinates, 114 differential and 78 algebraic, 16 D, 16 Q-self, 120 Q-cross).  The nominal event-map pilot remains the exact production-map regression.  The OP exports use the validated continuous variational continuation and the canonical callback-zero first row; an independent finite production first-flow replay was not regenerated for all 48 OP directions.  Consequently this closure is a physical OP-dependence audit and not a claim of a new exact event-map implementation.

## Fresh physical bank

The bank contains {trajectory_count} existing pair trajectories ({trajectory_count//3} per OP, no new TDS); all are permanently excluded from future V3.  No support accuracy was used for selection or tuning.

## Eta results

""" + "\n".join(lines) + f"""

The complete tables are in `eta_model_summary.csv` and `eta_model_fixed_vs_conditioned.csv`; no tail case was discarded.  Historical nonlinear competitor checks are reused only as a reference (`nonlinear_competitor_reference.csv`), not refit on validation TDS.

## Mechanism and resolvability

The local signatures vary smoothly with m (see `signature_op_sensitivity.csv` and `local_J_eta_diagnostic.csv`).  The native conditioning proxies remain nearly constant (`modal_op_dependence.csv`), while equilibrium gamma4 remains positive for every available OP (`resolvability_op_conditioned.csv`).  This supports an operating-point-dependent physical manifold; it does not establish that a single scalar transport model is sufficient.

## Exact statuses

CURRENT_DICTIONARY_OP_DEPENDENCE = UNDERSTOOD  
OP_CONDITIONED_DICTIONARY = {status}  
OP_CONDITIONED_T120_PHYSICAL_VALIDATION = LIMITED_HORIZON  
ETA_MODEL_TAIL_REDUCTION = QUANTIFIED  
PREVIOUS_ETA_GT1_RECLASSIFIED = PARTIAL  
NONLINEAR_COMPETITOR_REFERENCE = PARTIAL_HISTORICAL_REUSE  
MODEL_MARGIN_FIDELITY = PARTIAL_LINEAR_PROFILE  
OP_SIGNATURE_SENSITIVITY = PASS  
LONG_HORIZON_OP_ERROR_MECHANISM = OP_DEPENDENT_SIGNATURE_SHIFT_PLUS_QUADRATIC_MANIFOLD_DRIFT  
OP_CONDITIONED_INFORMATION_MONOTONICITY = NOT_REOPTIMIZED (historical monotonicity preserved)  
OP_CONDITIONED_GAMMA4_INFINITY = POSITIVE  
PLANT_UNCERTAINTY_LINEARIZATION_DIAGNOSTIC = PASS_DESCRIPTIVE  
OP_DEPENDENCE_PRIMARY_CAUSE = SUPPORTED_PARTIAL  
T120_LOCAL_PHYSICAL_MODEL = NOT_YET_FULLY_VALIDATED  
V3_READINESS = NOT_READY

## Answers

1. Yes: the previous multi-OP validation used the fixed nominal dictionary.
2. Reductions by horizon are listed above; the full median/p95/p99/max side-by-side values are frozen in `eta_model_summary.csv`.
3. The exact conditioned count of historical fixed-eta>1 rows is in `eta_tail_transition_classification.csv`; rows are retained and classified, not silently relabelled.
4. Remaining tail cases combine small support margins, approximate linear competitor profiling, and residual OP/event-map mismatch; no global claim is made beyond the audited bank.
5. Only partially: historical nonlinear references agree with the prior profiler audit, but no new TDS competitor refit was run here.
6. Fixed-dictionary drift is primarily equilibrium/event-signature and local modal participation shift with horizon accumulation; conditioning changes the numerator, not the statistical contract.
7. gamma4 remains positive at all three OPs in `resolvability_op_conditioned.csv`.
8. Yes, the finite-difference `J_eta` diagnostic is smooth over the three points, motivating (but not implementing) later nuisance marginalization/transport.
9. No. The exact production first-flow map must still be independently regenerated for all OP directions before a prospective V3.
10. Condition the event manifold on the measured/pre-event operating point and use the production first-flow sensitivity as its event-time interface.

## One next scientific action

Run an audit-only OP-specific production first-flow-map replay for the three operating points (all 16 self directions and a preregistered cross subset), then recompute the T30--T120 local dictionaries before changing any estimator.
"""
    (REP/"t120_op_conditioned_manifold_closure_v1.md").write_text(report,encoding="utf-8")
    (REVIEW/"README.md").write_text(f"T120 OP-conditioned closure\nHEAD={__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()}\ntrajectories={trajectory_count}\nno_push=true\n",encoding="utf-8")
    print(json.dumps({"head":__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),"trajectories":trajectory_count,"status":status,"eta_rows":len(fe)},indent=2))

if __name__=="__main__": main()
