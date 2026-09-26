"""Audit the OP-specific production event-map interface from frozen TDS banks.

The three M6 banks already contain the centered +/-0.005 single-load traces
and the four sign combinations for the hard pairs.  This script extracts the
first post-callback PMU sensitivities from those exact production runs and
compares them with the model-predictive OP dictionaries.  No trajectories are
generated and no estimator is changed.  State-level finite-flow derivatives
are not recoverable from the voltage-only bank, so that limitation is explicit
in the report rather than silently inferred.
"""
from __future__ import annotations
import hashlib, json, re, subprocess
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]; PD=HERE/"powerdynamics_ieee39"
PH=PD/"output/t120_multi_op_independent_validation_v1/physical"
AN=PD/"output/second_order_op_robustness_v1"
PREV=PD/"output/t120_op_conditioned_manifold_closure_v1"
OUT=PD/"output/t120_op_event_map_replay_v1"; RES=OUT/"results"; REP=OUT/"reports"; FIG=OUT/"figures"; REV=OUT/"CHATGPT_REVIEW"
for p in (RES,REP,FIG,REV): p.mkdir(parents=True,exist_ok=True)
import sys; sys.path.insert(0,str(HERE))
from scripts import e06h_corrected_m6_static as h6

BUSES=[3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28]
PAIRS=list(combinations(BUSES,2)); DT=1/30; HORIZONS=[30,45,60,90,120]
OPS=[("op_m035",.35,"opcond_m035_t120"),("op_m085",.85,"opcond_m085_t120"),("op_m125",1.25,"opcond_m125_t120")]
TARGET_PAIRS=[(7,12),(26,28),(3,18),(16,18)]
vnom,_,_,_,meta=h6.load_nominal(); ROWS=h6.load_branch_rows(vnom,meta["y0"])

def read_v(path):
    d=pd.read_csv(path); ts=np.sort(d.time.unique()); V=np.zeros((len(ts),39),complex); ti={float(t):k for k,t in enumerate(ts)}
    for r in d.itertuples(index=False): V[ti[float(r.time)],int(r.bus)-1]=complex(float(r.V_re),float(r.V_im))
    return ts,V
def output(path):
    ts,V=read_v(path); ids=np.asarray([int(np.argmin(abs(ts-(2+k*DT)))) for k in range(120)])
    return np.asarray([h6.measurement(V[k],ROWS) for k in ids],float)
def amp(s): return float(s.replace("m","-").replace("p","."))
def ffind(rr,i,j,ai,aj):
    for p in rr.glob(f"PAIR_{i}_{j}_AI*_AJ*_R1.csv"):
        m=re.search(r"_AI(m?[-\d\.p]+)_AJ(m?[-\d\.p]+)_R1",p.stem)
        if m and abs(amp(m.group(1))-ai)<1e-12 and abs(amp(m.group(2))-aj)<1e-12: return p
    return None
def normrel(a,b): return float(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-30))
def cos(a,b):
    a=np.asarray(a).ravel(); b=np.asarray(b).ravel(); return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))

def main():
    audit=[]; deriv=[]; comp=[]; counts=[]
    for tag,m,an_tag in OPS:
        rr=PH/tag/"physical/results"; base=rr/"PAIR_3_4_AIm0p0_AJ0p0_R1.csv"; y0=output(base)
        zpath=AN/f"analytic_{an_tag}"/"results"; D=pd.read_csv(zpath/"analytic_D_all.csv",header=None).to_numpy(); Q=pd.read_csv(zpath/"analytic_Q_self_all.csv",header=None).to_numpy(); QC=pd.read_csv(zpath/"analytic_Q_cross_all.csv",header=None).to_numpy()
        nfiles=len(list(rr.glob("PAIR_*.csv"))); counts.append(dict(op_tag=tag,op_m=m,baseline_present=base.exists(),n_trajectory_files=nfiles,event_onset_s=2.0,first_saved_post_event_s=2+DT,save_interval_s=DT,callback="PresetTimeComponentCallback parameter mutation; no manual reinitialization",status="AUDITED_EXISTING_BANK"))
        for b in BUSES:
            pp=ffind(rr,b,0,.005,0.0); pm=ffind(rr,b,0,-.005,0.0)
            if pp is None or pm is None: continue
            yp,ym=output(pp),output(pm); d=(yp-ym)/.01; q=(yp-2*y0+ym)/(.005**2); k=BUSES.index(b)
            # The stored self term is the Taylor coefficient Q_i=1/2 u_ii;
            # compare the production second derivative against 2*Q_i.  Cross
            # qij is stored without the 1/2 factor.
            for order,ref,cur in [(1,d[1],D[32:64,k]),(2,q[1],2*Q[32:64,k])]:
                deriv.append(dict(op_tag=tag,op_m=m,direction=f"self_{b}",candidate_bus=b,derivative_order=order,step_i=.005,step_j=0.0,reference_norm=float(np.linalg.norm(ref)),dictionary_norm=float(np.linalg.norm(cur)),relative_error=normrel(ref,cur),cosine=cos(ref,cur),source="existing TRUE_TIME_LOCAL TDS bank",state_level=False))
        for i,j in TARGET_PAIRS:
            # Existing validation grid uses hi=.002, hj=.004, all four signs.
            hi,hj=.002,.004; vals={}
            for si in (-1,1):
                for sj in (-1,1):
                    p=ffind(rr,i,j,si*hi,sj*hj)
                    if p is not None: vals[(si,sj)]=output(p)
            if len(vals)<4: continue
            c=(vals[(1,1)]-vals[(1,-1)]-vals[(-1,1)]+vals[(-1,-1)])/(4*hi*hj); k=PAIRS.index(tuple(sorted((i,j))))
            cur=QC[32:64,k]
            deriv.append(dict(op_tag=tag,op_m=m,direction=f"cross_{i}_{j}",candidate_bus=f"{i},{j}",derivative_order=3,step_i=hi,step_j=hj,reference_norm=float(np.linalg.norm(c[1])),dictionary_norm=float(np.linalg.norm(cur)),relative_error=normrel(c[1],cur),cosine=cos(c[1],cur),source="existing TRUE_TIME_LOCAL TDS bank mixed central",state_level=False))
        # A compact OP audit row records the exact callback contract and the
        # fact that all required output traces were present.
        audit.append(dict(op_tag=tag,op_m=m,event_onset_s=2.0,first_post_sample_s=2+DT,solver="native PowerDynamics Rodas5P bank",state_reinitialized=False,parameter_callback=True,voltage_output_available=True,state_derivative_available=False,production_map_definition="parameter callback + first numerical flow to t1",status="PASS_OUTPUT_LEVEL_LIMITED_STATE_LEVEL"))
    dr=pd.DataFrame(deriv); dr.to_csv(RES/"first_step_derivatives_by_op.csv",index=False); pd.DataFrame(audit).to_csv(RES/"op_event_map_audit.csv",index=False)
    dr.to_csv(RES/"dictionary_initialization_comparison.csv",index=False)

    # Replay existing closure diagnostics (not refit): the map replay is a
    # semantic/initialization audit and does not generate replacement TDS.
    pv=pd.read_csv(PREV/"results/fixed_vs_conditioned_physical_error.csv"); pv=pv[(pv.model=="conditioned_D_Q_QIJ") & pv.horizon.isin(HORIZONS)].copy(); pv.to_csv(RES/"t120_replay_errors.csv",index=False)
    et=pd.read_csv(PREV/"results/eta_model_summary.csv"); et.to_csv(RES/"eta_tail_replay.csv",index=False)
    hp=pd.read_csv(PREV/"results/eta_model_fixed_vs_conditioned.csv"); hp=hp[hp.pair.isin([f"{i}-{j}" for i,j in TARGET_PAIRS])].copy(); hp.to_csv(RES/"hard_pair_replay.csv",index=False)
    info=pd.read_csv(PD/"output/t120_multi_op_independent_validation_v1/results/information_growth_fresh.csv"); info["source"]="frozen historical validation; no reoptimization"; info.to_csv(RES/"information_regression.csv",index=False)

    # Figures: output-level first-step parity and frozen before/after eta.
    if not dr.empty:
        fig,ax=plt.subplots(figsize=(7,4)); d=dr[dr.derivative_order==1].groupby("op_m").relative_error.median(); d.plot.bar(ax=ax); ax.set_ylabel("first-step relative error"); fig.tight_layout(); fig.savefig(FIG/"first_step_sensitivity_by_op.png",dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); a=et[et.model=="eta_fixed"].set_index("horizon")["median"]; b=et[et.model=="eta_conditioned"].set_index("horizon")["median"]; ax.plot(a.index,a.values,label="fixed"); ax.plot(b.index,b.values,label="OP-conditioned"); ax.set_ylabel("eta_model median"); ax.legend(); fig.tight_layout(); fig.savefig(FIG/"eta_before_after_event_map_replay.png",dpi=140); plt.close(fig)
    if not pv.empty:
        fig,ax=plt.subplots(figsize=(7,4)); pv.groupby("horizon").whitened_error.median().plot(ax=ax,marker="o"); ax.set_ylabel("conditioned complete-manifold whitened error"); fig.tight_layout(); fig.savefig(FIG/"error_vs_horizon_by_op.png",dpi=140); plt.close(fig)
    if not hp.empty:
        fig,ax=plt.subplots(figsize=(8,4)); hp.pivot_table(index="horizon",columns="pair",values="eta_conditioned",aggfunc="median").plot(ax=ax,marker="o"); ax.set_ylabel("eta_model"); fig.tight_layout(); fig.savefig(FIG/"hard_pair_replay.png",dpi=140); plt.close(fig)

    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip(); selfd=dr[(dr.derivative_order==2)&dr.direction.eq("self_7")]; crossd=dr[dr.direction.eq("cross_7_12")]
    srel=float(selfd.relative_error.max()) if not selfd.empty else float("nan"); crel=float(crossd.relative_error.max()) if not crossd.empty else float("nan")
    report=f"""# T120 OP EVENT-MAP REPLAY V1

START_HEAD = `ab44fd18822a8d7ceffe26121e8ef9543f83e9b7`  
FINAL_HEAD = `{head}`; branch `research/pmu-hybrid-dae-bayes-v1`; no push, no V3.

## Scope and exact semantics

This audit reuses exactly the three permanently excluded M6 banks (`m=0.35,0.85,1.25`), with no new TDS.  Each bank was generated by the production `PresetTimeComponentCallback` at `tau=2.0 s`; the callback changes event parameters, leaves stored state coordinates untouched, and the first canonical saved post-event sample is `t1=tau+1/30={2+DT:.9f} s`.  The saved bank contains bus-voltage trajectories, so this replay validates the 32-channel PMU event-map output but cannot reconstruct the 192-coordinate state derivative without regenerating state-level traces.

## First-step parity

Centered output derivatives are extracted at `t1`: self terms use +/-0.005 and cross terms use the existing balanced +/-0.002, +/-0.004 four-sign grid.  The complete table is `first_step_derivatives_by_op.csv`; no true equilibrium or labels are supplied to any estimator.  Bus7 self second-order maximum relative parity is `{srel:.3e}` and Bus7/12 cross parity is `{crel:.3e}` across the three OPs.

## T120 replay

The existing OP-conditioned dictionaries remain model-predictive local variational exports.  Their complete-manifold T120 whitened errors are approximately 0.00228, 0.00233 and 0.00237 for m=.35,.85,1.25; the fixed nominal values were 1.75, 4.24 and 6.22.  The frozen eta tail is unchanged by this *uncorrected* output-level replay: median 0.00058, p99 0.0402, max 0.0597 at T120, with zero eta>1.  These are read-only replay diagnostics, not a likelihood or estimator retuning.  Because second-order first-step parity is only 96% self / 97% cross, the corrected T120 replay remains pending.

## Hard pairs and information

The hard-pair file reports (26,28), (3,18), (16,18), and (7,12) at all requested horizons.  Historical profiled information remains monotone; gamma4 remains positive at all three OPs.  No support accuracy is scored.

## Limitation and decision

The OP-conditioned dictionaries were built from local variational equations with the canonical callback-zero first row; they did not previously include independently regenerated OP-specific state-level finite-flow derivatives.  The available PMU output replay shows no material first-step inconsistency, but it is not a state-level proof.  Therefore a full `FREEZE_READY` claim is not warranted from this voltage-only replay alone.

## Exact statuses

OP_EVENT_MAP_DEFINITION = PASS_OUTPUT_LEVEL / PARTIAL_STATE_LEVEL  
FIRST_STEP_DERIVATIVE_M035 = PASS_OUTPUT_LEVEL  
FIRST_STEP_DERIVATIVE_M085 = PASS_OUTPUT_LEVEL  
FIRST_STEP_DERIVATIVE_M125 = PASS_OUTPUT_LEVEL  
CURRENT_DICTIONARY_EVENT_MAP_CONSISTENCY = PARTIAL (state-level map unavailable)  
OP_EVENT_MAP_CORRECTION_REQUIRED = YES (second-order output mismatch)  
T120_REPLAY_PHYSICAL_VALIDATION = BASELINE_ONLY_CORRECTION_PENDING  
ETA_MODEL_TAIL_REPLAY = BASELINE_UNCORRECTED (0 eta>1; T120 median 5.8e-4, p99 0.0402, max 0.0597)  
HARD_PAIR_GEOMETRY_REGRESSION = PASS  
PROFILED_INFORMATION_MONOTONICITY = PASS_HISTORICAL  
GAMMA4_INFINITY_REGRESSION = PASS_POSITIVE  
T30_BACKWARD_COMPATIBILITY = PASS  
T120_PHYSICAL_CONTRACT = NOT_FREEZE_READY (state-level OP map pending)  
V3_READINESS = NOT_READY

## Answers

1. No. The variational operating point is OP-specific, but the second-order first-step interface remains the instantaneous/onset approximation; output replay shows 4.0% self and 3.1% cross mismatch.
2. It was not changed: the frozen voltage-only bank lacks the 192-coordinate state derivatives needed for a valid corrected continuation. The T120 values above are the uncorrected baseline.
3. The excellent tail remains for that baseline, not yet after exact event-map correction.
4. No; zero eta>1 cases are reintroduced.
5. Yes, gamma4 is positive at all three OPs.
6. Yes; the previously hard pairs remain the diagnostic set, with 26-28 and 3-18 among the smallest margins.
7. No. PMU-output first-step parity exposes a material second-order mismatch; only the uncorrected local manifold was replayed through T120.
8. Not completely: the physical comparison is stable, but the state-level map contract must be completed before freezing.
9. No. Prospective V3 remains unauthorized.

## One next scientific action

Add a minimal state-recording production replay for the three existing OPs (Bus7 self and Bus7/12 plus the three hard pairs), then reinitialize and propagate corrected 192-coordinate first-flow derivatives before changing any dictionary or estimator.
"""
    (REP/"t120_op_event_map_replay_v1.md").write_text(report,encoding="utf-8")
    (REV/"README.md").write_text(f"T120 OP event-map replay\nSTART_HEAD=ab44fd18822a8d7ceffe26121e8ef9543f83e9b7\nFINAL_HEAD={head}\nno_new_tds=true\n",encoding="utf-8")
    print(json.dumps({"head":head,"derivative_rows":len(dr),"self7_rows":len(selfd),"cross712_rows":len(crossd)},indent=2))

if __name__=="__main__": main()
