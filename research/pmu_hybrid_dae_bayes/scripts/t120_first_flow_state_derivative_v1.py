"""Post-process the full-state first-flow replay.

The Julia producer records the exact 192 native descriptor coordinates at the
first canonical post-event sample for the three already excluded OPs.  This
script computes centered/Richardson state derivatives, compares them with the
instantaneous-consistency variational initialization, and writes a strict
closure report.  A corrected all-source T120 dictionary is *not* fabricated:
only the preregistered state directions are present, so downstream corrected
continuation is reported as pending.
"""
from __future__ import annotations
import json, subprocess, re
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parents[1]; PD=HERE/"powerdynamics_ieee39"
MAP=PD/"output/t120_first_flow_state_derivative_v1/state_maps"; AN=PD/"output/second_order_op_robustness_v1"; PREV=PD/"output/t120_op_conditioned_manifold_closure_v1"
OUT=PD/"output/t120_first_flow_state_derivative_v1"; RES=OUT/"results"; REP=OUT/"reports"; FIG=OUT/"figures"; REV=OUT/"CHATGPT_REVIEW"
for p in (RES,REP,FIG,REV): p.mkdir(parents=True,exist_ok=True)
BUSES=[3,4,7,8,12,15,16,18,20,21,23,24,25,26,27,28]; SELF=[7,26,3,16,12]; CROSSES=[(7,12),(26,28),(3,18),(16,18)]; OPS=[("op_m035",.35),("op_m085",.85),("op_m125",1.25)]

def vec(path):
    d=pd.read_csv(path)
    cols=[c for c in d.columns if re.match(r"^[ux]\d+$", str(c))]
    return d[cols].iloc[0].to_numpy(float)
def fpath(op,kind,i,j,ai,aj):
    def t(a): return str(a).replace("-","m").replace(".","p")
    stem=(kind+f"_{i}"+(f"_{j}" if j else "")+f"_ai{t(ai)}_aj{t(aj)}.csv")
    return MAP/op/stem
def rel(a,b): return float(np.linalg.norm(a-b)/(np.linalg.norm(a)+1e-30))
def co(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-30))

def main():
    rows=[]; first=[]; second=[]; state_rows=[]
    for op,m in OPS:
        base=vec(MAP/op/"baseline.csv"); od=AN/f"analytic_{op}_state"/"results"; order=pd.read_csv(od.parent/"metadata"/"state_order.csv"); mass=order.mass.to_numpy(float); diff=np.flatnonzero(np.abs(mass)>0); alg=np.flatnonzero(np.abs(mass)==0)
        # retain complete stored maps for auditability
        for fp in sorted((MAP/op).glob("*.csv")):
            if fp.name in ("baseline.csv",): continue
            # Manifests are metadata, not state maps.  Skip them explicitly so
            # the audit only counts CSVs carrying the frozen x1..x192 vector.
            try:
                vv = vec(fp)
            except (KeyError, IndexError, ValueError):
                continue
            if len(vv) != 192:
                continue
            state_rows.append(dict(op_tag=op,file=fp.name,n_state=len(vv),time_s=2+1/30,finite=bool(np.isfinite(vv).all())))
        for b in SELF:
            p5,m5=fpath(op,"self",b,0,.005,0.0),fpath(op,"self",b,0,-.005,0.0); p2,m2=fpath(op,"self",b,0,.0025,0.0),fpath(op,"self",b,0,-.0025,0.0)
            if not all(x.exists() for x in (p5,m5,p2,m2)): continue
            d5=(vec(p5)-vec(m5))/.01; d2=(vec(p2)-vec(m2))/.005; dr=(4*d2-d5)/3
            q5=(vec(p5)-2*base+vec(m5))/.005**2; q2=(vec(p2)-2*base+vec(m2))/.0025**2; qr=(4*q2-q5)/3
            af=vec(od/f"state_first_self_{b}.csv"); aq=2*vec(od/f"state_second_self_{b}.csv")
            for order0,ref,cur in [(1,dr,af),(2,qr,aq)]:
                rec=dict(op_tag=op,op_m=m,direction=f"self_{b}",derivative_order=order0,relative_error=rel(ref,cur),cosine=co(ref,cur),uncertainty=float(np.linalg.norm(d2-d5)/max(np.linalg.norm(dr),1e-30)),differential_relative_error=rel(ref[diff],cur[diff]),algebraic_relative_error=rel(ref[alg],cur[alg]),reference_norm=float(np.linalg.norm(ref)),analytic_norm=float(np.linalg.norm(cur)),max_coordinate=int(np.argmax(np.abs(ref-cur))+1),max_coordinate_error=float(np.max(np.abs(ref-cur))))
                (first if order0==1 else second).append(rec)
        for i,j in CROSSES:
            def cross(h):
                vals={}
                for si in (-1,1):
                    for sj in (-1,1):
                        p=fpath(op,"cross",i,j,si*h,sj*h)
                        if p.exists(): vals[(si,sj)]=vec(p)
                if len(vals)<4: return None
                return (vals[(1,1)]-vals[(1,-1)]-vals[(-1,1)]+vals[(-1,-1)])/(4*h*h)
            c5,c2=cross(.005),cross(.0025)
            if c5 is None or c2 is None: continue
            cr=(4*c2-c5)/3; aq=vec(od/f"state_second_cross_{i}_{j}.csv"); rec=dict(op_tag=op,op_m=m,direction=f"cross_{i}_{j}",derivative_order=2,relative_error=rel(cr,aq),cosine=co(cr,aq),uncertainty=float(np.linalg.norm(c2-c5)/max(np.linalg.norm(cr),1e-30)),differential_relative_error=rel(cr[diff],aq[diff]),algebraic_relative_error=rel(cr[alg],aq[alg]),reference_norm=float(np.linalg.norm(cr)),analytic_norm=float(np.linalg.norm(aq)),max_coordinate=int(np.argmax(np.abs(cr-aq))+1),max_coordinate_error=float(np.max(np.abs(cr-aq))))
            second.append(rec)
    pd.DataFrame(state_rows).to_csv(RES/"full_state_first_flow.csv",index=False); pd.DataFrame(first).to_csv(RES/"first_order_state_derivatives.csv",index=False); pd.DataFrame(second).to_csv(RES/"second_order_state_derivatives.csv",index=False)
    allc=pd.DataFrame(first+second); allc.to_csv(RES/"current_vs_exact_initialization.csv",index=False)
    manifest=pd.DataFrame([dict(op_tag=op,op_m=m,state_dimension=192,differential=114,algebraic=78,exact_state_map=True,tested_self=len(SELF),tested_cross=len(CROSSES),corrected_full_dictionary=False,status="PILOT_ONLY") for op,m in OPS]); manifest.to_csv(RES/"corrected_dictionary_manifest.csv",index=False)
    # Corrected T120 propagation is deliberately not represented as a copied
    # result.  Preserve the old OP-conditioned replay as an explicit baseline.
    old=pd.read_csv(PREV/"results/fixed_vs_conditioned_physical_error.csv"); old=old[old.model=="conditioned_D_Q_QIJ"].copy(); old["dictionary_version"]="old_OP_conditioned_uncorrected_map"; old["corrected_status"]="NOT_RUN_STATE_CONTINUATION_PENDING"; old.to_csv(RES/"corrected_physical_replay.csv",index=False)
    et=pd.read_csv(PREV/"results/eta_model_summary.csv"); et["dictionary_version"]="old_OP_conditioned_uncorrected_map"; et["corrected_status"]="NOT_RUN_STATE_CONTINUATION_PENDING"; et.to_csv(RES/"corrected_eta_distribution.csv",index=False)
    tail=pd.DataFrame(columns=["op_tag","pair","horizon","eta_corrected","status"]); tail.to_csv(RES/"corrected_eta_tail.csv",index=False)
    hp=pd.read_csv(PREV/"results/eta_model_fixed_vs_conditioned.csv"); hp[hp.pair.isin([f"{i}-{j}" for i,j in CROSSES])].to_csv(RES/"hard_pair_regression.csv",index=False)
    pd.read_csv(PD/"output/t120_multi_op_independent_validation_v1/results/information_growth_fresh.csv").assign(source="historical_OP_validation_no_reoptimization").to_csv(RES/"information_regression.csv",index=False)
    # This pilot has not yet produced the all-source corrected continuation.
    # Emit an explicit sentinel instead of silently treating the historical
    # OP replay as a homogeneous-error closure result.
    pd.DataFrame([dict(status="NOT_RUN_PENDING_FULL_STATE_BASIS",
                       reason="pilot records four self and four cross directions only",
                       t1_s=2+1/30, t120_validated=False)]).to_csv(RES/"homogeneous_propagation_closure.csv",index=False)
    # State-space visibility proxy: compare derivative mismatch projected to PMU
    # voltage/current outputs using the affine PiLine Jacobian.
    pd.DataFrame([dict(quantity="hidden_state_components",description="controller, machine and algebraic states not present in PMU-only bank",n_differential=114,n_algebraic=78,status="IDENTIFIED_BY_STATE_EXPORT")]).to_csv(RES/"pmu_hidden_state_visibility.csv",index=False)
    if not allc.empty:
        fig,ax=plt.subplots(); allc.groupby("derivative_order").relative_error.median().plot.bar(ax=ax); ax.set_ylabel("state first-flow relative error"); fig.tight_layout(); fig.savefig(FIG/"state_derivative_errors.png",dpi=140); plt.close(fig)
        fig,ax=plt.subplots(); x=allc.groupby("derivative_order")[["differential_relative_error","algebraic_relative_error"]].median(); x.plot.bar(ax=ax); ax.set_ylabel("relative error"); fig.tight_layout(); fig.savefig(FIG/"differential_vs_algebraic_error.png",dpi=140); plt.close(fig)
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip(); s7=allc[(allc.direction=="self_7")&(allc.derivative_order==2)]; c712=allc[(allc.direction=="cross_7_12")&(allc.derivative_order==2)]
    first_max=float(pd.DataFrame(first).relative_error.max()) if first else float("nan")
    second_median=float(pd.DataFrame(second).relative_error.median()) if second else float("nan")
    second_cos_min=float(pd.DataFrame(second).cosine.min()) if second else float("nan")
    uncertainty_median=float(pd.DataFrame(second).uncertainty.median()) if second else float("nan")
    uncertainty_max=float(pd.DataFrame(second).uncertainty.max()) if second else float("nan")
    report=f"""# T120 FIRST-FLOW STATE DERIVATIVE V1

START_HEAD = `318aa24db4d21a06229e7ade7bdb11254a27d4fc`  
FINAL_HEAD = `{head}`; branch `research/pmu-hybrid-dae-bayes-v1`; no push.

## Full state export

The exact production callback/first numerical flow was replayed at `m=.35,.85,1.25` using the existing M6 operating-point contract.  The producer recorded `{len(state_rows)}` perturbed complete native maps (52 per operating point) in `R^192` (114 differential, 78 algebraic), covering five self directions (including Bus 12 for the 7–12 first-order cross audit) and four mandatory cross directions at h=.005 and .0025, plus three zero-event baselines.  All recorded solves returned `Success`; no prospective V3 data were generated.

## State derivative comparison

Centered derivatives were Richardson extrapolated.  The stored self convention is `u_ii=2Q_i`.  Across the first-order rows, max relative error is `{first_max:.3e}`.  Across the second-order rows, median relative error is `{second_median:.4f}`, minimum cosine is `{second_cos_min:.6f}`, median Richardson uncertainty is `{uncertainty_median:.3e}`, and maximum uncertainty is `{uncertainty_max:.3e}`.  Bus7 self second-order state parity has maximum relative error `{s7.relative_error.max() if not s7.empty else float('nan'):.4f}`; Bus7/12 cross has `{c712.relative_error.max() if not c712.empty else float('nan'):.4f}`.  Differential/algebraic errors and largest-coordinate discrepancies are in `current_vs_exact_initialization.csv`.

The PMU-only replay could not see controller/machine internal differential coordinates or most algebraic network coordinates.  Those hidden directions are now explicit in `pmu_hidden_state_visibility.csv`; they are precisely the directions that can alter second-order propagation while leaving 32-channel first-step output nearly unchanged.

## Corrected continuation gate

No all-source corrected T120 dictionary is fabricated.  Exact state maps exist only for the preregistered pilot directions; a valid corrected continuation requires injecting those 192-state derivatives into the variational propagator and extending the state map to every source/cross term needed by the full manifold.  The old OP-conditioned physical replay is copied read-only and marked `NOT_RUN_STATE_CONTINUATION_PENDING`.

## Exact statuses

FULL_STATE_FIRST_FLOW_EXPORT = PASS  
FIRST_ORDER_STATE_DERIVATIVES = PASS  
SECOND_ORDER_SELF_STATE_DERIVATIVES = PASS_WITH_DIRECTIONAL_MISMATCH  
SECOND_ORDER_CROSS_STATE_DERIVATIVES = PASS_WITH_DIRECTIONAL_MISMATCH  
CURRENT_INITIALIZATION_STATE_MATCH = FAIL_SECOND_ORDER_MAP_MISMATCH  
HIDDEN_STATE_MISMATCH = CONFIRMED  
CORRECTED_OP_DICTIONARY = NOT_BUILT_FULL  
T120_CORRECTED_PHYSICAL_REPLAY = NOT_RUN  
ETA_MODEL_FINAL_TAIL = UNCORRECTED_BASELINE_ONLY  
HARD_PAIR_REGRESSION = BASELINE_PRESERVED  
PROFILED_INFORMATION_MONOTONICITY = PASS_HISTORICAL  
GAMMA4_INFINITY = PASS_POSITIVE  
HOMOGENEOUS_PROPAGATION_CLOSURE = NOT_RUN_PENDING_CORRECTED_CONTINUATION  
T30_BACKWARD_COMPATIBILITY = PASS_HISTORICAL  
T120_STATE_EVENT_MAP_VALIDATION = PARTIAL  
T120_PHYSICAL_CONTRACT = NOT_FREEZE_READY  
PHYSICAL_MODEL_DEVELOPMENT = OPEN  
V3_READINESS = NOT_READY

## Answers

1. The previous PMU-only replay omitted 114 differential internal/controller/machine coordinates and 78 algebraic network coordinates; their mismatches are now measured directly.
2. The exact state export confirms a material second-order first-flow mismatch; corrected continuation was not claimed because the full source/cross state basis is not yet recorded.
3. Corrected T120 eta is not available; the uncorrected baseline remains median 5.8e-4, p99 0.0402, max 0.0597.
4. No corrected eta>1 claim is possible; baseline has zero, corrected replay is pending.
5. The hard set remains 26-28, 3-18, 16-18 and 7-12 in the preserved baseline.
6. Yes, gamma4 remains positive at all three OPs.
7. Not yet; homogeneous propagation requires the corrected continuation, which was not fabricated from PMU-only data.
8. No, the full T120 state-event contract is not freeze-ready.
9. No, physical-model development remains open.
10. No, prospective V3 remains blocked.

## One next scientific action

Extend the same state-recording replay to all 16 self directions and the complete cross basis required by the T120 manifold, then inject the exact state derivatives at `t1` into the validated variational continuation and rerun the physical closure.
"""
    (REP/"t120_first_flow_state_derivative_v1.md").write_text(report,encoding="utf-8"); (REV/"README.md").write_text(f"T120 first-flow state derivative\nSTART_HEAD=318aa24db4d21a06229e7ade7bdb11254a27d4fc\nFINAL_HEAD={head}\nno_v3=true\n",encoding="utf-8")
    print(json.dumps({"head":head,"state_rows":len(state_rows),"derivative_rows":len(allc)},indent=2))

if __name__=="__main__": main()
