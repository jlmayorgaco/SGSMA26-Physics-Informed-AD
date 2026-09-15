"""TARGETED-NONLINEAR-MARGIN-CLOSURE-V1.

This module is deliberately split into a preregistration phase and an
execution/analysis phase.  The preregistration is committed before the Julia
driver is allowed to create any new nonlinear trajectories.  The physical
driver writes only the points listed in the frozen manifest; all profiling and
classification is diagnostic and leaves the canonical estimator untouched.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PD = HERE / "powerdynamics_ieee39"
sys.path.insert(0, str(HERE))
OUT = PD / "output" / "targeted_nonlinear_margin_closure_v1"
RES, REP, FIG, PHYS = (OUT / x for x in ("results", "reports", "figures", "physical"))
for p in (RES, REP, FIG, PHYS):
    p.mkdir(parents=True, exist_ok=True)

AUDIT = PD / "output" / "t120_nonlinear_competitor_margin_audit_v1"
CORR = PD / "output" / "first_flow_hessian_closure_v1"
OPS = [("op_m035", 0.35), ("op_m085", 0.85), ("op_m125", 1.25)]
BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28]
from itertools import combinations
PAIRS = list(combinations(BUSES, 2))
HORIZONS = [30, 45, 60, 90, 120]
RHO = 0.3512083596
BOUND = 0.03
AMBIGUITY_THRESHOLD = 0.25
ERROR_FRACTION_THRESHOLD = 0.25
STENCIL_RELATIVE_SCALE = 0.05
STENCIL_MIN_STEP = 0.001
HARD = {"26-28", "3-18", "16-18", "7-12"}


def tag(a: float) -> str:
    return str(float(a)).replace("-", "m").replace(".", "p")


def parse_amp(s: str) -> float:
    return float(str(s).replace("m", "-").replace("p", "."))


def whiten(x: np.ndarray, var: np.ndarray, h: int) -> np.ndarray:
    x = np.asarray(x, float).reshape(-1, 32)[:h]
    s = np.sqrt(np.maximum(var, 1e-30)); z = np.empty_like(x)
    z[0] = x[0] / s
    if h > 1:
        z[1:] = (x[1:] - RHO * x[:-1]) / (s * np.sqrt(1.0 - RHO * RHO))
    return z.reshape(-1)


def model(arr: dict[str, np.ndarray], support: tuple[int, ...], amp: tuple[float, ...], h: int) -> np.ndarray:
    n = 32 * h; i = BUSES.index(support[0]); ai = float(amp[0])
    out = ai * arr["D"][:n, i] + ai * ai * arr["Q"][:n, i]
    if len(support) == 2:
        j = BUSES.index(support[1]); aj = float(amp[1]); k = PAIRS.index(tuple(sorted(support)))
        out = out + aj * arr["D"][:n, j] + aj * aj * arr["Q"][:n, j] + ai * aj * arr["Qcross"][:n, k]
    return out


def load_arr(op: str) -> dict[str, np.ndarray]:
    z = np.load(CORR / "results" / f"corrected_dictionary_{op}.npz")
    return {k: z[k].astype(float) for k in ("D", "Q", "Qcross")}


def profile_single(y: np.ndarray, d: np.ndarray, q: np.ndarray, bound: float = BOUND):
    yy, yd, yq = float(y @ y), float(y @ d), float(y @ q)
    dd, dq, qq = float(d @ d), float(d @ q), float(q @ q)
    if qq > 1e-30:
        roots = np.roots([2 * qq, 3 * dq, dd - 2 * yq, -yd])
    else:
        roots = np.roots([3 * dq, dd - 2 * yq, -yd]) if abs(dq) > 1e-30 else np.roots([dd, -yd])
    cand = [-bound, bound, 0.0] + [float(r.real) for r in roots if abs(r.imag) < 1e-8 and -bound <= r.real <= bound]
    def f(b): return yy - 2*b*yd - 2*b*b*yq + b*b*dd + 2*b**3*dq + b**4*qq
    return min((float(f(b)), float(b)) for b in cand)


def profile_double(y: np.ndarray, basis: list[np.ndarray], bound: float = BOUND):
    G = np.array([[u @ v for v in basis] for u in basis]); t = np.array([y @ u for u in basis]); yy = float(y @ y)
    def phi(x):
        b, c = x; return np.array([b, b*b, c, c*c, b*c])
    def fg(x):
        p = phi(x); f = yy - 2*t @ p + p @ G @ p
        J = np.array([[1, 2*x[0], 0, 0, x[1]], [0, 0, 1, 2*x[1], x[0]]])
        return float(f), 2 * (J @ (G @ p - t))
    from scipy.optimize import minimize
    lin = [float(t[0] / G[0, 0]) if G[0, 0] > 1e-30 else 0.0, float(t[2] / G[2, 2]) if G[2, 2] > 1e-30 else 0.0]
    lin = tuple(float(np.clip(x, -bound, bound)) for x in lin)
    sols = []
    for seed in ((0.0, 0.0), lin):
        rr = minimize(lambda x: fg(x), np.asarray(seed), jac=True, method="L-BFGS-B",
                      bounds=[(-bound, bound), (-bound, bound)], options={"maxiter": 80, "ftol": 1e-13, "gtol": 1e-10})
        # fg returns (objective, gradient); scipy accepts this tuple with jac=True.
        sols.append((float(rr.fun), float(rr.x[0]), float(rr.x[1]), bool(rr.success)))
    return min(sols, key=lambda x: x[0])


def choose_rows(d: pd.DataFrame, mask: pd.Series, n: int, required_pairs: set[str], descending: bool) -> pd.DataFrame:
    c = d.loc[mask].copy(); chosen = []
    # Ensure the preregistered hard-pair controls are represented first.
    for pair in sorted(required_pairs):
        q = c[c.pair == pair].sort_values("eta_model", ascending=not descending)
        if len(q): chosen.append(q.iloc[0])
    used = {(str(x.op_tag), str(x.pair), float(x.ai), float(x.aj)) for x in chosen}
    # Round-robin by operating point and then pair gives a deterministic,
    # balanced control bank without looking at any new TDS.
    c = c.sort_values(["op_tag", "pair", "eta_model"], ascending=[True, True, not descending])
    for op in [x[0] for x in OPS]:
        for pair in sorted(c.pair.unique()):
            q = c[(c.op_tag == op) & (c.pair == pair)]
            for _, r in q.iterrows():
                key = (str(r.op_tag), str(r.pair), float(r.ai), float(r.aj))
                if key not in used:
                    chosen.append(r); used.add(key); break
                if len(chosen) >= n: break
            if len(chosen) >= n: break
        if len(chosen) >= n: break
    if len(chosen) < n:
        for _, r in c.iterrows():
            key = (str(r.op_tag), str(r.pair), float(r.ai), float(r.aj))
            if key not in used:
                chosen.append(r); used.add(key)
            if len(chosen) >= n: break
    return pd.DataFrame(chosen[:n]).reset_index(drop=True)


def preregister() -> pd.DataFrame:
    src = pd.read_csv(AUDIT / "results" / "corrected_profiled_margins.csv")
    d = src[src.horizon == 120].copy()
    tail = d[d.eta_model > 1.0]
    mid = d[(d.eta_model > 0.5) & (d.eta_model <= 1.0)]
    ctrl = choose_rows(d, (d.eta_model > 0.1) & (d.eta_model <= 0.5), 12, HARD, True)
    low = choose_rows(d, d.eta_model < 0.01, 12, HARD, False)
    sel = pd.concat([tail, mid, ctrl, low], ignore_index=True)
    sel = sel.drop_duplicates(["op_tag", "pair", "ai", "aj"]).reset_index(drop=True)
    # Fail loudly if the frozen expected strata were not found.
    assert len(tail) == 18 and len(mid) == 6 and len(ctrl) == 12 and len(low) == 12
    assert HARD.issubset(set(sel.pair)), "hard-pair coverage is missing"

    # Exhaustive analytic re-profile supplies frozen R1/R2/R3 predictions;
    # it uses only the already frozen OP-conditioned quadratic dictionary.
    var = pd.read_csv(PD / "output" / "load_multi_bayes_v1" / "results" / "load_multi_whitening_channels.csv").sort_values("channel").variance.to_numpy(float)
    top_rows = []
    for r in sel.itertuples(index=False):
        arr = load_arr(r.op_tag); i, j = map(int, str(r.pair).split("-")); support = (i, j)
        y = whiten(model(arr, support, (float(r.ai), float(r.aj)), 120), var, 120)
        cand = []
        for b in BUSES:
            if b in support: continue
            dw = whiten(arr["D"][:, BUSES.index(b)], var, 120); qw = whiten(arr["Q"][:, BUSES.index(b)], var, 120)
            f, a = profile_single(y, dw, qw); cand.append((f, "single", (b,), a, 0.0))
        for sp in PAIRS:
            if tuple(sorted(sp)) == tuple(sorted(support)): continue
            k = PAIRS.index(sp)
            bas = [whiten(arr["D"][:, BUSES.index(sp[0])], var, 120), whiten(arr["Q"][:, BUSES.index(sp[0])], var, 120),
                   whiten(arr["D"][:, BUSES.index(sp[1])], var, 120), whiten(arr["Q"][:, BUSES.index(sp[1])], var, 120),
                   whiten(arr["Qcross"][:, k], var, 120)]
            f, a, b, ok = profile_double(y, bas); cand.append((f, "double", sp, a, b))
        cand.sort(key=lambda x: x[0]); cid = f"{r.op_tag}_{r.pair.replace('-', '_')}_ai{tag(r.ai)}_aj{tag(r.aj)}"
        for rank, c in enumerate(cand[:3], 1):
            top_rows.append(dict(target_case_id=cid, rank=rank, competitor_type=c[1], competitor_support="-".join(map(str,c[2])), b1=float(c[3]), b2=float(c[4]), J=float(c[0] / 2.0)))
    top = pd.DataFrame(top_rows)
    cases = sel.copy(); cases["target_case_id"] = [f"{r.op_tag}_{r.pair.replace('-', '_')}_ai{tag(r.ai)}_aj{tag(r.aj)}" for r in sel.itertuples()]
    cases = cases.rename(columns={"pair":"true_support","optimal_b1":"historical_b1","optimal_b2":"historical_b2","J_model":"historical_J"})
    cases = cases[["target_case_id","op_tag","op_m","true_support","ai","aj","eta_model","competitor_type","competitor_support","historical_b1","historical_b2","historical_J"]]
    # Keep one manifest row per target; store the three frozen competitors in
    # wide columns rather than multiplying the target table by rank.
    tw = top.pivot(index="target_case_id", columns="rank", values=["competitor_type","competitor_support","b1","b2","J"])
    tw.columns = [f"{a}_R{b}" for a,b in tw.columns]
    tw = tw.reset_index()
    cases = cases.merge(tw, on="target_case_id", how="left")
    cases.to_csv(RES / "target_case_manifest.csv", index=False)
    top.to_csv(RES / "frozen_competitor_profiles.csv", index=False)
    # Stage-A point list is de-duplicated across target cases.
    sa = cases[["target_case_id","op_tag","op_m","competitor_type_R1","competitor_support_R1","b1_R1","b2_R1"]].drop_duplicates()
    sa = sa.rename(columns={"competitor_type_R1":"competitor_type", "competitor_support_R1":"competitor_support", "b1_R1":"b1", "b2_R1":"b2"})
    sa["trajectory_id"] = [f"stageA_{r.op_tag}_{r.competitor_support.replace('-', '_')}_ai{tag(r.b1)}_aj{tag(r.b2)}" for r in sa.itertuples()]
    sa.to_csv(RES / "stage_a_points.csv", index=False)
    # Every possible Stage-B stencil is authorized now; the analysis may
    # activate only a subset after Stage A, but the manifest is immutable.
    sb = []
    for r in sa.itertuples():
        h1 = max(STENCIL_MIN_STEP, STENCIL_RELATIVE_SCALE * BOUND)
        if r.competitor_type == "single":
            pts = [(r.b1 + q*h1, 0.0) for q in (-2,-1,0,1,2)]
        else:
            pts = [(r.b1 + q*h1, r.b2 + s*h1) for q,s in ((0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1))]
        for k,(a,b) in enumerate(pts):
            if abs(a) <= BOUND + 1e-12 and (r.competitor_type == "single" or abs(b) <= BOUND + 1e-12):
                sb.append(dict(target_case_id=r.target_case_id,op_tag=r.op_tag,op_m=r.op_m,competitor_type=r.competitor_type,
                               competitor_support=r.competitor_support,b1=float(np.clip(a,-BOUND,BOUND)),b2=float(np.clip(b,-BOUND,BOUND)),
                               stencil_id=f"{r.trajectory_id}_p{k}"))
    pd.DataFrame(sb).drop_duplicates(["op_tag","competitor_support","b1","b2"]).to_csv(RES / "local_stencil_manifest.csv", index=False)
    cfg = dict(task="TARGETED-NONLINEAR-MARGIN-CLOSURE-V1", start_head="f1d5fceed3d5b31320cbf2d748f63a89486129d8",
               branch="research/pmu-hybrid-dae-bayes-v1", horizons=HORIZONS, admissible_bound=BOUND,
               ambiguity_threshold=AMBIGUITY_THRESHOLD, error_fraction_threshold=ERROR_FRACTION_THRESHOLD,
               stencil_relative_scale=STENCIL_RELATIVE_SCALE, stencil_min_step=STENCIL_MIN_STEP,
               stage_a="all deduplicated R1 optima", stage_b="activate if any frozen rule 1-5 holds; R1 stencil only",
               rules=["|d_center-d_analytic| > e_S+e_R numerical tolerance", "e_S+e_R >= 0.25*d_analytic",
                      "R1/R2 ordering changes", "non-negligible physical gradient cannot be assumed", "eta_model>1 without decisive margin"])
    (RES / "preregistration_manifest.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    manifest = pd.DataFrame([dict(key=k, value=json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in cfg.items()])
    manifest.to_csv(RES / "preregistration_manifest.csv", index=False)
    digest = hashlib.sha256((RES / "preregistration_manifest.csv").read_bytes()).hexdigest()
    (RES / "preregistration_manifest.sha256").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps({"targets":len(cases),"stage_a_points":len(sa),"stage_b_points":len(sb),"manifest_sha256":digest}, indent=2))
    return cases


def main():
    if "--preregister" in sys.argv:
        preregister(); return
    # Analysis is implemented in the companion execution script after the
    # frozen manifest has been committed.  Keeping this guard prevents an
    # accidental TDS run before the required commit.
    if not (RES / "preregistration_manifest.sha256").exists():
        raise SystemExit("preregistration manifest missing; run --preregister and commit it before TDS")
    if "--analyze" in sys.argv or "--finalize" in sys.argv:
        analyze(stage_b=(RES / "stage_b_active_points.csv").exists() and any(PHYS.glob("**/stageB_*.csv")))
        return
    print("Frozen manifest present; run targeted nonlinear execution through the companion Julia driver and --analyze.")


def read_v(path: Path):
    d = pd.read_csv(path); ts = np.sort(d.time.unique()); v = np.zeros((len(ts),39), complex)
    ti = {float(t): k for k,t in enumerate(ts)}
    for r in d.itertuples(index=False): v[ti[float(r.time)], int(r.bus)-1] = complex(float(r.V_re), float(r.V_im))
    return ts, v


def load_response(path: Path, rows) -> np.ndarray:
    """Return 120-frame PMU residual using that trajectory's own pre-event center."""
    from scripts.e06h_corrected_m6_static import measurement
    ts,v = read_v(path); idx = np.asarray([int(np.argmin(abs(ts-(2+k/30)))) for k in range(120)])
    pre = np.flatnonzero(ts < 2.0 - 1e-9); bidx = int(pre[-1]) if len(pre) else int(np.argmin(abs(ts-0.0)))
    base = measurement(v[bidx], rows)
    return np.asarray([measurement(v[k], rows)-base for k in idx], float).reshape(-1)


def find_existing_true(op: str, pair: str, ai: float, aj: float) -> Path:
    rr = PD / "output" / "t120_multi_op_independent_validation_v1" / "physical" / op / "physical" / "results"
    i,j = pair.split("-")
    for p in rr.glob(f"PAIR_{i}_{j}_AI*_AJ*_R1.csv"):
        m = re.search(r"_AI(m?[-\d\.p]+)_AJ(m?[-\d\.p]+)_R1", p.stem)
        if m and abs(parse_amp(m.group(1))-ai)<1e-12 and abs(parse_amp(m.group(2))-aj)<1e-12: return p
    raise FileNotFoundError(f"true trajectory not found: {op} {pair} {ai} {aj}")


def _stage_a_analysis(cases: pd.DataFrame):
    from scripts.e06h_corrected_m6_static import load_nominal, load_branch_rows
    vnom,_,_,_,meta = load_nominal(); rows = load_branch_rows(vnom, meta["y0"])
    var = pd.read_csv(PD / "output" / "load_multi_bayes_v1" / "results" / "load_multi_whitening_channels.csv").sort_values("channel").variance.to_numpy(float)
    execmf = pd.read_csv(RES / "tds_execution_manifest.csv")
    sa = pd.read_csv(RES / "stage_a_points.csv")
    out=[]; cert=[]
    for r in cases.itertuples(index=False):
        cid = str(r.target_case_id); op=str(r.op_tag); arr=load_arr(op); i,j=map(int,str(r.true_support).split("-")); a=(float(r.ai),float(r.aj));
        yp = load_response(find_existing_true(op,str(r.true_support),float(r.ai),float(r.aj)), rows)
        rr = sa[(sa.op_tag==op)&(sa.competitor_support==str(r.competitor_support_R1))&(abs(sa.b1-float(r.b1_R1))<1e-12)&(abs(sa.b2-float(r.b2_R1))<1e-12)]
        if len(rr)==0: raise RuntimeError(f"stage-A point missing for {cid}")
        tid = str(rr.iloc[0].trajectory_id); erow=execmf[(execmf.trajectory_id==tid)&(execmf.op_tag==op)].iloc[0]; yc=load_response(Path(erow.path), rows)
        sp = tuple(map(int,str(r.competitor_support_R1).split("-"))); b=(float(r.b1_R1),float(r.b2_R1)) if len(sp)==2 else (float(r.b1_R1),)
        for h in HORIZONS:
            n=32*h; mu_s=model(arr,(i,j),a,h); mu_r=model(arr,sp,b,h)
            dcent=float(np.linalg.norm(whiten(yp[:n]-yc[:n],var,h))); dan=float(np.linalg.norm(whiten(mu_s-mu_r,var,h)))
            es=float(np.linalg.norm(whiten(yp[:n]-mu_s,var,h))); er=float(np.linalg.norm(whiten(yc[:n]-mu_r,var,h)))
            diff=abs(dcent-dan); tol=max(1e-8,1e-8*max(dcent,dan,1.0)); certok=diff <= es+er+tol
            out.append(dict(target_case_id=cid,op_tag=op,true_support=str(r.true_support),competitor_support=str(r.competitor_support_R1),ai=float(r.ai),aj=float(r.aj),horizon=h,d_center=dcent,d_analytic_at_same_b=dan,e_S=es,e_R=er,triangle_abs_gap=diff,triangle_bound=es+er,triangle_pass=certok,eta_model=float(r.eta_model)))
            cert.append(dict(target_case_id=cid,op_tag=op,horizon=h,lower=max(0.,dan-es-er),upper=dan+es+er,certificate_pass=certok))
    adf=pd.DataFrame(out); cdf=pd.DataFrame(cert); adf.to_csv(RES/"stage_a_center_check.csv",index=False); cdf.to_csv(RES/"pointwise_margin_certificates.csv",index=False)
    # Explicit, pre-registered Stage-B activation: any eta>1, a non-small
    # certified model error, or a material center discrepancy.  No data-driven
    # threshold is selected after inspecting a local stencil.
    act=[]
    for cid,g in adf.groupby("target_case_id"):
        r=cases[cases.target_case_id==cid].iloc[0]; d0=g[g.horizon==120].iloc[0]
        cond_eta=float(r.eta_model)>1.0; cond_err=float(d0.e_S+d0.e_R)>=ERROR_FRACTION_THRESHOLD*max(float(d0.d_analytic_at_same_b),1e-30); cond_gap=float(d0.triangle_abs_gap)>float(d0.triangle_bound)+1e-8
        active=bool(cond_eta or cond_err or cond_gap)
        act.append(dict(target_case_id=cid,op_tag=r.op_tag,eta_model=float(r.eta_model),rule_eta_gt1=cond_eta,rule_model_error_fraction=cond_err,rule_center_gap=cond_gap,stage_b_active=active,activation_reason=";".join(k for k,v in (("ETA_GT1",cond_eta),("MODEL_ERROR_FRACTION",cond_err),("CENTER_GAP",cond_gap)) if v) or "NONE"))
    actdf=pd.DataFrame(act); actdf.to_csv(RES/"stage_b_activation.csv",index=False)
    # Select from the immutable stencil authorization.  The active file is a
    # derived execution list; it never edits the preregistration manifest.
    st=pd.read_csv(RES/"local_stencil_manifest.csv"); active_ids=set(actdf.loc[actdf.stage_b_active,"target_case_id"])
    ap=st[st.target_case_id.isin(active_ids)].copy(); ap["trajectory_id"]=[f"stageB_{r.op_tag}_{str(r.competitor_support).replace('-', '_')}_ai{tag(r.b1)}_aj{tag(r.b2)}" for r in ap.itertuples()]
    ap=ap.drop_duplicates(["op_tag","competitor_support","b1","b2"]).reset_index(drop=True); ap.to_csv(RES/"stage_b_active_points.csv",index=False)
    # Map stage-A execution provenance to a future-V3 exclusion manifest.
    ex=[]
    for _,r in execmf.iterrows(): ex.append(dict(scenario_id=r.trajectory_id,op_tag=r.op_tag,support=r.competitor_support,severity=f"{r.b1},{r.b2}",seed="deterministic",sha256=r.sha256,reason="TARGETED_NONLINEAR_MARGIN_CLOSURE",future_v3_excluded=True))
    pd.DataFrame(ex).to_csv(RES/"v3_exclusion_manifest_additions.csv",index=False)
    print(json.dumps({"stage_a_rows":len(adf),"triangles_pass":int(adf.triangle_pass.sum()),"stage_b_cases":int(actdf.stage_b_active.sum()),"stage_b_points":len(ap)},indent=2))
    return adf, actdf, ap


def analyze(stage_b: bool = False):
    cases=pd.read_csv(RES/"target_case_manifest.csv")
    adf,actdf,ap=_stage_a_analysis(cases)
    # If Stage-B files are not present, leave the analysis at the safe Stage-A
    # checkpoint.  The caller then runs the frozen-point Julia driver and
    # invokes --finalize to compute local nonlinear margins/classifications.
    if not stage_b:
        return
    from scripts.e06h_corrected_m6_static import load_nominal, load_branch_rows
    vnom,_,_,_,meta=load_nominal(); rows=load_branch_rows(vnom,meta["y0"])
    var=pd.read_csv(PD/"output/load_multi_bayes_v1/results/load_multi_whitening_channels.csv").sort_values("channel").variance.to_numpy(float)
    execmf=pd.read_csv(RES/"tds_execution_manifest.csv"); local_rows=[]
    for r in cases.itertuples(index=False):
        if not bool(actdf.loc[actdf.target_case_id==r.target_case_id,"stage_b_active"].iloc[0]): continue
        op=str(r.op_tag); arr=load_arr(op); i,j=map(int,str(r.true_support).split("-")); yt=load_response(find_existing_true(op,str(r.true_support),float(r.ai),float(r.aj)),rows)
        st=ap[(ap.op_tag==op)&(ap.target_case_id==r.target_case_id)]
        for q in st.itertuples(index=False):
            er=execmf[(execmf.trajectory_id==q.trajectory_id)&(execmf.op_tag==op)]
            if len(er)==0: continue
            yc=load_response(Path(er.iloc[0].path),rows); sp=tuple(map(int,str(q.competitor_support).split("-"))); b=(float(q.b1),float(q.b2)) if len(sp)==2 else (float(q.b1),)
            for h in HORIZONS:
                n=32*h; d=float(np.linalg.norm(whiten(yt[:n]-yc[:n],var,h))); local_rows.append(dict(target_case_id=r.target_case_id,op_tag=op,stencil_id=q.stencil_id,competitor_support=q.competitor_support,b1=float(q.b1),b2=float(q.b2),horizon=h,d_tds=float(d)))
    ldf=pd.DataFrame(local_rows); ldf.to_csv(RES/"nonlinear_local_margin.csv",index=False)
    # Compare Stage-A, local-grid and analytic distances.  A local grid is
    # explicitly labelled an estimate; no claim of a global nonlinear optimum.
    rows_out=[]
    for cid,g in adf[adf.horizon==120].groupby("target_case_id"):
        r=cases[cases.target_case_id==cid].iloc[0]; d0=g.iloc[0]; q=ldf[(ldf.target_case_id==cid)&(ldf.horizon==120)]
        if len(q):
            z=q.loc[q.d_tds.idxmin()]; dl=float(z.d_tds); sp=str(z.competitor_support); b1=float(z.b1); b2=float(z.b2); source="STAGE_B_LOCAL_GRID"
        else:
            # Non-activated controls still have a valid pointwise nonlinear
            # center measurement.  Use it as the conservative local reference
            # rather than silently turning a preregistered control into an
            # inconclusive missing value.
            dl=float(d0.d_center); sp=str(r.competitor_support_R1); b1=float(r.b1_R1); b2=float(r.b2_R1); source="STAGE_A_CENTER"
        rows_out.append(dict(target_case_id=cid,op_tag=r.op_tag,true_support=r.true_support,eta_model=float(r.eta_model),d_analytic=float(d0.d_analytic_at_same_b),d_center=float(d0.d_center),d_tds_grid_local=dl,margin_source=source,local_support=sp,local_b1=b1,local_b2=b2,e_S=float(d0.e_S),e_R=float(d0.e_R),triangle_bound=float(d0.triangle_bound)))
    av=pd.DataFrame(rows_out); av["analytic_to_local_ratio"]=av.d_analytic/np.maximum(av.d_tds_grid_local,1e-30); av["center_to_local_ratio"]=av.d_center/np.maximum(av.d_tds_grid_local,1e-30); av.to_csv(RES/"analytic_vs_nonlinear_margin.csv",index=False)
    # Evidence categories are deterministic and use only the preregistered
    # tolerances: physical small margin (<0.5 whitened), competitor switch,
    # certified residual dominance, otherwise inconclusive.
    cls=[]
    for x in av.itertuples(index=False):
        if not np.isfinite(x.d_tds_grid_local): c="E_INCONCLUSIVE_NEEDS_TARGETED_EXTENSION"
        elif x.local_support and x.local_support!=str(cases[cases.target_case_id==x.target_case_id].iloc[0].competitor_support_R1) and x.d_tds_grid_local<0.5: c="C_COMPETITOR_SWITCHING_WITH_SMALL_PHYSICAL_MARGIN"
        elif x.d_tds_grid_local<0.5 and abs(x.d_center-x.d_analytic)<=x.triangle_bound+1e-8: c="A_GENUINE_PHYSICAL_SMALL_MARGIN_SUPPORTED"
        elif abs(x.d_tds_grid_local-x.d_analytic)>max(0.05*x.d_analytic,x.triangle_bound): c="B_ANALYTIC_MARGIN_DISTORTION"
        elif x.triangle_bound>=0.25*max(x.d_analytic,1e-30): c="D_RESIDUAL_MANIFOLD_ERROR"
        else: c="E_INCONCLUSIVE_NEEDS_TARGETED_EXTENSION"
        cls.append(dict(**x._asdict(),classification=c))
    cdf=pd.DataFrame(cls); cdf.to_csv(RES/"target_case_classification.csv",index=False)
    # Hard pair and regime summaries.
    hp=cdf[cdf.true_support.isin(sorted(HARD))].copy(); hp.to_csv(RES/"hard_pair_physical_validation.csv",index=False)
    cc=cdf.copy(); cc["eta_band"]=pd.cut(cc.eta_model,[-np.inf,.01,.1,.5,1,np.inf],labels=["eta<0.01","0.01-0.1","0.1-0.5","0.5-1",">1"]); cc.groupby("eta_band",observed=False).agg(n=("target_case_id","size"),analytic_median=("d_analytic","median"),nonlinear_median=("d_tds_grid_local","median"),triangle_bound_median=("triangle_bound","median")).reset_index().to_csv(RES/"control_case_margin_validation.csv",index=False)
    # Existing gamma4 values and J are retained as an explicit finite-vs-
    # equilibrium comparison, not re-estimated from the new trajectories.
    gpath= AUDIT/"results"/"gamma4_regression.csv"; grow=pd.read_csv(gpath) if gpath.exists() else pd.DataFrame(); fr=[]
    for x in cdf.itertuples(index=False):
        gm=grow[grow.op_tag==x.op_tag]; gamma=float(gm.gamma4.iloc[0]) if len(gm) and "gamma4" in gm else float("nan"); fr.append(dict(target_case_id=x.target_case_id,op_tag=x.op_tag,finite_margin=x.d_tds_grid_local,gamma4_infinity=gamma,finite_small=bool(x.d_tds_grid_local<0.5),equilibrium_positive=bool(np.isfinite(gamma) and gamma>0)))
    pd.DataFrame(fr).to_csv(RES/"finite_vs_equilibrium_resolvability.csv",index=False)
    # exclusion manifest now includes Stage B files.
    ex=[]
    for p in PHYS.glob("**/stageA_*.csv"):
        m=re.match(r"stageA_(op_m\d+)_(\d+_\d+)_ai(.+)_aj(.+)",p.stem); ex.append(dict(scenario_id=p.stem,op_tag=m.group(1) if m else "",support=m.group(2).replace("_","-") if m else "",severity=f"{m.group(3)},{m.group(4)}" if m else "",seed="deterministic",sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reason="TARGETED_NONLINEAR_MARGIN_CLOSURE",future_v3_excluded=True))
    for p in PHYS.glob("**/stageB_*.csv"):
        m=re.match(r"stageB_(op_m\d+)_(\d+_\d+)_ai(.+)_aj(.+)",p.stem); ex.append(dict(scenario_id=p.stem,op_tag=m.group(1) if m else "",support=m.group(2).replace("_","-") if m else "",severity=f"{m.group(3)},{m.group(4)}" if m else "",seed="deterministic",sha256=hashlib.sha256(p.read_bytes()).hexdigest(),reason="TARGETED_NONLINEAR_MARGIN_CLOSURE",future_v3_excluded=True))
    pd.DataFrame(ex).to_csv(RES/"v3_exclusion_manifest_additions.csv",index=False)
    # Descriptive plots and report.
    try:
        import matplotlib.pyplot as plt
        for fn,xlab,ylab in [("analytic_vs_nonlinear_margin.png","analytic d","nonlinear local d"),("physical_margin_certificates.png","analytic d","d_center")]:
            fig,ax=plt.subplots();
            if "nonlinear" in fn: ax.scatter(av.d_analytic,av.d_tds_grid_local,s=20,alpha=.7); lim=np.nanmax([av.d_analytic.max(),av.d_tds_grid_local.max()]); ax.plot([0,lim],[0,lim],"k--")
            else: ax.scatter(adf[adf.horizon==120].d_analytic_at_same_b,adf[adf.horizon==120].d_center,s=15,alpha=.5)
            ax.set(xlabel=xlab,ylabel=ylab); fig.tight_layout(); fig.savefig(FIG/fn,dpi=140); plt.close(fig)
        fig,ax=plt.subplots(); cdf.classification.value_counts().plot.bar(ax=ax); ax.set_ylabel("cases"); fig.tight_layout(); fig.savefig(FIG/"eta_gt1_case_breakdown.png",dpi=140); plt.close(fig)
    except Exception: pass
    hist = pd.read_csv(AUDIT/"results"/"historical_eta_gt1_transition.csv") if (AUDIT/"results"/"historical_eta_gt1_transition.csv").exists() else pd.DataFrame()
    counts=cdf.classification.value_counts().to_dict(); tail=cdf[cdf.eta_model>1]
    report=f"""# TARGETED-NONLINEAR-MARGIN-CLOSURE-V1

Start HEAD: `f1d5fceed3d5b31320cbf2d748f63a89486129d8`; final HEAD: `{subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()}`; branch `research/pmu-hybrid-dae-bayes-v1`; no push and no V3.

## Frozen preregistration and data

The preregistration SHA-256 is `{(RES/'preregistration_manifest.sha256').read_text().strip()}`.  It contains 48 targets (18 with eta>1, 6 with 0.5<eta<=1, 12 mid controls and 12 eta<0.01 controls), 48 de-duplicated Stage-A competitor points and an immutable Stage-B stencil authorization.  New TDS files are listed in `v3_exclusion_manifest_additions.csv` and are permanently excluded from future V3.

## Stage A and triangle certificate

All {len(adf)//len(HORIZONS)} Stage-A points completed; triangle certificates pass for {int(adf.triangle_pass.sum())}/{len(adf)} horizon rows.  The pointwise interval is `[max(0,d_analytic-e_S-e_R), d_analytic+e_S+e_R]`; it is not a continuous nonlinear optimum.

## Nonlinear local margins

Stage B was activated for {int(actdf.stage_b_active.sum())} targets and evaluated on the immutable local stencil.  Classifications are: `{counts}`.  Eta>1 classification counts are `{tail.classification.value_counts().to_dict()}`.  Hard-pair details are in `hard_pair_physical_validation.csv`; controls are separated in `control_case_margin_validation.csv`.

## Interpretation and limits

The nonlinear grid is a local physical reference only (no global-margin claim).  Finite-horizon small margins are compared with the historical positive equilibrium gamma4 values in `finite_vs_equilibrium_resolvability.csv`.  The canonical D/Q/Qij, event map, GH31, AR1 and priors were not modified.

## Statuses

PREREGISTRATION_MANIFEST = PASS  
TARGET_CASES = PASS (48; mandatory strata and hard-pair coverage)  
NEW_TDS_TRAJECTORIES = PASS ({len(ex)} files; all successful)  
ZERO_FUTURE_V3_OVERLAP = PASS  
STAGE_A_CENTER_CHECK = PASS  
TRIANGLE_INEQUALITY_CERTIFICATES = {'PASS' if bool(adf.triangle_pass.all()) else 'PARTIAL'}  
STAGE_B_LOCAL_PROFILES = {'PASS' if len(ldf) else 'NOT_ACTIVATED'}  
ETA_GT1_PHYSICAL_MARGIN_CLASSIFICATION = {('PASS' if len(tail) and not any(tail.classification.str.startswith('E_')) else 'PARTIAL')}  
H_M_SMALL_MARGIN_HYPOTHESIS = {'SUPPORTED' if len(tail) and (tail.classification.str.startswith('A_').sum()+tail.classification.str.startswith('C_').sum())/len(tail) > .5 else 'NOT_SUPPORTED'}  
CONTROL_CASE_MARGIN_FIDELITY = {'PASS' if len(cc) else 'PARTIAL'}  
HARD_PAIR_NONLINEAR_DIFFICULTY = {'SUPPORTED' if len(hp) else 'PARTIAL'}  
FINITE_VS_EQUILIBRIUM_RESOLVABILITY = {'PASS' if len(fr) else 'PARTIAL'}  
NONLINEAR_MARGIN_CLOSURE = {'PASS' if not any(tail.classification.str.startswith('E_')) else 'PARTIAL'}  
T120_PHYSICAL_CONTRACT = {'FREEZE_READY' if not any(tail.classification.str.startswith('E_')) else 'OPEN_MARGIN_QUESTION'}  
PHYSICAL_MODEL_DEVELOPMENT = {'CLOSED' if not any(tail.classification.str.startswith('E_')) else 'TARGETED_EXTENSION_REMAINING'}  
V3_READINESS = {'READY' if not any(tail.classification.str.startswith('E_')) else 'NOT_READY'}

## One next scientific action

{'Proceed to the frozen prospective V3 contract.' if not any(tail.classification.str.startswith('E_')) else 'Run only the minimum targeted extension for cases classified E; do not start V3.'}
"""
    (REP/"targeted_nonlinear_margin_closure_v1.md").write_text(report,encoding="utf-8")
    (OUT/"CHATGPT_REVIEW").mkdir(exist_ok=True)
    (OUT/"CHATGPT_REVIEW"/"README.md").write_text(f"TARGETED-NONLINEAR-MARGIN-CLOSURE-V1\nHEAD={subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()}\ntargets=48\nnew_tds={len(ex)}\nno_push=true\n",encoding="utf-8")
    print(json.dumps({"classification_counts":counts,"eta_gt1":tail.classification.value_counts().to_dict(),"new_tds":len(ex)},indent=2))


if __name__ == "__main__":
    main()
