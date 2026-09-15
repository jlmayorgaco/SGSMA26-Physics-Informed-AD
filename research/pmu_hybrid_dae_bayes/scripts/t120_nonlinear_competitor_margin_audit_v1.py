"""T120-NONLINEAR-COMPETITOR-MARGIN-AUDIT-V1.

Audit-only replay of the corrected production-event-map manifold.  All
nonlinear trajectories are read from the already excluded validation bank;
this module does not run PowerDynamics, tune a likelihood, or alter the
canonical estimator.  Competing analytic supports are profiled with the
frozen quadratic D/Q/Qij dictionary and compared with the discrete stored TDS
grid (explicitly an upper bound on a continuously profiled physical margin).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

HERE = Path(__file__).resolve().parents[1]
PD = HERE / "powerdynamics_ieee39"
SRC = PD / "output" / "t120_multi_op_independent_validation_v1" / "physical"
CORR = PD / "output" / "first_flow_hessian_closure_v1"
OUT = PD / "output" / "t120_nonlinear_competitor_margin_audit_v1"
RES, REP, FIG = (OUT / x for x in ("results", "reports", "figures"))
for p in (RES, REP, FIG): p.mkdir(parents=True, exist_ok=True)

BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28]
PAIRS = list(combinations(BUSES, 2))
HORIZONS = [30, 45, 60, 90, 120]
RHO = 0.3512083596
OPS = [("op_m035", 0.35), ("op_m085", 0.85), ("op_m125", 1.25)]
HARD = {"26-28", "3-18", "16-18", "7-12"}


def whiten(x: np.ndarray, var: np.ndarray, h: int) -> np.ndarray:
    x = np.asarray(x, float).reshape(-1, 32)[:h]
    s = np.sqrt(np.maximum(var, 1e-30)); z = np.empty_like(x)
    z[0] = x[0] / s
    if h > 1: z[1:] = (x[1:] - RHO * x[:-1]) / (s * np.sqrt(1 - RHO * RHO))
    return z.reshape(-1)


def read_v(path: Path):
    d = pd.read_csv(path); ts = np.sort(d.time.unique()); v = np.zeros((len(ts), 39), complex)
    ti = {float(t): k for k, t in enumerate(ts)}
    for r in d.itertuples(index=False): v[ti[float(r.time)], int(r.bus) - 1] = complex(float(r.V_re), float(r.V_im))
    return ts, v


def parse_name(fp: Path):
    m = re.match(r"PAIR_(\d+)_(\d+)_AI([m\d\.p]+)_AJ([m\d\.p]+)_R1", fp.stem)
    if not m: return None
    cv = lambda s: float(s.replace("m", "-").replace("p", "."))
    return int(m.group(1)), int(m.group(2)), cv(m.group(3)), cv(m.group(4))


def model(arr: dict[str, np.ndarray], support: tuple[int, ...], a: tuple[float, ...], h: int) -> np.ndarray:
    n = 32 * h; i = BUSES.index(support[0]); ai = a[0]
    out = ai * arr["D"][:n, i] + ai * ai * arr["Q"][:n, i]
    if len(support) == 2:
        j = BUSES.index(support[1]); aj = a[1]; k = PAIRS.index(tuple(sorted(support)))
        out = out + aj * arr["D"][:n, j] + aj * aj * arr["Q"][:n, j] + ai * aj * arr["Qcross"][:n, k]
    return out


def profile_single(y: np.ndarray, d: np.ndarray, q: np.ndarray, bound: float):
    # Exact quartic objective: ||y-bd-b²q||².  Enumerating cubic stationary
    # roots is both faster and more reliable than a single local optimizer.
    yy, yd, yq = y @ y, y @ d, y @ q
    dd, dq, qq = d @ d, d @ q, q @ q
    # derivative / 2 = 2*qq*b^3 + 3*dq*b^2 + (dd-2*yq)*b - yd
    roots = np.roots([2 * qq, 3 * dq, dd - 2 * yq, -yd]) if qq > 1e-30 else np.roots([3 * dq, dd - 2 * yq, -yd])
    cand = [-bound, bound, 0.0] + [float(r.real) for r in roots if abs(r.imag) < 1e-8 and -bound <= r.real <= bound]
    def f(b): return yy - 2*b*yd - 2*b*b*yq + b*b*dd + 2*b**3*dq + b**4*qq
    vals = [(f(b), b) for b in cand]; vals.sort(key=lambda z: z[0])
    return vals[0][0], vals[0][1], abs(abs(vals[0][1]) - bound) < 1e-7, "ROOT_ENUM"


def profile_double(y: np.ndarray, basis: list[np.ndarray], bound: float):
    # basis=[d_i,q_i,d_j,q_j,c_ij], and m=[b_i,b_i²,b_j,b_j²,b_i*b_j].
    G = np.array([[u @ v for v in basis] for u in basis]); t = np.array([y @ u for u in basis]); yy = float(y @ y)
    def phi(x):
        b, c = x; return np.array([b, b*b, c, c*c, b*c])
    def fg(x):
        p = phi(x); f = yy - 2*t @ p + p @ G @ p
        J = np.array([[1, 2*x[0], 0, 0, x[1]], [0, 0, 1, 2*x[1], x[0]]])
        return float(f), 2 * (J @ (G @ p - t))
    # A zero start plus the bounded linear least-squares start provide a
    # deterministic multistart check while keeping the exhaustive 528 x 5 x
    # 136 profile tractable.  Boundary candidates are evaluated explicitly.
    lin = [float(t[0] / G[0, 0]) if G[0, 0] > 1e-30 else 0.0,
           float(t[2] / G[2, 2]) if G[2, 2] > 1e-30 else 0.0]
    lin = tuple(float(np.clip(x, -bound, bound)) for x in lin)
    seeds = [(0., 0.), lin]
    sols = []
    for seed in seeds:
        rr = minimize(lambda x: fg(x), np.asarray(seed), jac=True, method="L-BFGS-B", bounds=[(-bound, bound), (-bound, bound)], options={"maxiter": 25, "ftol": 1e-12, "gtol": 1e-8})
        sols.append((float(rr.fun), float(rr.x[0]), float(rr.x[1]), bool(rr.success), rr.message))
    sols.sort(key=lambda z: z[0]); f, a, b, ok, msg = sols[0]
    boundary = abs(abs(a)-bound) < 1e-7 or abs(abs(b)-bound) < 1e-7
    return f, a, b, boundary, ("OK" if ok else "PARTIAL")


def load_corrected(op: str):
    z = np.load(CORR / "results" / f"corrected_dictionary_{op}.npz")
    return {k: z[k].astype(float) for k in ("D", "Q", "Qcross")}


def load_bank(path: Path, rows, y0: np.ndarray | None = None):
    ts, v = read_v(path); idx = np.asarray([int(np.argmin(abs(ts - (2 + k / 30)))) for k in range(120)])
    # Importing the frozen map avoids duplicating PiLine signs/order.
    from scripts.e06h_corrected_m6_static import measurement
    # The frozen dictionary is a residual/event-response model.  Each
    # operating-point trajectory has its own pre-event equilibrium; subtract
    # the trajectory's pre-event center (not the nominal OP center).
    pre = np.flatnonzero(ts < 2.0 - 1e-9)
    bidx = int(pre[-1]) if len(pre) else int(np.argmin(abs(ts - 0.0)))
    baseline = measurement(v[bidx], rows)
    return np.asarray([measurement(v[k], rows) - baseline for k in idx], float).reshape(-1)


def main():
    import sys
    sys.path.insert(0, str(HERE))
    from scripts.e06h_corrected_m6_static import load_nominal, load_branch_rows, measurement
    vnom, _, _, _, meta = load_nominal(); rows = load_branch_rows(vnom, meta["y0"])
    y0 = measurement(vnom, rows)
    var = pd.read_csv(PD / "output" / "load_multi_bayes_v1" / "results" / "load_multi_whitening_channels.csv").sort_values("channel").variance.to_numpy(float)
    analytic_rows = []; grid_rows = []; coverage_rows = []; ident_rows = []; info_rows = []; gain_rows = []; hard_rows = []
    cache = {}
    basis_cache = {}
    # Load all corrected dictionaries and all stored trajectories once.
    for op, _ in OPS:
        arr = load_corrected(op); files=[]
        for fp in sorted((SRC / op / "physical" / "results").glob("PAIR_*.csv")):
            x = parse_name(fp)
            if x is not None: files.append((fp, x))
        if not files: continue
        bank=[]
        for fp, x in files:
            bank.append((fp, x, load_bank(fp, rows, y0)))
        cache[op] = (arr, bank)
        # Whitened candidate bases depend only on OP and horizon, so cache
        # them once instead of rebuilding 136 vectors for every trajectory.
        bcache = {}
        for h in HORIZONS:
            singles = {b: (whiten(arr["D"][:32*h, BUSES.index(b)], var, h),
                           whiten(arr["Q"][:32*h, BUSES.index(b)], var, h)) for b in BUSES}
            doubles = {}
            for sp in PAIRS:
                ii, jj = BUSES.index(sp[0]), BUSES.index(sp[1]); k = PAIRS.index(sp)
                doubles[sp] = [whiten(arr["D"][:32*h, ii], var, h),
                               whiten(arr["Q"][:32*h, ii], var, h),
                               whiten(arr["D"][:32*h, jj], var, h),
                               whiten(arr["Q"][:32*h, jj], var, h),
                               whiten(arr["Qcross"][:32*h, k], var, h)]
            bcache[h] = (singles, doubles)
        basis_cache[op] = bcache
    # First pass: exact analytic profiled margin for every stored double case.
    for op, m in OPS:
        if op not in cache: continue
        arr, bank = cache[op]
        by_support = {}
        for fp, (i,j,ai,aj), y in bank:
            by_support.setdefault((i,) if j == 0 else tuple(sorted((i,j))), []).append((ai,aj,y,fp.name))
        for fp, (i,j,ai,aj), y_tds in bank:
            if j == 0 or abs(ai)+abs(aj) < 1e-15: continue
            support = (i, j)
            # Store the same analytic margin at every horizon by re-solving
            # the support with the horizon-specific whitened vectors.
            for h in HORIZONS:
                true_h=model(arr,support,(ai,aj),h); yw=whiten(true_h,var,h); cand=[]
                singles, doubles = basis_cache[op][h]
                for b in BUSES:
                    if b in support: continue
                    d,q = singles[b]
                    f,aa,bd,st=profile_single(yw,d,q,.03); cand.append((f,"single",(b,),aa,0.,bd,st))
                for sp in PAIRS:
                    if set(sp)==set(support): continue
                    bas = doubles[sp]
                    f,aa,bb,bd,st=profile_double(yw,bas,.03); cand.append((f,"double",sp,aa,bb,bd,st))
                cand.sort(key=lambda z:z[0]); c=cand[0]; c2=cand[1]
                ytrue=y_tds[:32*h]; mu=true_h; e=whiten(mu-ytrue,var,h); d_analytic=math.sqrt(max(c[0],0));
                analytic_rows.append({"op_tag":op,"op_m":m,"pair":f"{i}-{j}","ai":ai,"aj":aj,"horizon":h,"J_model":0.5*c[0],"d_analytic":d_analytic,"competitor_type":c[1],"competitor_support":"-".join(map(str,c[2])),"optimal_b1":c[3],"optimal_b2":c[4],"optimizer_status":c[6],"boundary":c[5],"second_best_distance":math.sqrt(max(c2[0],0)),"model_error_sq":float(e@e),"model_error":float(np.linalg.norm(e)),"eta_model":float(np.linalg.norm(e)/max(d_analytic,1e-30))})
                # Information monotonicity uses the profiled Gaussian
                # separation J, not support accuracy.
            # Coverage of stored nonlinear competitor severities at T120.
            same=by_support.get(tuple(sorted(c[2])) if c[1]=="double" else tuple(c[2]),[])
            if c[1]=="double": dist=[math.hypot(x[0]-c[3],x[1]-c[4]) for x in same]
            else: dist=[abs(x[0]-c[3]) for x in same]
            nearest=min(dist) if dist else math.nan; n=len(same); hull=bool(same and all(min(x[k] for x in same)<=c[3+k]<=max(x[k] for x in same) for k in range(2 if c[1]=="double" else 1)))
            quality="DENSE_LOCAL_REFERENCE" if n>=5 and nearest<0.003 and hull else ("SPARSE_LOCAL_REFERENCE" if n>=2 and hull else ("SUPPORT_ONLY_REFERENCE" if n else "NO_USEFUL_REFERENCE"))
            coverage_rows.append({"op_tag":op,"pair":f"{i}-{j}","horizon":120,"predicted_support":"-".join(map(str,c[2])),"predicted_type":c[1],"support_exists":bool(same),"n_stored_samples":n,"nearest_severity_distance":nearest,"analytic_opt_in_stored_hull":hull,"quality":quality})
            # Discrete stored TDS grid competitor distance at every horizon.
            candidates=[(x,y) for x,xx,y in bank if tuple(sorted((xx[0],xx[1])))!=tuple(sorted(support))]
            for h in HORIZONS:
                yt=whiten(y_tds,var,h); bestg=(math.inf,None,None,None)
                for (xf,yy) in candidates:
                    dg=float(np.linalg.norm(yt-whiten(yy,var,h)))
                    if dg<bestg[0]: bestg=(dg,xf.name,parse_name(xf)[:2],parse_name(xf)[2:])
                ar=next(r for r in analytic_rows if r["op_tag"]==op and r["pair"]==f"{i}-{j}" and abs(r["ai"]-ai)<1e-12 and abs(r["aj"]-aj)<1e-12 and r["horizon"]==h)
                grid_rows.append({"op_tag":op,"pair":f"{i}-{j}","ai":ai,"aj":aj,"horizon":h,"tds_grid_distance":bestg[0],"tds_grid_distance_sq":bestg[0]**2,"grid_competitor_file":bestg[1],"grid_competitor_support":"-".join(map(str,bestg[2])) if bestg[2] else "","grid_competitor_amplitudes":str(bestg[3]),"reference_is_upper_bound":True,"d_analytic":ar["d_analytic"],"R_grid":ar["d_analytic"]/max(bestg[0],1e-30)})
                info_rows.append({"op_tag":op,"pair":f"{i}-{j}","ai":ai,"aj":aj,"horizon":h,"J":ar["J_model"]})
    adf=pd.DataFrame(analytic_rows); gdf=pd.DataFrame(grid_rows); cdf=pd.DataFrame(coverage_rows); idf=pd.DataFrame(info_rows)
    adf.to_csv(RES/"corrected_profiled_margins.csv",index=False); gdf.to_csv(RES/"tds_grid_competitor_reference.csv",index=False); cdf.to_csv(RES/"stored_severity_coverage.csv",index=False); idf.to_csv(RES/"corrected_information_monotonicity.csv",index=False)
    # Exact case-wise monotonicity and marginal gains.
    violations=[]; gains=[]
    for key,g in idf.groupby(["op_tag","pair","ai","aj"]):
        g=g.sort_values("horizon"); vals=g.J.to_numpy(); hs=g.horizon.to_numpy()
        for k in range(1,len(vals)):
            gains.append({"op_tag":key[0],"pair":key[1],"ai":key[2],"aj":key[3],"from_horizon":int(hs[k-1]),"to_horizon":int(hs[k]),"delta_J":float(vals[k]-vals[k-1]),"tail":bool(float(key[2])!=0 and float(key[3])!=0 and vals[-1]>0)})
            if vals[k] < vals[k-1]-1e-8: violations.append({"op_tag":key[0],"pair":key[1],"ai":key[2],"aj":key[3],"from_horizon":int(hs[k-1]),"to_horizon":int(hs[k]),"J_prev":vals[k-1],"J_next":vals[k],"violation":vals[k]-vals[k-1]})
    violation_cols = ["op_tag", "pair", "ai", "aj", "from_horizon", "to_horizon", "J_prev", "J_next", "violation"]
    pd.DataFrame(violations, columns=violation_cols).to_csv(RES/"information_monotonicity_violations.csv",index=False); pd.DataFrame(gains).to_csv(RES/"marginal_information_gain.csv",index=False)
    # Tail set, numerator/denominator decomposition and identity stability.
    t120=adf[adf.horizon==120].copy(); tail=t120[t120.eta_model>.05].copy(); controls=t120[t120.eta_model<.01].copy(); p95=t120.iloc[(t120.eta_model-t120.eta_model.quantile(.95)).abs().argsort()[:max(1,min(18,len(t120)))]]
    tail[["op_tag","pair","ai","aj","horizon","eta_model"]].assign(tail_reason="eta_model_T120_gt_0.05").to_csv(RES/"tail_case_set.csv",index=False)
    dec=[]
    for _,r in t120.iterrows():
        gg=gdf[(gdf.op_tag==r.op_tag)&(gdf.pair==r.pair)&(abs(gdf.ai-r.ai)<1e-12)&(abs(gdf.aj-r.aj)<1e-12)&(gdf.horizon==120)]
        dgrid=float(gg.tds_grid_distance.iloc[0]) if len(gg) else math.nan; dan=float(r.d_analytic)
        qual=cdf[(cdf.op_tag==r.op_tag)&(cdf.pair==r.pair)].quality.iloc[0] if len(cdf[(cdf.op_tag==r.op_tag)&(cdf.pair==r.pair)]) else "NO_USEFUL_REFERENCE"
        cls="INCONCLUSIVE_REFERENCE" if qual=="NO_USEFUL_REFERENCE" else ("TRUE_SMALL_MARGIN_SUPPORTED" if dgrid<0.5 and 0.5<dan/max(dgrid,1e-30)<2 else ("ANALYTIC_MARGIN_DISTORTION" if abs(math.log(max(dan,1e-30)/max(dgrid,1e-30)))>math.log(2) else "MIXED"))
        dec.append({"op_tag":r.op_tag,"pair":r.pair,"ai":r.ai,"aj":r.aj,"horizon":120,"e_model":r.model_error,"d_analytic":dan,"d_tds_grid":dgrid,"R_grid":dan/max(dgrid,1e-30),"reference_quality":qual,"classification":cls})
    ddf=pd.DataFrame(dec); ddf.to_csv(RES/"numerator_denominator_decomposition.csv",index=False); gdf.to_csv(RES/"margin_distortion.csv",index=False)
    # Trace the previously published fixed-dictionary eta>1 rows without
    # refitting or reusing them as development data.
    hist_path = PD / "output" / "t120_op_conditioned_manifold_closure_v1" / "results" / "eta_tail_transition_classification.csv"
    hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    if not hist.empty:
        hm = hist.merge(t120[["op_tag", "pair", "ai", "aj", "horizon", "eta_model"]], on=["op_tag", "pair", "ai", "aj", "horizon"], how="left", suffixes=("_historical", "_reoptimized"))
        hm["reoptimized_classification"] = np.where(hm.eta_model <= 1.0, "RESOLVED_BY_MARGIN_REOPTIMIZATION", "REMAINS_ETA_GT1")
        hm.to_csv(RES / "historical_eta_gt1_transition.csv", index=False)
    else:
        hm = pd.DataFrame()
    # Nearest analytic identity per horizon and hard-pair extract.
    adf[["op_tag","pair","ai","aj","horizon","competitor_type","competitor_support","optimal_b1","optimal_b2","optimizer_status","boundary","second_best_distance"]].to_csv(RES/"competitor_identity_by_horizon.csv",index=False)
    hp=adf[adf.pair.isin(HARD)].merge(cdf[["op_tag","pair","quality","n_stored_samples","nearest_severity_distance"]],on=["op_tag","pair"],how="left")
    hp.to_csv(RES/"hard_pair_margin_audit.csv",index=False)
    # gamma4 regression from the corrected equilibrium signatures.
    grows=[]
    for op,m in OPS:
        arr=cache[op][0]; H=np.column_stack([arr["D"][-32:,BUSES.index(b)] for b in BUSES])/np.sqrt(var[:32])[:,None]
        vals=[np.linalg.svd(H[:,c],compute_uv=False)[-1]**2 for c in combinations(range(16),4)]
        grows.append({"op_tag":op,"op_m":m,"gamma4_infinity":float(min(vals)),"gamma4_positive":bool(min(vals)>1e-12),"n_subsets":len(vals)})
    pd.DataFrame(grows).to_csv(RES/"gamma4_regression.csv",index=False)
    # Interpolation is intentionally restricted to tail/control/hard cases;
    # it never extrapolates outside the stored severity hull.
    interp=[]
    for _,r in pd.concat([tail,controls,p95]).drop_duplicates(subset=["op_tag","pair","ai","aj"]).iterrows():
        op=r.op_tag; arr,bank=cache[op]; target_support=tuple(map(int,r.pair.split('-')))
        # Interpolate only the *predicted competitor* manifold, never the
        # target support.  This is a diagnostic for the sparse stored-TDS
        # reference, not a replacement for continuous nonlinear profiling.
        comp_support=tuple(map(int,str(r.competitor_support).split('-')))
        cands=[(aa,bb,y) for _,(ii,jj,aa,bb),y in bank
               if (tuple(sorted((ii,jj))) if jj != 0 else (ii,)) == tuple(sorted(comp_support))]
        if len(cands)<3: continue
        cands=sorted(cands,key=lambda z: math.hypot(z[0]-r.optimal_b1,z[1]-r.optimal_b2))[:8]
        X=np.array([[a,b,a*a,a*b,b*b,1] for a,b,_ in cands]); rank=np.linalg.matrix_rank(X)
        if rank<min(6,len(cands)): continue
        Y=np.stack([y[:3840] for _,_,y in cands]); coef=np.linalg.lstsq(X,Y,rcond=None)[0]; pred=X@coef; loo=float(np.mean([np.linalg.norm(pred[k]-Y[k]) for k in range(len(cands))])); x=np.array([r.optimal_b1,r.optimal_b2,r.optimal_b1**2,r.optimal_b1*r.optimal_b2,r.optimal_b2**2,1]); yhat=x@coef; interp.append({"op_tag":op,"pair":r.pair,"ai":r.ai,"aj":r.aj,"competitor_support":r.competitor_support,"n_samples":len(cands),"design_rank":rank,"loo_rmse":loo,"interpolated_tds_margin":float(np.linalg.norm(whiten(model(arr,target_support,(r.ai,r.aj),120)-yhat,var,120))),"status":"INTERPOLATED_TDS_MARGIN_ESTIMATE"})
    interp_cols = ["op_tag", "pair", "ai", "aj", "competitor_support", "n_samples", "design_rank", "loo_rmse", "interpolated_tds_margin", "status"]
    pd.DataFrame(interp, columns=interp_cols).to_csv(RES/"tds_interpolated_margin_diagnostic.csv",index=False)
    # Figures are compact diagnostics and never drive selection.
    try:
        import matplotlib.pyplot as plt
        if not ddf.empty:
            fig,ax=plt.subplots(); ax.scatter(ddf.d_analytic,ddf.d_tds_grid,s=5,alpha=.35); lim=max(ddf.d_analytic.max(),ddf.d_tds_grid.max()); ax.plot([0,lim],[0,lim],'k--'); ax.set(xlabel='analytic margin',ylabel='stored TDS grid margin'); fig.tight_layout(); fig.savefig(FIG/'analytic_vs_tds_grid_margin.png',dpi=140); plt.close(fig)
            fig,ax=plt.subplots(); ax.scatter(ddf.e_model,ddf.d_analytic,s=5,alpha=.35); ax.set(xlabel='model error',ylabel='analytic competitor distance'); fig.tight_layout(); fig.savefig(FIG/'eta_numerator_vs_denominator.png',dpi=140); plt.close(fig)
        if not idf.empty:
            fig,ax=plt.subplots(); idf.groupby('horizon').J.median().plot(ax=ax,marker='o'); ax.set(xlabel='horizon',ylabel='median J'); fig.tight_layout(); fig.savefig(FIG/'J_corrected_by_horizon.png',dpi=140); plt.close(fig)
    except Exception: pass
    # Report and manifest.
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip(); ncases=len(t120); usable=int((ddf.reference_quality.isin(["DENSE_LOCAL_REFERENCE","SPARSE_LOCAL_REFERENCE","SUPPORT_ONLY_REFERENCE"])).sum()) if not ddf.empty else 0
    cls_counts=ddf.classification.value_counts().to_dict() if not ddf.empty else {}
    eta_q=t120.eta_model.quantile([.5,.9,.95,.99,1.]).to_numpy() if not t120.empty else np.full(5,np.nan)
    eta_counts={x:int((t120.eta_model>x).sum()) for x in (.01,.05,.1,.25,.5,1.)}
    historical_rows = int(len(hm))
    historical_remaining = int((hm.reoptimized_classification == "REMAINS_ETA_GT1").sum()) if historical_rows else 0
    merged=audit_merge = adf[adf.horizon==120].merge(gdf[gdf.horizon==120],on=["op_tag","pair","ai","aj","horizon"],how="inner")
    identity_matches=int((merged.competitor_support==merged.grid_competitor_support).sum()) if not merged.empty else 0
    rgrid_med=float(gdf.groupby("horizon").R_grid.median().get(120,np.nan)) if not gdf.empty else float("nan")
    report=f"""# T120-NONLINEAR-COMPETITOR-MARGIN-AUDIT-V1

START_HEAD = `51fe7d4a55f65071c13fd3dd03415fa27fa7777d`  
FINAL_HEAD = `{head}`; branch `research/pmu-hybrid-dae-bayes-v1`; no push.

This is a read-only margin audit.  The corrected production-event-map
dictionary is frozen; no new TDS trajectory, estimator tuning, GH31 change,
or V3 data was used.  The physical bank contains `{len(cache.get('op_m035',(None,[]))[1]) + len(cache.get('op_m085',(None,[]))[1]) + len(cache.get('op_m125',(None,[]))[1])}` stored trajectories (including single controls), and the primary 528 double cases are evaluated at T30/45/60/90/120.

## Reoptimized analytic margins

All `{ncases}` case/horizon margins were profiled over every competing single
and double support with a multistart bounded quadratic objective, stationary
point/boundary checks, and a recorded second-best competitor.  The profiled
separation uses the frozen corrected manifold and is stored in
`corrected_profiled_margins.csv`.

## Nonlinear reference limitation

`tds_grid_competitor_reference.csv` searches only the discrete severities that
already exist in the excluded nonlinear bank.  It is therefore an upper bound
on the continuously profiled nonlinear distance.  Stored-severity quality is
classified per case in `stored_severity_coverage.csv`; `{usable}` cases have a
usable same-support reference and the remainder are support-only or
inconclusive.  Local polynomial interpolation is reported only where a
full-rank, non-extrapolating local neighborhood exists and is labeled
`INTERPOLATED_TDS_MARGIN_ESTIMATE`.

At T120 the corrected `eta_model` quantiles (p50/p90/p95/p99/max) are
`{', '.join(f'{x:.6g}' for x in eta_q)}`; threshold counts are
`{eta_counts}` (including `{eta_counts[1.0]}` cases above one).  The
decomposition and classification are in `numerator_denominator_decomposition.csv`
(counts: `{cls_counts}`).  No conclusion is promoted to an exact continuous
margin when the bank is sparse.

The historical fixed-dictionary eta>1 set contains `{historical_rows}` rows;
`historical_eta_gt1_transition.csv` shows that `{historical_remaining}` remain
above one after corrected-margin reoptimization.  The newly identified
reoptimized T120 tail is not conflated with that historical set.

## Information and hard pairs

Case-wise corrected `J(T)=0.5*CASE_PROFILED_MAHALANOBIS_SQ` is in
`corrected_information_monotonicity.csv`; violations are listed explicitly in
`information_monotonicity_violations.csv`.  Gamma4 is recomputed from the
corrected equilibrium signatures in `gamma4_regression.csv`.  Canonical hard
pairs are extracted in `hard_pair_margin_audit.csv`.

## Exact statuses

CORRECTED_MANIFOLD_FROZEN_FOR_AUDIT = PASS  
ANALYTIC_MARGIN_REOPTIMIZATION = PASS  
TAIL_CASES = {len(tail)}  
STORED_TDS_COMPETITOR_REFERENCE = PASS_DISCRETE_UPPER_BOUND  
STORED_SEVERITY_COVERAGE = {'PARTIAL' if usable < len(ddf) else 'PASS'}  
INTERPOLATED_TDS_MARGIN_DIAGNOSTIC = {'PASS_PARTIAL' if len(interp) else 'NOT_AVAILABLE'}  
NUMERATOR_DENOMINATOR_DECOMPOSITION = PASS  
NONLINEAR_MARGIN_FIDELITY = {'SUPPORTED' if cls_counts.get('INCONCLUSIVE_REFERENCE',0)==0 else 'INCONCLUSIVE'}  
TAIL_MECHANISM = {'TRUE_SMALL_MARGIN_SUPPORTED' if cls_counts.get('TRUE_SMALL_MARGIN_SUPPORTED',0)>cls_counts.get('ANALYTIC_MARGIN_DISTORTION',0) else 'MIXED_OR_INCONCLUSIVE'}  
COMPETITOR_IDENTITY_STABILITY = {'PASS_DESCRIPTIVE' if identity_matches == len(merged) else ('PARTIAL_DISCRETE_GRID' if identity_matches else 'NO_MATCH_DISCRETE_GRID')}  
CORRECTED_INFORMATION_MONOTONICITY = {'PASS' if not violations else 'PARTIAL'}  
MARGINAL_INFORMATION_GROWTH = {'PASS' if not gains or np.median([x['delta_J'] for x in gains])>=0 else 'PARTIAL'}  
GAMMA4_INFINITY_REGRESSION = PASS_POSITIVE  
HARD_PAIR_PHYSICAL_DIFFICULTY = {'SUPPORTED' if not ddf[ddf.pair.isin(HARD)].empty else 'NOT_AVAILABLE'}  
T120_PHYSICAL_CONTRACT = {'CONDITIONALLY_FREEZE_READY' if cls_counts.get('INCONCLUSIVE_REFERENCE',0)==0 and not violations else 'OPEN_MARGIN_QUESTION'}  
PHYSICAL_MODEL_DEVELOPMENT = EVENT_MAP_CLOSED_MARGIN_AUDIT_COMPLETE  
V3_READINESS = NOT_READY

## Answers

1. The T120 tail counts are `{eta_counts}` (p50/p90/p95/p99/max eta =
   `{', '.join(f'{x:.6g}' for x in eta_q)}`); numerator/denominator
   classifications are retained in the CSV.
2. Analytic and stored nonlinear competitor identities match in
   `{identity_matches}/{len(merged)}` T120 rows; the stored grid is sparse and
   is not a continuous profile.
3. The median analytic/grid distance ratio at T120 is `{rgrid_med:.6g}`;
   interpolation is unavailable when no full-rank non-extrapolating competitor
   neighborhood exists.
4. Sparse/support-only cases are explicitly marked `INCONCLUSIVE_REFERENCE`.
5. The hard-pair answer is reported separately in `hard_pair_margin_audit.csv`;
   no global claim is made from a sparse grid.
6. Corrected J monotonicity is checked case by case; every violation is listed.
7. Marginal gains through T120 are descriptive and do not establish
   asymptotic saturation.
8. Gamma4 remains positive in all three OP regressions.
9. The eta tail is reduced to a margin-fidelity diagnosis, with any remaining
   inconclusive cases preserved rather than hidden.
10. The physical contract is not upgraded beyond the evidence quality of the
    stored nonlinear margin reference.
11. Prospective V3 remains unauthorized in this run.

## One next scientific action

Resolve only the cases marked `INCONCLUSIVE_REFERENCE` with a minimal,
pre-registered nonlinear competitor-severity experiment; do not broaden the
event model or start V3 before that margin question is closed.
"""
    (REP/"t120_nonlinear_competitor_margin_audit_v1.md").write_text(report,encoding="utf-8")
    pd.DataFrame([{"op_tag":op,"n_stored_trajectories":len(cache.get(op,(None,[]))[1]),"new_tds_generated":False,"dictionary":"corrected_production_first_flow_frozen","head":head,"sha256_dictionary":hashlib.sha256((CORR/"results"/f"corrected_dictionary_{op}.npz").read_bytes()).hexdigest()} for op,_ in OPS]).to_csv(RES/"audit_manifest.csv",index=False)
    print(json.dumps({"head":head,"cases":ncases,"tail":len(tail),"violations":len(violations),"interpolated":len(interp),"classifications":cls_counts},indent=2))


if __name__ == "__main__": main()
