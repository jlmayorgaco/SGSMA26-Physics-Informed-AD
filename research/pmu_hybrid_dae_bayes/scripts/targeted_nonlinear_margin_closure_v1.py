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
    print("Frozen manifest present; run targeted nonlinear execution through the companion Julia driver and --analyze.")


if __name__ == "__main__":
    main()
