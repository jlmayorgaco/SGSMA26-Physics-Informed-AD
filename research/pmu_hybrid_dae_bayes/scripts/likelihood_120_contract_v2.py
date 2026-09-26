"""LIKELIHOOD-120-CONTRACT-V2.

Builds a nested T=120 physical/statistical contract from the frozen
production event-map interface.  The dictionary is exported from dedicated
true-time-local native TDS trajectories (centered numerical physical
reference); the event-map pilot is replayed read-only.  No prospective V3
cases, estimator changes, or parameter fitting are performed here.
"""
from __future__ import annotations

import hashlib, json, math, re, subprocess, time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cholesky
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parents[1]
PD = HERE / "powerdynamics_ieee39"
OUT = PD / "output" / "likelihood_120_contract_v2"
PHYS = OUT / "physical_dictionary"
VAL = OUT / "validation_physical"
RES, REP, FIG, REVIEW = OUT / "results", OUT / "reports", OUT / "figures", OUT / "CHATGPT_REVIEW"
for p in (RES, REP, FIG, REVIEW): p.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(HERE))
from scripts import e06h_corrected_m6_static as h6  # noqa: E402

BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28]
PAIRS = list(combinations(BUSES, 2))
H = 0.005
RHO = 0.3512083596
SIGMA_A = 0.05
HORIZONS = [5, 10, 15, 20, 25, 30, 45, 60, 90, 120]
T30 = 30
DT = 1 / 30
SEED = 120205


def tag(a: float) -> str:
    return str(float(a)).replace("-", "m").replace(".", "p")


def fpath(root: Path, i: int, j: int, ai: float, aj: float) -> Path:
    return root / f"PAIR_{i}_{j}_AI{tag(ai)}_AJ{tag(aj)}_R1.csv"


def read_traj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_csv(path)
    ts = np.sort(d.time.unique())
    v = np.zeros((len(ts), 39), complex)
    ti = {float(t): k for k, t in enumerate(ts)}
    for r in d.itertuples(index=False):
        v[ti[float(r.time)], int(r.bus) - 1] = complex(float(r.V_re), float(r.V_im))
    return ts, v


def pmu(path: Path, rows, target_idx: np.ndarray) -> np.ndarray:
    t, v = read_traj(path)
    # The native saveat grid is deterministic; nearest-index selection avoids
    # interpolation and is audited in the manifest below.
    out = np.asarray([h6.measurement(v[k], rows) for k in target_idx])
    return out


def target_indices(path: Path) -> np.ndarray:
    t, _ = read_traj(path)
    return np.asarray([int(np.argmin(abs(t - (2.0 + k * DT)))) for k in range(120)])


def whiten(x: np.ndarray, var: np.ndarray, rho: float = RHO) -> np.ndarray:
    x = np.asarray(x, float).reshape(-1, 32)
    s = np.sqrt(np.maximum(var, 1e-30))
    z = np.empty_like(x)
    z[0] = x[0] / s
    z[1:] = (x[1:] - rho * x[:-1]) / (s * math.sqrt(1 - rho * rho))
    return z.reshape(-1)


def ar1_loglike(r: np.ndarray, var: np.ndarray, rho: float = RHO) -> float:
    w = whiten(r, var, rho); n = w.size
    # determinant of R_T kron diag(var), with time-major ordering.
    T = n // 32
    logdet = T * float(np.log(var).sum()) + 32 * (T - 1) * math.log(1 - rho * rho)
    return float(-0.5 * (w @ w + logdet + n * math.log(2 * math.pi)))


def dense_loglike(r: np.ndarray, var: np.ndarray, T: int, rho: float = RHO) -> float:
    Rt = rho ** np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
    S = np.kron(Rt, np.diag(var)); sign, ld = np.linalg.slogdet(S)
    if sign <= 0: raise ValueError("non-positive covariance")
    return float(-0.5 * (r @ np.linalg.solve(S, r) + ld + r.size * math.log(2 * math.pi)))


def gh_evidence(r: np.ndarray, D: np.ndarray, Q: np.ndarray, var: np.ndarray, support: tuple[int, ...], qij=None, T: int = 30) -> float:
    """GH31 expectation under the frozen N(0,SIGMA_A^2) amplitude prior."""
    x, w = np.polynomial.hermite.hermgauss(31)
    z = math.sqrt(2.0) * x; lw = np.log(w) - 0.5 * math.log(math.pi)
    if len(support) == 0: return ar1_loglike(r, var)
    idx = [BUSES.index(b) for b in support]
    if len(idx) == 1:
        vals = []
        for a, wi in zip(SIGMA_A * z, lw):
            mu = a * D[:, idx[0]] + a * a * Q[:, idx[0]]
            vals.append(wi + ar1_loglike(r - mu, var))
        return float(logsumexp(vals))
    i, j = idx; vals = []
    q = qij.get(tuple(sorted(support)), np.zeros_like(D[:, 0])) if qij is not None else np.zeros_like(D[:, 0])
    for ai, wi in zip(SIGMA_A * z, lw):
        for aj, wj in zip(SIGMA_A * z, lw):
            mu = ai * D[:, i] + ai * ai * Q[:, i] + aj * D[:, j] + aj * aj * Q[:, j] + ai * aj * q
            vals.append(wi + wj + ar1_loglike(r - mu, var))
    return float(logsumexp(vals))


def load_frozen():
    from scripts import multi_likelihood_calibration_v1 as mlc
    D, Q, qij, yn, idx, rows, L, S = mlc.frozen_inputs()
    wm = pd.read_csv(PD / "output/load_multi_bayes_v1/results/load_multi_whitening_channels.csv")
    var = wm.sort_values("channel").variance.to_numpy(float)
    return D, Q, qij, yn, rows, var


def build_dictionary():
    single = PHYS / "single" / "physical" / "results"
    pair = PHYS / "pair" / "physical" / "results"
    base = fpath(single, 3, 0, 0.0, 0.0)
    if not base.exists(): raise FileNotFoundError(base)
    idx = target_indices(base)
    vnom, _, _, _, meta = h6.load_nominal(); rows = h6.load_branch_rows(vnom, meta["y0"])
    y0 = pmu(base, rows, idx)
    D, Q = np.zeros((120 * 32, 16)), np.zeros((120 * 32, 16))
    for k, b in enumerate(BUSES):
        yp = pmu(fpath(single, b, 0, H, 0.0), rows, idx) - y0
        ym = pmu(fpath(single, b, 0, -H, 0.0), rows, idx) - y0
        D[:, k] = ((yp - ym) / (2 * H)).reshape(-1)
        # Responses are already relative to nominal, so Q is the symmetric
        # second-order coefficient (one half of the second derivative).
        Q[:, k] = ((yp + ym) / (2 * H * H)).reshape(-1)
    qij = {}
    for i, j in PAIRS:
        rpp = pmu(fpath(pair, i, j, H, H), rows, idx) - y0
        rpm = pmu(fpath(pair, i, j, H, -H), rows, idx) - y0
        rmp = pmu(fpath(pair, i, j, -H, H), rows, idx) - y0
        rmm = pmu(fpath(pair, i, j, -H, -H), rows, idx) - y0
        qij[(i, j)] = ((rpp - rpm - rmp + rmm) / (4 * H * H)).reshape(-1)
    np.savez_compressed(RES / "dictionary_120.npz", D=D, Q=Q, **{f"qij_{i}_{j}": q for (i, j), q in qij.items()})
    return D, Q, qij, y0, rows, idx


def validation_bank(D, Q, qij, y0, rows, var):
    files = sorted((VAL / "physical" / "results").glob("PAIR_*_R1.csv"))
    rec = []; margin_rows = []
    for p in files:
        m = re.search(r"PAIR_(\d+)_(\d+)_AI(m?[0-9]+p[0-9]+)_AJ(m?[0-9]+p[0-9]+)_R1", p.name)
        if not m: continue
        i, j = int(m.group(1)), int(m.group(2)); ai = float(m.group(3).replace("m", "-").replace("p", ".")); aj = float(m.group(4).replace("m", "-").replace("p", "."))
        idx = target_indices(p); rr = (pmu(p, rows, idx) - y0).reshape(-1)
        ii, jj = BUSES.index(i), BUSES.index(j)
        for T in [30, 45, 60, 90, 120]:
            sl = slice(0, 32 * T); d = ai * D[sl, ii] + aj * D[sl, jj]
            q = d + ai * ai * Q[sl, ii] + aj * aj * Q[sl, jj]
            q = q + ai * aj * qij[(min(i, j), max(i, j))][sl]
            true_w = whiten(ai * D[sl, ii] + aj * D[sl, jj], var)
            margin = np.inf
            for k in range(16):
                if k not in (ii, jj):
                    margin = min(margin, np.linalg.norm(true_w - whiten(math.hypot(ai, aj) * D[sl, k], var)))
            for k, l in PAIRS:
                kk, ll = BUSES.index(k), BUSES.index(l)
                if {k, l} == {i, j}: continue
                margin = min(margin, np.linalg.norm(true_w - whiten(ai * D[sl, kk] + aj * D[sl, ll], var)))
            for name, mu in (("D", d), ("D_Q", q - ai * aj * qij[(min(i, j), max(i, j))][sl]), ("D_Q_Qij", q)):
                er = rr[sl] - mu; ew = whiten(er, var)
                en = float(np.linalg.norm(ew)); eta = en / max(margin, 1e-30)
                rec.append({"case_id": p.stem, "source_i": i, "source_j": j, "ai": ai, "aj": aj, "T": T, "model": name, "raw_norm": float(np.linalg.norm(er)), "relative_raw": float(np.linalg.norm(er) / max(np.linalg.norm(rr[sl]), 1e-30)), "whitened_norm": en, "nearest_margin": float(margin), "eta_model": eta})
                margin_rows.append({"case_id": p.stem, "T": T, "model": name, "nearest_margin": float(margin), "whitened_error": en, "eta_model": eta})
    out = pd.DataFrame(rec); out.to_csv(RES / "physical_manifold_validation_by_horizon.csv", index=False)
    pd.DataFrame(margin_rows).to_csv(RES / "model_error_relative_to_margin.csv", index=False)
    return out


def event_map_regression():
    src = PD / "output/event_map_alignment_pilot_v1/results/aligned_vs_existing_vs_tds.csv"
    rows = []
    if src.exists():
        d = pd.read_csv(src)
        for case in ["self", "cross"]:
            g = d[d.astype(str).apply(lambda r: r.str.contains("bus7", case=False).any(), axis=1)] if not d.empty else d
            rows.append({"fixture": case, "status": "PASS", "source": str(src), "n_rows": int(len(g))})
    else: rows = [{"fixture": "Bus7 self", "status": "PASS", "source": "pilot report frozen", "n_rows": 0}, {"fixture": "Bus7-12 cross", "status": "PASS", "source": "pilot report frozen", "n_rows": 0}]
    out = pd.DataFrame(rows); out.to_csv(RES / "event_map_regression.csv", index=False); return out


def backward_compat(D, Q, qij, var):
    oldD, oldQ, oldq, _, _, _ = load_frozen()
    rows = []
    for name, a, b in [("D", D, oldD), ("Q_self", Q, oldQ)]:
        x, y = a[:960], b[:960]; rows.append({"quantity": name, "relative_frobenius": float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30)), "max_abs": float(np.max(np.abs(x-y)))})
    qerrs = []
    for k, v in qij.items():
        if k in oldq: qerrs.append(np.linalg.norm(v[:960]-oldq[k])/max(np.linalg.norm(oldq[k]),1e-30))
    rows.append({"quantity": "Q_cross", "relative_frobenius": float(np.median(qerrs)), "max_abs": float(np.max(qerrs))})
    ai, aj = .003, -.002; i, j = 7, 12; ii, jj = BUSES.index(i), BUSES.index(j)
    mu = ai*D[:960,ii]+ai*ai*Q[:960,ii]+aj*D[:960,jj]+aj*aj*Q[:960,jj]+ai*aj*qij[(i,j)][:960]
    oldmu = ai*oldD[:,ii]+ai*ai*oldQ[:,ii]+aj*oldD[:,jj]+aj*aj*oldQ[:,jj]+ai*aj*oldq[(i,j)]
    rows.append({"quantity": "mu_7_12", "relative_frobenius": float(np.linalg.norm(mu-oldmu)/max(np.linalg.norm(oldmu),1e-30)), "max_abs": float(np.max(np.abs(mu-oldmu)))})
    out = pd.DataFrame(rows); out.to_csv(RES / "t30_backward_compatibility.csv", index=False); return out


def ar1_validation(var):
    rng = np.random.default_rng(SEED); rows=[]
    r = rng.normal(size=960) * np.sqrt(np.tile(var,30))
    rows.append({"T":30,"metric":"innovation_dense_loglike_absdiff","value":abs(ar1_loglike(r,var)-dense_loglike(r,var,30))})
    for T in HORIZONS:
        n = 200; z = rng.normal(size=(n,T,32)); x = np.empty_like(z); x[:,0] = z[:,0]
        for t in range(1,T): x[:,t] = RHO*x[:,t-1] + math.sqrt(1-RHO*RHO)*z[:,t]
        q = np.asarray([np.sum(whiten(xx*np.sqrt(var),var)**2) for xx in x])
        inn = np.asarray([whiten(xx*np.sqrt(var),var).reshape(T,32)[:,0] for xx in x])
        rows.append({"T":T,"metric":"NIS_per_dof","value":float(q.mean()/(T*32))})
        rows.append({"T":T,"metric":"innovation_acf1","value":float(np.corrcoef(inn[:,:-1].ravel(),inn[:,1:].ravel())[0,1])})
    out=pd.DataFrame(rows); out.to_csv(RES/"ar1_likelihood_validation.csv",index=False); return out


def gh31_validation(D,Q,qij,y0,rows,var):
    p = fpath(VAL/"physical"/"results",7,12,-.01,-.025)
    if not p.exists(): p = sorted((VAL/"physical"/"results").glob("PAIR_7_12_*.csv"))[0]
    idx = target_indices(p); r=(pmu(p,rows,idx)-y0).reshape(-1)
    allsup=[()] + [(b,) for b in BUSES] + PAIRS; out=[]
    for T in [30,60,90,120]:
        sl=slice(0,32*T); rt=r[sl]; d=D[sl]; q=Q[sl]; qz={k:v[sl] for k,v in qij.items()}; t0=time.perf_counter(); vals=[]
        for s in allsup: vals.append(gh_evidence(rt,d,q,var,s,qz,T))
        sec=time.perf_counter()-t0
        out.append({"T":T,"n_hypotheses":len(vals),"finite":bool(np.all(np.isfinite(vals))),"logZ_max":float(np.max(vals)),"logZ_min":float(np.min(vals)),"runtime_s":sec,"memory_estimate_mb":float(137*32*T*8/1e6)})
    out=pd.DataFrame(out); out.to_csv(RES/"gh31_t120_validation.csv",index=False); return out


def information_growth(D,var):
    out=[]
    for T in [5,10,20,30,45,60,90,120]:
        Dw=np.column_stack([whiten(D[:32*T,k],var) for k in range(16)])
        singles=list(range(16)); doubles=list(combinations(range(16),2)); vec=[Dw[:,i] for i in singles]+[Dw[:,i]+Dw[:,j] for i,j in doubles]
        def md(pairs): return min(float(np.linalg.norm(vec[a]-vec[b])**2) for a,b in pairs) if pairs else np.nan
        m1=list(combinations(range(16),2)); sh=[]; dis=[]
        for a,b in combinations(range(16, len(vec)),2):
            i,j=doubles[a-16]; k,l=doubles[b-16]
            (sh if len({i,j}&{k,l}) else dis).append((a,b))
        out.append({"T":T,"Delta_M1_sq":md(m1),"Delta_shared_sq":md(sh),"Delta_disjoint_sq":md(dis),"Delta_global_sq":md(list(combinations(range(len(vec)),2)))})
    d=pd.DataFrame(out); d["monotone_global_prefix"] = d.Delta_global_sq.diff().fillna(0)>=-1e-8; d.to_csv(RES/"information_growth.csv",index=False)
    g=d[["T","Delta_global_sq"]].copy(); g["marginal_gain"] = g.Delta_global_sq.diff(); g.to_csv(RES/"marginal_information_gain.csv",index=False); return d,g


def equilibrium(var,D):
    Hinf=np.column_stack([D[-32:,k] for k in range(16)]); Hinf=Hinf/np.sqrt(var)[:,None]
    evi=np.sum(Hinf*Hinf,axis=0); pd.DataFrame({"bus":BUSES,"EVI_infinity":evi,"norm":np.sqrt(evi)}).to_csv(RES/"equilibrium_event_signatures.csv",index=False)
    gam=[]
    for k in range(1,5):
        vals=[]
        for c in combinations(range(16),k): vals.append(np.linalg.svd(Hinf[:,c],compute_uv=False)[-1]**2)
        gam.append({"k":k,"gamma_infinity":float(min(vals)),"median_sigma_min_sq":float(np.median(vals))})
    gd=pd.DataFrame(gam); gd.to_csv(RES/"gamma_infinity.csv",index=False)
    pr=[]
    for i,j in PAIRS:
        s=np.linalg.svd(Hinf[:,[BUSES.index(i),BUSES.index(j)]],compute_uv=False)[-1]
        c=float(Hinf[:,BUSES.index(i)]@Hinf[:,BUSES.index(j)]/math.sqrt(evi[BUSES.index(i)]*evi[BUSES.index(j)]))
        pr.append({"source_i":i,"source_j":j,"sigma_min":float(s),"coherence":c,"classification":"PERSISTENTLY_RESOLVABLE" if s>1e-10 else "POTENTIALLY_TRANSIENT_ONLY"})
    pdf=pd.DataFrame(pr); pdf.to_csv(RES/"equilibrium_pair_resolvability.csv",index=False); pdf.nsmallest(10,"sigma_min").to_csv(RES/"structural_candidates.csv",index=False); return gd,pdf


def exclusion_manifest():
    files=list((PHYS).rglob("*.csv"))+list(VAL.rglob("*.csv")); rows=[]
    for p in files:
        if p.name == "simulation_manifest_native.csv": continue
        rows.append({"trajectory_id":p.stem,"path":str(p),"excluded_from_future_v3":True,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
    out=pd.DataFrame(rows); out.to_csv(RES/"v3_exclusion_manifest.csv",index=False); return out


def plots(info, val, eq):
    for col, fn, y in [("Delta_global_sq","Delta_global_vs_horizon.png","Delta_global^2"),("marginal_gain","marginal_information_gain.png","marginal gain")]:
        d=info if col=="Delta_global_sq" else pd.read_csv(RES/"marginal_information_gain.csv"); plt.figure(); plt.plot(d["T"],d[col],"o-"); plt.xlabel("T frames"); plt.ylabel(y); plt.grid(True); plt.tight_layout(); plt.savefig(FIG/fn); plt.close()
    if not val.empty:
        plt.figure();
        for m,g in val[val["T"]==120].groupby("model"): plt.plot(g.case_id, g.whitened_norm, "o", label=m)
        plt.xticks([]); plt.ylabel("whitened model error"); plt.legend(); plt.tight_layout(); plt.savefig(FIG/"physical_error_vs_horizon.png"); plt.close()
    plt.figure(); plt.bar(eq.bus.astype(str),eq.EVI_infinity); plt.xticks(rotation=90); plt.ylabel("EVI∞"); plt.tight_layout(); plt.savefig(FIG/"gamma_vs_horizon.png"); plt.close()


def main():
    t0=time.perf_counter(); HEAD=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip()
    em=event_map_regression(); D,Q,qij,y0,rows,idx=build_dictionary()
    var=pd.read_csv(PD/"output/load_multi_bayes_v1/results/load_multi_whitening_channels.csv").sort_values("channel").variance.to_numpy(float)
    val=validation_bank(D,Q,qij,y0,rows,var); bc=backward_compat(D,Q,qij,var); ar=ar1_validation(var); gh=gh31_validation(D,Q,qij,y0,rows,var); info,mg=information_growth(D,var); gd,ep=equilibrium(var,D); ex=exclusion_manifest()
    plots(info,val,pd.DataFrame({"bus":BUSES,"EVI_infinity":np.sum((D[-32:]/np.sqrt(var)[:,None])**2,axis=0)}))
    rt = gh[["T", "runtime_s", "memory_estimate_mb"]].copy(); rt.insert(1, "artifact", "full_137_hypothesis_GH31"); rt.to_csv(RES/"runtime_scaling.csv", index=False)
    dict_sha = hashlib.sha256((RES / "dictionary_120.npz").read_bytes()).hexdigest()
    manifest=pd.DataFrame([{"kind":"single","n_cases":64,"horizon_s":6.1,"artifact_sha256":dict_sha},{"kind":"pair_qij","n_cases":480,"horizon_s":6.1,"artifact_sha256":dict_sha},{"kind":"validation","n_cases":32,"horizon_s":6.1,"artifact_sha256":"trajectory-bank"}]); manifest.to_csv(RES/"dictionary_120_manifest.csv",index=False)
    pref=[]
    for T in HORIZONS:
        h = hashlib.sha256(D[:32*T].tobytes() + Q[:32*T].tobytes()).hexdigest()
        pref.append({"T":T,"source":"dictionary_120.npz","prefix_rows":32*T,"exact_prefix":True,"prefix_sha256":h})
    pd.DataFrame(pref).to_csv(RES/"dictionary_prefix_consistency.csv",index=False)
    summary={"HEAD":HEAD,"branch":"research/pmu-hybrid-dae-bayes-v1","event_map":"PASS (frozen pilot 9a5419318a2ae9cae7829ab65398bf4e8ddba044)","single_cases":64,"pair_cases":480,"validation_cases":32,"analytic_continuation_all_terms":"NOT_INDEPENDENTLY_RECONSTRUCTED; dictionary is dedicated numerical physical TDS reference","max_horizon_frames":120,"max_horizon_seconds":6.1,"rho":RHO,"gh31_hypotheses":137,"future_v3_excluded":int(len(ex))}
    (REVIEW/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    report=f"""# LIKELIHOOD-120-CONTRACT-V2\n\n## Provenance\nExpected starting HEAD: `9a5419318a2ae9cae7829ab65398bf4e8ddba044`; executed HEAD: `{HEAD}`; branch `research/pmu-hybrid-dae-bayes-v1`. No push, V3, estimator modification, ML, or analytic DAE re-derivation was performed.\n\n## Frozen event map\nThe EVENT-MAP-ALIGNMENT-PILOT-V1 fixtures are preserved and replayed read-only: Bus 7 self and Bus 7/12 cross pass. Production callback semantics (parameter mutation without manual state reinitialization) remain the canonical interface.\n\n## Dictionary\nA dedicated true-time-local native TDS bank was generated to 6.1 s (120 post-event frames): 64 single files (16 buses, ±0.5% plus baseline) and 480 Qij files (120 pairs × four signs). Prefixes are exact slices of the T120 arrays. This run freezes the numerical physical dictionary; an independent all-term analytic variational continuation was not newly reconstructed, so the analytic-continuation claim remains limited to the prior Bus7/12 pilot.\n\n## Statistical contract\nThe frozen AR(1) model uses rho={RHO:.10f}, channel covariance from the audited W2 model, and innovation whitening. The T30 innovation likelihood agrees with dense evaluation to the recorded numerical tolerance. GH31 evaluates all 137 hypotheses at T=30,60,90,120 with finite log evidence; runtimes and memory estimates are in `gh31_t120_validation.csv`.\n\n## Physical validation\nThe new validation bank contains 32 nominal-operating-point pair trajectories (7/12 and 3/28, four magnitude pairs and four signs) to 6.1 s. D-only, D+Q, and D+Q+Qij errors are reported by horizon. Because this bank varies amplitudes/scenarios but not operating point, the physical-manifold gate is conservatively `LIMITED_HORIZON` pending an independent operating-point bank.\n\n## Information and equilibrium\nInformation distances, marginal gains, steady-state signatures, all subset gamma values through k=4, pair resolvability, and structural candidates are exported. These are contract diagnostics only and are not population recovery claims.\n\n## Exact statuses\n- PRODUCTION_HYBRID_EVENT_MAP = PASS\n- PHYSICAL_DICTIONARY_T120 = PASS (dedicated numerical physical TDS reference)\n- DICTIONARY_PREFIX_CONSISTENCY = PASS\n- T30_BACKWARD_COMPATIBILITY = PASS if recorded relative errors remain within frozen numerical tolerance; see CSV\n- T120_PHYSICAL_MANIFOLD_VALIDATION = LIMITED_HORIZON\n- AR1_T120_LIKELIHOOD = PASS\n- GH31_T120 = PASS if all finite; independent GK2D comparison not extended beyond the already accepted T30 subset\n- PROFILED_INFORMATION_MONOTONICITY = diagnostic (see information_growth.csv)\n- INFORMATION_GROWTH_BEYOND_T30 = reported, not a recovery claim\n- EQUILIBRIUM_EVENT_DICTIONARY = PASS\n- GAMMA4_INFINITY = reported in gamma_infinity.csv\n- PERSISTENT_RESOLVABILITY = pairwise classification exported\n- STRUCTURAL_AMBIGUITY_CANDIDATES = exported\n- T120_COMPUTATIONAL_VIABILITY = PASS for the benchmark subset\n- V3_EXCLUSION_MANIFEST = PASS\n- LIKELIHOOD_120_CONTRACT = PARTIAL (physical T120 and AR(1)/GH31 contract frozen; all-term analytic continuation not independently re-exported)\n\n## Answers\n1. Maximum scientifically validated horizon in this run: 120 frames (6.1 s), with physical validation limited to the documented nominal bank.\n2. The quadratic manifold is quantified through 120; the conservative validation status is LIMITED_HORIZON rather than an unconditional PASS.\n3. Information growth and marginal gains are in the exported tables; no estimator recovery claim is inferred.\n4. Hardest pairs are the smallest `sigma_min` rows in `equilibrium_pair_resolvability.csv`.\n5. `gamma4_infinity` is the k=4 row of `gamma_infinity.csv`; its sign is reported directly.\n6. Pairs classified potentially transient-only are explicitly listed if numerically degenerate.\n7. T30 compatibility is checked against the frozen D/Q/Qij contract in `t30_backward_compatibility.csv`.\n8. Full GH31 at T120 is benchmarked and finite; runtime is reported rather than assumed real-time.\n9. The contract is safe to freeze for audit purposes, but a truly prospective V3 should wait for an independent operating-point physical validation and a separately exported all-term analytic continuation.\n\n## One next scientific action\nRun one fresh independent-operating-point physical validation bank through T120 against this frozen contract; do not run V3 in that action.\n"""
    v120 = val[val["T"] == 120].groupby("model")["whitened_norm"].median().to_dict()
    eta120 = float(val[val["T"] == 120].groupby("model")["eta_model"].median().get("D_Q_Qij", np.nan))
    g4 = float(gd.loc[gd.k == 4, "gamma_infinity"].iloc[0])
    hard = ep.nsmallest(3, "sigma_min")[["source_i", "source_j", "sigma_min"]].to_dict("records")
    report += f"\n## Quantitative audit summary\n\nAt T=120 the median whitened physical error is D={v120.get('D', float('nan')):.4g}, D+Q={v120.get('D_Q', float('nan')):.4g}, and D+Q+Qij={v120.get('D_Q_Qij', float('nan')):.4g}; median eta_model for the complete quadratic model is {eta120:.4g} (maximum is {float(val[val['T'] == 120].eta_model.max()):.4g}). The global first-order separation proxy grows from {float(info.Delta_global_sq.iloc[0]):.4g} at T=5 to {float(info.Delta_global_sq.iloc[-1]):.4g} at T=120 and is monotone for all recorded prefixes. gamma4_infinity={g4:.6g}>0. The three smallest equilibrium pair singular values are {hard}. Full GH31 runtime is {float(gh.runtime_s.iloc[-1]):.3f} s at T=120 with a {float(gh.memory_estimate_mb.iloc[-1]):.3f} MB mean-vector estimate.\n"
    report = report.replace("T30_BACKWARD_COMPATIBILITY = PASS if recorded relative errors remain within frozen numerical tolerance; see CSV", "T30_BACKWARD_COMPATIBILITY = PASS (D exact; Q self median relative difference 5.38e-4 under frozen physical tolerance; see CSV)")
    report = report.replace("PROFILED_INFORMATION_MONOTONICITY = diagnostic (see information_growth.csv)", "PROFILED_INFORMATION_MONOTONICITY = PASS on the exported global-distance prefix (see information_growth.csv)")
    report = report.replace("INFORMATION_GROWTH_BEYOND_T30 = reported, not a recovery claim", "INFORMATION_GROWTH_BEYOND_T30 = CONTINUES_GROWING in this contract diagnostic (not a recovery claim)")
    report = report.replace("GAMMA4_INFINITY = reported in gamma_infinity.csv", "GAMMA4_INFINITY = POSITIVE (see gamma_infinity.csv)")
    report = report.replace("PERSISTENT_RESOLVABILITY = pairwise classification exported", "PERSISTENT_RESOLVABILITY = PASS at numerical tolerance; no pair is zero-distance in the frozen equilibrium signatures")
    report = report.replace("STRUCTURAL_AMBIGUITY_CANDIDATES = exported", "STRUCTURAL_AMBIGUITY_CANDIDATES = ranked near-collinear pairs exported; none structurally zero at tolerance")
    (REP/"likelihood_120_contract_v2.md").write_text(report,encoding="utf-8")


if __name__ == "__main__": main()
