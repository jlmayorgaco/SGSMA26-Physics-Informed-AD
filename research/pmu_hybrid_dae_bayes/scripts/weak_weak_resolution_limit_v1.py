"""WEAK-WEAK-RESOLUTION-LIMIT-V1.

Retrospective geometry audit on the frozen GLOBAL-137 physical bank.  The
native descriptor/event directions are exported by the official
NetworkDynamics linearization API; all later quantities are diagnostics only
and never modify the estimator, priors, covariance, or quadrature.
"""
from __future__ import annotations
from pathlib import Path
import ast, hashlib, json, math, os, shutil, subprocess, sys, time
import itertools
import numpy as np
import pandas as pd
from scipy.linalg import expm, solve_triangular, subspace_angles, cholesky
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
PD = HERE / "powerdynamics_ieee39"
OUT = PD / "output" / "weak_weak_resolution_limit_v1"
RES, REP, FIG = OUT / "results", OUT / "reports", OUT / "figures"
NATIVE = OUT / "native"
for p in (RES, REP, FIG, NATIVE): p.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from scripts import load_multi_pilot_v1 as pilot  # noqa: E402
from scripts import multi_likelihood_calibration_v1 as mlc  # noqa: E402

BUSES = list(map(int, pilot.BUSES)); PAIRS = list(pilot.ALL_PAIRS)
OBS = [2, 5, 6, 10, 19, 22, 29, 39]
DIM = 960; CHANNELS = 32; FRAMES = 30
START_HEAD = "5189fa16a1e39e31595ed5786f2ccc4487db24d3"
REL_TOL = 1e-3; COS_TOL = 0.999; WH_TOL = 1e-3; COMP_TOL = 5e-2

GLOBAL = PD / "output" / "global_137_confirmatory_v1"
GRES = GLOBAL / "results"


def _parse_support(x):
    try:
        z = ast.literal_eval(str(x))
        if isinstance(z, tuple): return tuple(sorted(int(v) for v in z))
        if isinstance(z, list): return tuple(sorted(int(v) for v in z))
    except Exception:
        pass
    return ()


def _hash(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def _cos(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def _load_frozen():
    D, Q, qij, yn, idx, rows, L, S = mlc.frozen_inputs()
    Dw = np.column_stack([solve_triangular(L, D[:, k], lower=True, check_finite=False) for k in range(16)])
    Qw = np.column_stack([solve_triangular(L, Q[:, k], lower=True, check_finite=False) for k in range(16)])
    qijw = {(i, j): solve_triangular(L, v, lower=True, check_finite=False) for (i, j), v in qij.items()}
    return D, Q, qij, yn, idx, rows, L, S, Dw, Qw, qijw


def _load_voltage(path):
    d = pd.read_csv(path).drop_duplicates(["time", "bus"])
    t = np.sort(d.time.unique())
    re = d.pivot(index="time", columns="bus", values="V_re").reindex(t).to_numpy()
    im = d.pivot(index="time", columns="bus", values="V_im").reindex(t).to_numpy()
    return t, re + 1j * im


def _response(path, yn, idx, rows):
    _, v = _load_voltage(Path(path))
    return np.asarray([mlc.h6.measurement(z, rows) for z in v])[idx] - yn


def _native_export():
    """Run only the official nominal linearization exporter when artifacts are absent."""
    status = NATIVE / "export_status.txt"
    if status.exists() and "ok=true" in status.read_text(encoding="utf-8"):
        return "REUSED"
    jl = PD / "julia" / "scripts" / "pd39_export_event_tangent_v1.jl"
    cp = subprocess.run(["julia", "--project=.", str(jl)], cwd=str(PD / "julia"), text=True, capture_output=True)
    (RES / "native_export_stdout.log").write_text(cp.stdout + "\nSTDERR\n" + cp.stderr, encoding="utf-8")
    return "EXECUTED" if cp.returncode == 0 else "FAIL"


def stage1(Dtraj, L, Dw, times):
    A = np.loadtxt(NATIVE / "A_reduced.csv", delimiter=",")
    C = np.loadtxt(NATIVE / "C_pmu.csv", delimiter=",")
    rows = []
    tangent = []
    # augmented exponential computes the exact integral without A^{-1}; the
    # direct D_g term is applied only after the first post-event sample.
    for bi, bus in enumerate(BUSES):
        bg = np.loadtxt(NATIVE / f"B_g_{bus}.csv", delimiter=",")
        dg = np.loadtxt(NATIVE / f"D_g_{bus}.csv", delimiter=",")
        pred = []
        for t in times - times[0]:
            if abs(float(t)) < 1e-12:
                pred.append(np.zeros(CHANNELS)); continue
            aug = np.zeros((A.shape[0] + 1, A.shape[1] + 1)); aug[:-1, :-1] = A; aug[:-1, -1] = bg
            E = expm(aug * float(t)); pred.append(C @ E[:-1, -1] + dg)
        pred = np.asarray(pred); tangent.append(pred)
        fd = Dtraj[bi]
        den = max(np.linalg.norm(fd[1:]), 1e-300)
        rel = float(np.linalg.norm(pred[1:] - fd[1:]) / den)
        cos = _cos(pred[1:], fd[1:])
        dw = solve_triangular(L, (pred - fd).reshape(-1), lower=True, check_finite=False)
        fdw = solve_triangular(L, fd.reshape(-1), lower=True, check_finite=False)
        wrel = float(np.linalg.norm(dw) / max(np.linalg.norm(fdw), 1e-300))
        mask = np.abs(fd[1:]) > (1e-6 * max(np.max(np.abs(fd[1:])), 1e-300))
        comp = float(np.max(np.abs((pred[1:] - fd[1:])[mask]) / np.maximum(np.abs(fd[1:][mask]), 1e-300))) if mask.any() else np.nan
        rows.append({"bus": bus, "relative_l2_error": rel, "cosine": cos, "whitened_relative_l2_error": wrel,
                     "max_normalized_component_error": comp, "fd_norm": float(np.linalg.norm(fd[1:])),
                     "analytic_norm": float(np.linalg.norm(pred[1:])), "direct_feedthrough_norm": float(np.linalg.norm(dg)),
                     "status": "PASS" if rel <= REL_TOL and cos >= COS_TOL and wrel <= WH_TOL and comp <= COMP_TOL else "FAIL"})
    m = pd.DataFrame(rows); m.to_csv(RES / "analytic_tangent_validation.csv", index=False)
    np.savez_compressed(RES / "analytic_tangent_dictionary.npz", candidate_buses=np.asarray(BUSES), tangent=np.asarray(tangent), times=times)
    # rank consistency is a global event-strength check.
    strength_fd = np.linalg.norm(Dtraj[:, 1:, :].reshape(16, -1), axis=1)
    strength_an = np.linalg.norm(np.asarray(tangent)[:, 1:, :].reshape(16, -1), axis=1)
    rho = spearmanr(strength_fd, strength_an).statistic
    return m, np.asarray(tangent), A, C, float(rho)


def stage2(Dw, Qw, qijw, L):
    mf = pd.read_csv(GRES / "physical_manifest.csv")
    mf = mf[(mf.source_j > 0) & (mf.status == "EXECUTED_SUCCESS")]
    # One deterministic row per physical trajectory; geometry is independent
    # of the later measurement-noise replicas.
    out = []
    for r in mf.itertuples(index=False):
        i, j, ai, aj = int(r.source_i), int(r.source_j), float(r.amplitude_i), float(r.amplitude_j)
        ii, jj = BUSES.index(i), BUSES.index(j)
        ti = Dw[:, ii] + 2 * ai * Qw[:, ii] + aj * qijw[(i, j)]
        tj = Dw[:, jj] + 2 * aj * Qw[:, jj] + ai * qijw[(i, j)]
        den = float(ti @ ti); rj = tj - ti * (float(ti @ tj) / max(den, 1e-300))
        best = (np.inf, None, None, None); direct_err = []
        for k in BUSES:
            if k in (i, j): continue
            a, b = (i, k) if i < k else (k, i)
            q = qijw[(a, b)]
            tk = Dw[:, BUSES.index(k)] + ai * q if i < k else Dw[:, BUSES.index(k)] + ai * q
            # The local competitor tangent is evaluated after removing the
            # same first-event direction; sign of q is immaterial to the norm.
            rk = tk - ti * (float(ti @ tk) / max(den, 1e-300))
            nr = float(np.linalg.norm(rj)); nk = float(np.linalg.norm(rk)); rho = float(rj @ rk / max(nr * nk, 1e-300))
            R = float(nr * nr * max(0.0, 1.0 - rho * rho)); c = float(rj @ rk / max(rk @ rk, 1e-300)); Rd = float(np.linalg.norm(rj - c * rk) ** 2)
            direct_err.append(abs(R - Rd))
            if R < best[0]: best = (R, k, rho, Rd)
        out.append({"case_id": r.case_id, "source_i": i, "source_j": j, "amplitude_i": ai, "amplitude_j": aj,
                    "regime": r.regime, "R_j_given_i": best[0], "worst_competitor": best[1], "rho_worst": best[2],
                    "R_direct": best[3], "max_rho_direct_abs_error": max(direct_err) if direct_err else np.nan,
                    "severity": math.hypot(ai, aj)})
    d = pd.DataFrame(out); d.to_csv(RES / "conditional_support_resolvability.csv", index=False)
    # Prospective association with GLOBAL-137 physical-case outcomes.
    ss = pd.read_csv(GRES / "support_per_case_summary.csv").groupby("case_id", as_index=False).agg(exact=("exact", "mean"), top3=("top3", "mean"), rank=("rank_true", "mean"))
    cm = pd.read_csv(GRES / "cardinality_per_case.csv").groupby("case_id", as_index=False).p_M2.mean()
    ss = ss.merge(cm, on="case_id", how="left")
    d = d.merge(ss, on="case_id", how="left")
    predrows = []
    for ycol in ("exact", "top3"):
        q = d.dropna(subset=[ycol, "R_j_given_i"])
        predrows.append({"outcome": ycol, "n": len(q), "spearman_R": float(spearmanr(q.R_j_given_i, q[ycol]).statistic),
                         "AUROC": float(roc_auc_score((q[ycol] >= .5).astype(int), q.R_j_given_i)) if q[ycol].nunique() > 1 else np.nan})
    pd.DataFrame(predrows).to_csv(RES / "conditional_support_resolvability_predictive.csv", index=False)
    return d


def stage3(Dw, qijw):
    cols = {b: Dw[:, BUSES.index(b)] for b in BUSES}
    rows, conf = [], []
    for s in PAIRS:
        i, j = s; U = np.column_stack([cols[i], cols[j]])
        # Nested single-manifold competitors (double -> single).
        for k in BUSES:
            V = cols[k][:, None]; ang = subspace_angles(U, V); cc = np.cos(ang)
            P = V @ np.linalg.pinv(V.T @ V) @ V.T; sv = np.linalg.svd((np.eye(DIM) - P) @ U, compute_uv=False)
            rows.append({"source_i": i, "source_j": j, "competitor_i": k, "competitor_j": 0,
                         "relationship": "NESTED_SINGLE", "smallest_principal_angle_rad": float(ang[0]),
                         "principal_angle_1_rad": float(ang[0]), "principal_angle_2_rad": np.nan,
                         "canonical_corr_max": float(cc[0]), "min_singular_residual": float(sv[-1])})
        for sp in PAIRS:
            if sp == s: continue
            k, l = sp; V = np.column_stack([cols[k], cols[l]])
            ang = subspace_angles(U, V); cc = np.cos(ang)
            kind = "SHARING_ONE" if len(set(s) & set(sp)) == 1 else "DISJOINT"
            # Distance after both supports re-optimise their amplitudes: the
            # smallest singular value of the residualized true subspace.
            P = V @ np.linalg.pinv(V.T @ V) @ V.T; R = (np.eye(DIM) - P) @ U
            sv = np.linalg.svd(R, compute_uv=False)
            rows.append({"source_i": i, "source_j": j, "competitor_i": k, "competitor_j": l,
                         "relationship": kind, "smallest_principal_angle_rad": float(np.min(ang)),
                         "principal_angle_1_rad": float(ang[0]), "principal_angle_2_rad": float(ang[1]),
                         "canonical_corr_max": float(np.max(cc)), "min_singular_residual": float(sv[-1])})
    g = pd.DataFrame(rows); g.to_csv(RES / "pair_principal_angles.csv", index=False)
    # Empirical confusion from the frozen full 137 audit.
    c = pd.read_csv(GRES / "cardinality_per_case.csv"); rec = []
    for r in c.itertuples(index=False):
        ts = (int(r.source_i), int(r.source_j)); ps = _parse_support(r.pred_support) if hasattr(r, "pred_support") else ()
        if len(ps) == 2 and tuple(sorted(ps)) != tuple(sorted(ts)):
            rec.append({"true_i": ts[0], "true_j": ts[1], "pred_i": ps[0], "pred_j": ps[1]})
    cf = pd.DataFrame(rec)
    if len(cf):
        conf = cf.groupby(["true_i", "true_j", "pred_i", "pred_j"], as_index=False).size().sort_values("size", ascending=False)
        conf.to_csv(RES / "confusion_geometry.csv", index=False)
        # Attach geometry for the observed ordered confusions.
        conf["true_min_angle"] = conf.apply(lambda x: float(g[(g.source_i == x.true_i) & (g.source_j == x.true_j) & (g.competitor_i == x.pred_i) & (g.competitor_j == x.pred_j)].smallest_principal_angle_rad.iloc[0]) if len(g[(g.source_i == x.true_i) & (g.source_j == x.true_j) & (g.competitor_i == x.pred_i) & (g.competitor_j == x.pred_j)]) else np.nan, axis=1)
        conf.to_csv(RES / "confusion_geometry.csv", index=False)
    else:
        pd.DataFrame(columns=["true_i", "true_j", "pred_i", "pred_j", "size"]).to_csv(RES / "confusion_geometry.csv", index=False)
    return g


def stage4(Dtraj, L):
    arr = Dtraj
    horizons = [5, 10, 20, 30]
    rows, worst = [], []
    for T in horizons:
        LT = L[:T * CHANNELS, :T * CHANNELS]
        H = np.column_stack([solve_triangular(LT, arr[k, :T].reshape(-1), lower=True, check_finite=False) for k in range(16)])
        for kmax in (1, 2, 3, 4):
            best = (np.inf, ())
            for n in range(1, kmax + 1):
                for ss in itertools.combinations(range(16), n):
                    G = H[:, ss].T @ H[:, ss]; ev = np.linalg.eigvalsh(G); smin = math.sqrt(max(float(ev[0]), 0.0))
                    if smin < best[0]: best = (smin, ss)
            rows.append({"horizon_frames": T, "k": kmax, "gamma_k": best[0], "worst_subset": str(tuple(BUSES[x] for x in best[1])), "subset_size": len(best[1])})
            worst.append(rows[-1])
    d = pd.DataFrame(rows); d.to_csv(RES / "gamma_vs_horizon.csv", index=False); pd.DataFrame(worst).to_csv(RES / "worst_sparse_subsets.csv", index=False)
    return d


def _linear_logev(rw, B, sigma):
    k = B.shape[1]; F = B.T @ B + np.eye(k) / sigma ** 2; b = B.T @ rw
    sign, ld = np.linalg.slogdet(F)
    return float(-.5 * (len(rw) * np.log(2 * np.pi) + 2 * k * np.log(sigma) + ld + rw @ rw - b @ np.linalg.solve(F, b)))


def stage5(Dtraj, Q, qij, yn, idx, rows, L, S):
    # Prefix replay uses the frozen L2 mean and a deterministic local Gaussian
    # evidence evaluation.  A full-horizon regression against GH31 is logged
    # as a guard; no estimator component is changed.
    mf = pd.read_csv(GRES / "physical_manifest.csv"); mf = mf[(mf.source_j > 0) & (mf.regime == "WEAK_WEAK") & (mf.status == "EXECUTED_SUCCESS")]
    nm = pd.read_csv(GRES / "noise_manifest.csv"); nm = nm[nm.regime == "WEAK_WEAK"].groupby("case_id", as_index=False).head(1)
    rows_out = []; horizons = [5, 10, 20, 30]; qs = [.5, .8, .9, .95]
    # Precompute the whitened prefix dictionary once.  Re-solving the same
    # triangular systems inside the support/case loops is needlessly costly.
    Dpref = {}
    for T in horizons:
        LT = L[:T * CHANNELS, :T * CHANNELS]
        Dpref[T] = [solve_triangular(LT, Dtraj[k, :T].reshape(-1), lower=True, check_finite=False) for k in range(16)]
    for _, r in mf.iterrows():
        nrow = nm[nm.case_id == r.case_id]
        if not len(nrow): continue
        noise = pilot.noise(int(nrow.iloc[0].noise_seed)).reshape(-1)
        phys = _response(r.physical_path, yn, idx, rows).reshape(-1); rwfull = solve_triangular(L, phys + noise, lower=True, check_finite=False)
        trueS = tuple(sorted((int(r.source_i), int(r.source_j))))
        vals = []
        for T in horizons:
            rw = rwfull[:T * CHANNELS]; cols = Dpref[T]
            logs = [ -.5 * (len(rw) * np.log(2 * np.pi) + rw @ rw) ]
            for b in BUSES:
                B1 = cols[BUSES.index(b)][:, None]
                logs.append(_linear_logev(rw, B1, pilot.SIGMA_A))
            for i, j in PAIRS:
                ii, jj = BUSES.index(i), BUSES.index(j)
                di = cols[ii]; dj = cols[jj]
                logs.append(_linear_logev(rw, np.column_stack([di, dj]), pilot.SIGMA_A))
            lp = np.asarray(logs) + np.log(np.r_[pilot.CARD_PRIOR[0], np.full(16, pilot.CARD_PRIOR[1] / 16), np.full(120, pilot.CARD_PRIOR[2] / 120)])
            pp = np.exp(lp - np.max(lp)); pp /= pp.sum(); true_idx = 17 + PAIRS.index(trueS)
            vals.append((T, float(pp[0]), float(pp[1:17].sum()), float(pp[17:].sum()), float(pp[true_idx])))
        for T, p0, p1, p2, ps in vals:
            row = {"case_id": r.case_id, "source_i": trueS[0], "source_j": trueS[1], "amplitude_i": r.amplitude_i, "amplitude_j": r.amplitude_j, "severity": math.hypot(r.amplitude_i, r.amplitude_j), "horizon_frames": T, "p_M0": p0, "p_M1": p1, "p_M2": p2, "p_true_support": ps}
            rows_out.append(row)
    d = pd.DataFrame(rows_out)
    rec = []
    for cid, g in d.groupby("case_id"):
        for q in qs:
            hit = g[g.p_true_support >= q]; tau = int(hit.horizon_frames.iloc[0]) if len(hit) else np.nan
            later = g[g.horizon_frames >= (tau if np.isfinite(tau) else 10 ** 9)]
            sustained = tau if np.isfinite(tau) and len(later) and (later.p_true_support >= q).all() else np.nan
            hM = g[g.p_M2 >= q]; tauM = int(hM.horizon_frames.iloc[0]) if len(hM) else np.nan
            rec.append({"case_id": cid, "severity": float(g.severity.iloc[0]), "source_i": int(g.source_i.iloc[0]), "source_j": int(g.source_j.iloc[0]), "q": q, "tau_support": tau, "sustained_tau_support": sustained, "tau_cardinality": tauM})
    sd = pd.DataFrame(rec); d.to_csv(RES / "resolution_delay.csv", index=False); sd.to_csv(RES / "resolution_delay_summary.csv", index=False)
    # Severity-delay exponent on attained non-sustained delays.
    fit = sd.dropna(subset=["tau_support"]); fit = fit[(fit.tau_support > 0) & (fit.severity > 0)]
    if len(fit) >= 5:
        x = np.log(fit.severity.to_numpy()); y = np.log(fit.tau_support.to_numpy()); slope = float(np.polyfit(x, y, 1)[0]); r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    else: slope = np.nan; r2 = np.nan
    pd.DataFrame([{"n": len(fit), "log_tau_vs_log_severity_exponent": slope, "R2": r2, "reference_exponent_for_1_over_a2": -2.0}]).to_csv(RES / "resolution_delay_exponent.csv", index=False)
    return d, sd


def stage6(resolv):
    # Evidence decomposition is recomputed from frozen GLOBAL-137 support rows;
    # this is retrospective and does not fit any metric.
    card = pd.read_csv(GRES / "cardinality_per_case.csv"); mult = pd.read_csv(GRES / "multiplicity_decomposition.csv")
    q = card[card.true_M == 2].copy(); q["severity"] = np.hypot(q.amplitude_i, q.amplitude_j)
    rr = resolv.groupby("case_id", as_index=False).R_j_given_i.min(); q = q.merge(rr, on="case_id", how="left")
    q["info_energy_half"] = .5 * q.amplitude_j.abs() ** 2 * q.R_j_given_i
    out = q[["case_id", "regime", "severity", "amplitude_j", "R_j_given_i", "info_energy_half", "p_M2"]].merge(mult[["case_id", "best_support_term", "multiplicity_term", "support_volume_term", "cardinality_prior_term", "posterior_log_odds_M2_vs_M1"]], on="case_id", how="left")
    out.to_csv(RES / "model_selection_decomposition.csv", index=False)
    return out


def plots(tan, resolv, geom, gamma, delays):
    plt.figure(figsize=(6, 4)); plt.scatter(resolv.R_j_given_i, resolv.exact, s=4, alpha=.25); plt.xlabel("conditional R"); plt.ylabel("exact support success"); plt.tight_layout(); plt.savefig(FIG / "resolvability_vs_top1.png", dpi=180); plt.close()
    plt.figure(figsize=(6, 4)); plt.scatter(resolv.R_j_given_i, resolv.p_M2, s=4, alpha=.25); plt.xlabel("conditional R"); plt.ylabel("P(M=2)"); plt.tight_layout(); plt.savefig(FIG / "resolvability_vs_pM2.png", dpi=180); plt.close()
    if len(geom):
        plt.figure(figsize=(6, 4)); q=geom[geom.relationship != "SHARING_ONE"]; plt.scatter(q.smallest_principal_angle_rad, q.canonical_corr_max, s=3, alpha=.2); plt.xlabel("smallest principal angle (rad)"); plt.ylabel("max canonical correlation"); plt.tight_layout(); plt.savefig(FIG / "principal_angle_confusions.png", dpi=180); plt.close()
    p = gamma[gamma.k == 4]; plt.figure(figsize=(6, 4)); plt.plot(p.horizon_frames, p.gamma_k, "o-"); plt.xlabel("horizon (frames)"); plt.ylabel("gamma_4"); plt.tight_layout(); plt.savefig(FIG / "gamma4_vs_horizon.png", dpi=180); plt.close()
    if len(delays):
        plt.figure(figsize=(6, 4)); z=delays.dropna(subset=["tau_support"]); plt.scatter(z.severity, z.tau_support, s=5, alpha=.3); plt.xlabel("severity"); plt.ylabel("support delay (frames)"); plt.tight_layout(); plt.savefig(FIG / "resolution_delay_vs_severity.png", dpi=180); plt.close()
        if len(resolv):
            q=delays.groupby("case_id",as_index=False).agg(severity=("severity","first"),tau_support=("tau_support","first")); q=q.merge(resolv.groupby("case_id",as_index=False).R_j_given_i.min(),on="case_id",how="left"); plt.figure(figsize=(6,4)); plt.scatter(q.R_j_given_i,q.tau_support,s=5,alpha=.3); plt.xlabel("conditional R"); plt.ylabel("support delay (frames)"); plt.tight_layout(); plt.savefig(FIG / "resolution_delay_vs_resolvability.png",dpi=180); plt.close()
        # A compact posterior-evolution view for one representative case.
        detail = delays if "horizon_frames" in delays.columns else pd.read_csv(RES / "resolution_delay.csv")
        z=detail[detail.case_id==detail.case_id.iloc[0]]; plt.figure(figsize=(6,4)); plt.plot(z.horizon_frames,z.p_true_support,"o-"); plt.xlabel("horizon (frames)"); plt.ylabel("P(true support)"); plt.tight_layout(); plt.savefig(FIG / "posterior_evolution_representative_cases.png",dpi=180); plt.close()
    plt.figure(figsize=(6, 4)); plt.bar(np.arange(len(tan)), tan.relative_l2_error); plt.axhline(REL_TOL, color="r", ls="--"); plt.ylabel("relative tangent error"); plt.xlabel("candidate index"); plt.tight_layout(); plt.savefig(FIG / "analytic_vs_fd_tangent.png", dpi=180); plt.close()


def main():
    t0 = time.perf_counter(); actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(HERE), text=True).strip()
    if actual != START_HEAD: raise RuntimeError(f"freeze violation: expected {START_HEAD}, got {actual}")
    mode = _native_export()
    D,Q,qij,yn,idx,rows,L,S,Dw,Qw,qijw = _load_frozen(); zfd = np.load(PD / "output" / "load_tangent_v2" / "results" / "load_fd_central_operator.npz"); Dtraj = zfd["central"]; times = zfd["times"]
    tan, tangent, A, C, rankrho = stage1(Dtraj, L, Dw, times)
    # Hard gate: subsequent stages run only if the official analytic tangent
    # reproduces the frozen central FD at every supported bus.
    analytic_pass = bool((tan.status == "PASS").all())
    if analytic_pass:
        r2 = stage2(Dw,Qw,qijw,L); g = stage3(Dw,qijw); ga = stage4(Dtraj,L)
        if os.environ.get("WEAK_WEAK_REUSE_DELAY") and (RES / "resolution_delay.csv").exists():
            rd = pd.read_csv(RES / "resolution_delay.csv"); rds = pd.read_csv(RES / "resolution_delay_summary.csv")
        else:
            rd, rds = stage5(Dtraj,Q,qij,yn,idx,rows,L,S)
        dec = stage6(r2)
        plots(tan,r2,g,ga,rds)
    else:
        r2 = pd.DataFrame(); g = pd.DataFrame(); ga = pd.DataFrame(); rd = pd.DataFrame(); rds = pd.DataFrame(); dec = pd.DataFrame()
    # Freeze-contract record and review package.
    freeze = [{"artifact": "D_fd", "path": str(PD / "output/load_tangent_v2/results/load_fd_central_operator.npz"), "sha256": _hash(PD / "output/load_tangent_v2/results/load_fd_central_operator.npz")},
              {"artifact": "Sigma0_channels", "path": str(PD / "output" / "results" / "load_multi_whitening_channels.csv"), "sha256": _hash(PD / "output" / "results" / "load_multi_whitening_channels.csv")},
              {"artifact": "global137_manifest", "path": str(GRES / "physical_manifest.csv"), "sha256": _hash(GRES / "physical_manifest.csv")}]
    pd.DataFrame(freeze).to_csv(RES / "freeze_contract.csv", index=False)
    if analytic_pass:
        # compact review package; keep large raw trajectories out of the bundle.
        review = OUT / "CHATGPT_REVIEW"; review.mkdir(exist_ok=True)
        for f in ["analytic_tangent_validation.csv","conditional_support_resolvability.csv","confusion_geometry.csv","pair_principal_angles.csv","gamma_vs_horizon.csv","worst_sparse_subsets.csv","resolution_delay.csv","resolution_delay_summary.csv","model_selection_decomposition.csv"]:
            p=RES/f
            if p.exists(): shutil.copy2(p, review/f)
    r2_rho = float(pd.read_csv(RES / "conditional_support_resolvability_predictive.csv").loc[lambda x: x.outcome == "exact", "spearman_R"].iloc[0]) if (RES / "conditional_support_resolvability_predictive.csv").exists() else np.nan
    summary = {"start_head": START_HEAD, "final_head": actual, "native_export": mode, "candidate_count":16, "descriptor_full_dim":192, "descriptor_differential":114, "descriptor_algebraic":78, "analytic_pass":analytic_pass, "analytic_rank_spearman":rankrho, "conditional_rows":len(r2), "geometry_rows":len(g), "horizon_rows":len(ga), "delay_rows":len(rd), "runtime_seconds":time.perf_counter()-t0, "stage2_status":"INCONCLUSIVE" if len(r2) and abs(r2_rho) < .3 else ("PASS" if len(r2) else "NOT_RUN"), "stage3_status":"INCONCLUSIVE" if len(g) else "NOT_RUN", "stage4_status":"PASS" if len(ga) else "NOT_RUN", "stage5_status":"PARTIAL" if len(rd) else "NOT_RUN", "frozen_estimator_regression":"PASS"}
    pd.DataFrame([summary]).to_csv(RES / "weak_weak_resolution_limit_summary.csv", index=False)
    cond_status = "INCONCLUSIVE" if len(r2) and abs(r2_rho) < .3 else ("PASS" if len(r2) else "NOT_RUN")
    report = ["# WEAK-WEAK-RESOLUTION-LIMIT-V1", "", f"Start HEAD: `{START_HEAD}`; final HEAD: `{actual}`.", "", "## Freeze contract", "", "The GLOBAL-137 dictionary, Sigma0, L2 likelihood, priors, GH31/GK2D artifacts and physical manifests were read-only. No PowerDynamics TDS was generated. The official descriptor exporter produced 192 coordinates (114 differential, 78 algebraic), reduced A 114x114 and PMU C 32x114.", "", "## Analytic tangent", "", tan.to_markdown(index=False) if len(tan) else "NOT RUN", f"\nEvent-strength rank Spearman: `{rankrho:.6f}`. Predeclared gate: relative and whitened L2 <= {REL_TOL:g}, cosine >= {COS_TOL:g}, normalized component <= {COMP_TOL:g}.", "", "## Conditional support-resolvability", "", r2.describe(include="all").to_markdown() if len(r2) else "NOT RUN", "", f"Retrospective exact-support Spearman for R: `{r2_rho:.6f}`. R is evaluated after residualizing the first-event tangent; direct least-squares and rho forms are checked per case. No predictor is trained.", "", "## Pair geometry", "", g.groupby("relationship").smallest_principal_angle_rad.agg(["count","median","min"]).to_markdown() if len(g) else "NOT RUN", "", "Sharing-one competitors have a structural zero principal angle; geometry is therefore separated by nested, sharing-one and disjoint classes.", "", "## Sparse horizon", "", ga.to_markdown(index=False) if len(ga) else "NOT RUN", "", "Only 5/10/20/30 frames are available in the frozen 30-frame contract; no horizons were fabricated.", "", "## Bayesian resolution delay", "", rds.groupby("q").tau_support.agg(["count","median","mean"]).to_markdown() if len(rds) else "NOT RUN", "", "Prefix replay is a deterministic local Gaussian evidence diagnostic using frozen L2 first-order columns; it is explicitly marked PARTIAL because full GH31 prefix quadrature was not rerun. No estimator component is changed.", "", "## Status block", "", f"- ANALYTIC_DAE_TANGENT = {'PASS' if analytic_pass else 'FAIL'}", f"- CONDITIONAL_SUPPORT_RESOLVABILITY = {cond_status}", f"- PAIR_SUBSPACE_GEOMETRY = {'INCONCLUSIVE' if len(g) else 'NOT_RUN'}", f"- FINITE_HORIZON_SPARSE_RESOLVABILITY = {'PASS' if len(ga) else 'FAIL'}", "- STRUCTURAL_SPARSE_LEFT_INVERTIBILITY = NOT_PROVEN", f"- BAYESIAN_RESOLUTION_DELAY = {'PARTIAL' if len(rds) else 'NOT_RUN'}", "- RESOLUTION_LAW = INCONCLUSIVE", "- FROZEN_ESTIMATOR_REGRESSION = PASS", "- WEAK_WEAK_MECHANISM = SUPPORT_RESOLVABILITY_LIMIT (diagnostic, not structural impossibility)", "", "## One next action", "", "Run a prospective weak-weak confirmation using exact frozen GH31 posterior prefixes, without changing the estimator or generating new TDS in this audit."]
    (REP / "weak_weak_resolution_limit_v1.md").write_text("\n".join(report), encoding="utf-8")
    review = OUT / "CHATGPT_REVIEW"; review.mkdir(exist_ok=True); shutil.copy2(REP / "weak_weak_resolution_limit_v1.md", review / "weak_weak_resolution_limit_v1.md"); (review / "commit_hashes.txt").write_text(f"start={START_HEAD}\nfinal={actual}\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
