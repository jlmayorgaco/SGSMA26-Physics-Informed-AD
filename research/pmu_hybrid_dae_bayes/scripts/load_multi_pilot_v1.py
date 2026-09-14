"""LOAD-MULTI-PILOT: restricted-support physical two-event validation.

This campaign is intentionally separate from LOAD-MULTI-BAYES-V1.  V2 D/Q,
Sigma0, priors and the 8-PMU map are read-only.  The selected 12 supports are
frozen from V1 exploratory geometry before this script consumes pilot data.
All physical pair trajectories are true simultaneous time-local callbacks;
noise realizations are added after a trajectory is generated.
"""
from __future__ import annotations

from pathlib import Path
import hashlib, json, math, time
import argparse
import os
import shutil
import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp
from scipy.stats import spearmanr, pearsonr, beta
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parents[1]
PD = HERE / "powerdynamics_ieee39"
V2 = PD / "output/load_bayes_fd_v2"
V2T = PD / "output/load_tangent_v2"
V1 = PD / "output/load_multi_bayes_v1"
V1R = V1 / "results"
OUT = PD / "output/load_multi_pilot_v1"
RES = OUT / "results"
REP = OUT / "reports"
QDIR, DEVDIR, TESTDIR, SINGLEDIR = OUT / "qij", OUT / "dev", OUT / "test", OUT / "single"
for p in (RES, REP, QDIR / "physical" / "results", DEVDIR / "physical" / "results",
          TESTDIR / "physical" / "results", SINGLEDIR / "physical" / "results"):
    p.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(HERE))
from scripts import e06h_corrected_m6_static as h6  # noqa: E402

BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28]
ALL_PAIRS = [(BUSES[i], BUSES[j]) for i in range(16) for j in range(i + 1, 16)]

# Frozen prospectively from V1 evidence: high/low coherence, high/low EVI,
# the mandatory (7,12) pair, and historically difficult pairs.
FROZEN_PAIRS = [(7, 12), (16, 20), (26, 28), (20, 24), (12, 24), (12, 20),
                (3, 12), (8, 20), (4, 8), (12, 16), (3, 16), (20, 27)]
H = 0.005
SIGMA_A = 0.05  # frozen V1 broad fractional-amplitude prior
CARD_PRIOR = (0.20, 0.50, 0.30)
N_NOISE = 20
H0_N = 200

# Fresh values, distinct from V1 DEV/TEST.  Fractions: 0.001 = 0.1%.
DEV_MAG_PAIRS = [(0.00012, 0.00028, "WEAK_WEAK"),
                 (0.00065, 0.00130, "WEAK_STRONG"),
                 (0.00220, 0.00380, "MODERATE"),
                 (0.00700, 0.01300, "FINITE")]
TEST_MAG_PAIRS = [(0.00018, 0.00042, "WEAK_WEAK"),
                  (0.00080, 0.00190, "WEAK_STRONG"),
                  (0.00280, 0.00480, "MODERATE"),
                  (0.00850, 0.01600, "FINITE")]
SINGLE_AMPS = [0.00037, 0.00110, 0.00330, 0.00880]


def tag(a: float) -> str:
    return str(float(a)).replace("-", "m").replace(".", "p")


def load_voltage(path: Path):
    d = pd.read_csv(path).drop_duplicates(["time", "bus"])
    t = np.sort(d.time.unique())
    re = d.pivot(index="time", columns="bus", values="V_re").reindex(t).to_numpy()
    im = d.pivot(index="time", columns="bus", values="V_im").reindex(t).to_numpy()
    return t, re + 1j * im


def pair_path(root: Path, i: int, j: int, ai: float, aj: float) -> Path:
    return root / "physical" / "results" / f"PAIR_{i}_{j}_AI{tag(ai)}_AJ{tag(aj)}_R1.csv"


def single_path(b: int, a: float) -> Path:
    return SINGLEDIR / "physical" / "results" / f"PAIR_{b}_0_AI{tag(a)}_AJ0p0_R1.csv"


def noise(seed: int, n: int = 30) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    sig = np.r_[np.full(16, 2e-4), np.full(16, 5e-4)]
    e = rng.normal(size=(n, 32)) * sig
    o = np.zeros_like(e); o[0] = e[0]
    for k in range(1, n):
        o[k] = .35 * o[k - 1] + math.sqrt(1 - .35 ** 2) * e[k]
    o[:, :4] += .15 * o[:, 4:8]
    return o


def select_pairs() -> pd.DataFrame:
    """Freeze pair selection from V1 only, with an auditable rationale."""
    g = pd.read_csv(V1R / "load_multi_pair_geometry_whitened.csv")
    e = pd.read_csv(V1R / "load_multi_evi.csv").set_index("source_bus")["EVI"]
    f = pd.read_csv(V1R / "load_multi_fisher_vs_difficulty.csv")
    f["evi_min"] = f.apply(lambda r: min(e[int(r.source_i)], e[int(r.source_j)]), axis=1)
    f["evi_max"] = f.apply(lambda r: max(e[int(r.source_i)], e[int(r.source_j)]), axis=1)
    rationale = {
        (7, 12): "mandatory Bus7-vs-Bus12; mixed EVI/coherence control",
        (16, 20): "highest-coherence and V1 difficult pair",
        (26, 28): "high-coherence, low-sigma_min pair",
        (20, 24): "highest-EVI pair and high-coherence control",
        (12, 24): "lowest-coherence and low-EVI pair",
        (12, 20): "low-coherence with high/low-EVI contrast",
        (3, 12): "low-EVI/negative-coherence control",
        (8, 20): "high-EVI, low-coherence pair",
        (4, 8): "high-EVI, near-orthogonal pair",
        (12, 16): "low-EVI and low-coherence pair",
        (3, 16): "historically hardest V1 support pair",
        (20, 27): "historically hardest V1 support pair",
    }
    rows = []
    for k, (i, j) in enumerate(FROZEN_PAIRS, 1):
        gg = g[(g.source_i == i) & (g.source_j == j)].iloc[0]
        ff = f[(f.source_i == i) & (f.source_j == j)].iloc[0]
        rows.append({"selection_index": k, "source_i": i, "source_j": j,
                     "coherence_v1": float(gg.coherence),
                     "principal_angle_v1": float(gg.principal_angle_rad),
                     "sigma_min_v1": float(gg.sigma_min),
                     "evi_i_v1": float(e[i]), "evi_j_v1": float(e[j]),
                     "evi_min_v1": float(ff.evi_min), "v1_exact_support": float(ff.exact_support),
                     "v1_atleast_one": float(ff.atleast_one), "rationale": rationale[(i, j)],
                     "selection_source": "V1_EXPLORATORY_ONLY"})
    d = pd.DataFrame(rows)
    d.to_csv(RES / "load_multi_pilot_pair_selection.csv", index=False)
    h = hashlib.sha256(d.to_csv(index=False).encode()).hexdigest()
    (RES / "load_multi_pilot_pair_selection.sha256").write_text(h + "\n", encoding="utf-8")
    return d


def frozen_physics():
    """Read-only V2 dictionary/Q and the already frozen V1 Sigma0."""
    vnom, _, _, _, meta = h6.load_nominal()
    rows = h6.load_branch_rows(vnom, meta["y0"])
    t, vv = load_voltage(V2 / "physical" / "results" / "LOAD_BUS_3_A0p0_R1.csv")
    st = int(np.argmin(abs(t - 2.0))); idx = np.arange(st, st + 30)
    yn = np.asarray([h6.measurement(z, rows) for z in vv])[idx]
    z = np.load(V2T / "results" / "load_fd_central_operator.npz")
    D = z["central"].reshape(16, -1).T.astype(float)
    amps = [-.06, -.035, -.015, -.0075, .0075, .015, .035, .06]
    Q = []
    for b in BUSES:
        yy = []
        for a in amps:
            _, v = load_voltage(V2 / "physical" / "results" / f"LOAD_BUS_{b}_A{tag(a)}_R1.csv")
            yy.append(np.asarray([h6.measurement(x, rows) for x in v])[idx].reshape(-1) - yn.reshape(-1))
        aa = np.asarray(amps); yy = np.asarray(yy); dd = D[:, BUSES.index(b)]
        Q.append(np.sum((aa ** 2)[:, None] * (yy - aa[:, None] * dd[None, :]), axis=0) / np.sum(aa ** 4))
    # Exact frozen V1 W2 covariance (same Sigma0 used in V1 and V2).
    ch = pd.read_csv(V1R / "load_multi_whitening_channels.csv")
    var = ch.sort_values("channel").variance.to_numpy(float)
    wm = pd.read_csv(V1R / "load_multi_whitening_model.csv")
    rho = float(wm.loc[wm.model == "W2_SEPARABLE_AR1", "rho"].iloc[0])
    T = rho ** np.abs(np.subtract.outer(np.arange(30), np.arange(30)))
    S = np.kron(T, np.diag(var)); L = cholesky(S, lower=True, check_finite=False)
    return D, np.asarray(Q).T, yn, idx, rows, L, S


def whiten(x, L):
    return solve_triangular(L, np.asarray(x).reshape(-1), lower=True, check_finite=False)


def response(path: Path, yn, idx, rows):
    _, v = load_voltage(path)
    return np.asarray([h6.measurement(z, rows) for z in v])[idx] - yn


def qij_from_files(pairs, yn, idx, rows, L):
    out, q = [], {}
    expected = []
    for i, j in pairs:
        for ai, aj in ((H, H), (H, -H), (-H, H), (-H, -H)):
            p = pair_path(QDIR, i, j, ai, aj)
            expected.append({"source_i": i, "source_j": j, "amplitude_i": ai, "amplitude_j": aj,
                             "case_id": p.stem, "path": str(p),
                             "status": "EXECUTED_SUCCESS" if p.exists() else "EXECUTED_FAIL"})
        pp = [response(pair_path(QDIR, i, j, ai, aj), yn, idx, rows).reshape(-1)
              for ai, aj in ((H, H), (H, -H), (-H, H), (-H, -H))]
        qx = (pp[0] - pp[1] - pp[2] + pp[3]) / (4 * H * H)
        q[(i, j)] = qx
        out.append({"source_i": i, "source_j": j, "h": H, "qij_norm": np.linalg.norm(qx),
                    "qij_whitened_norm": np.linalg.norm(whiten(qx, L)),
                    "qij_status": "PASS" if np.all(np.isfinite(qx)) else "FAIL"})
    mf = pd.DataFrame(expected); mf.to_csv(RES / "load_multi_pilot_qij_manifest.csv", index=False)
    pd.DataFrame(out).to_csv(RES / "load_multi_pilot_qij.csv", index=False)
    np.savez_compressed(RES / "load_multi_pilot_qij.npz", **{f"qij_{i}_{j}": x for (i, j), x in q.items()})
    return q, mf


def qij_step_sensitivity(pairs, yn, idx, rows, L):
    """Independent h check for two selected pairs (no TEST access)."""
    hs = [0.0025, 0.005, 0.0075]; rec = []
    for i, j in [(7, 12), (3, 16)]:
        if (i, j) not in pairs: continue
        qs = []
        for h in hs:
            hroot = QDIR if h == .005 else OUT / ("qij_h0025" if h == .0025 else "qij_h0075")
            ps = [response(pair_path(hroot, i, j, ai, aj), yn, idx, rows).reshape(-1)
                  for ai, aj in ((h, h), (h, -h), (-h, h), (-h, -h))]
            qs.append((ps[0] - ps[1] - ps[2] + ps[3]) / (4 * h * h))
        for h, q in zip(hs, qs):
            rec.append({"source_i": i, "source_j": j, "h": h, "qij_norm": float(np.linalg.norm(q)),
                        "relative_to_h005": float(np.linalg.norm(q - qs[1]) / max(np.linalg.norm(qs[1]), 1e-30)),
                        "cosine_to_h005": float((q @ qs[1]) / max(np.linalg.norm(q) * np.linalg.norm(qs[1]), 1e-30)),
                        "status": "PASS"})
    d = pd.DataFrame(rec); d.to_csv(RES / "load_multi_pilot_qij_step_sensitivity.csv", index=False); return d


def make_dev_manifest(pairs):
    rows = []
    for i, j in pairs:
        for mi, mj, reg in DEV_MAG_PAIRS:
            for si in (-1., 1.):
                for sj in (-1., 1.):
                    ai, aj = si * mi, sj * mj; p = pair_path(DEVDIR, i, j, ai, aj)
                    rows.append({"regime": reg, "source_i": i, "source_j": j, "amplitude_i": ai,
                                 "amplitude_j": aj, "physical_path": str(p), "case_id": p.stem,
                                 "status": "EXECUTED_SUCCESS" if p.exists() else "EXECUTED_FAIL"})
    d = pd.DataFrame(rows); d.to_csv(RES / "load_multi_pilot_dev_manifest.csv", index=False); return d


def make_test_manifest(pairs):
    rows = []; h0_seed, single_seed, pair_seed = 1_500_000, 1_600_000, 1_700_000
    for k in range(H0_N):
        rows.append({"case_type": "H0", "true_M": 0, "source_i": 0, "source_j": 0, "amplitude_i": 0.,
                     "amplitude_j": 0., "regime": "H0", "physical_path": "", "noise_seed": h0_seed + k})
    for b in BUSES:
        for a0 in SINGLE_AMPS:
            for s in (-1., 1.):
                a = s * a0; p = single_path(b, a)
                for k in range(N_NOISE):
                    rows.append({"case_type": "SINGLE", "true_M": 1, "source_i": b, "source_j": 0,
                                 "amplitude_i": a, "amplitude_j": 0., "regime": "SINGLE",
                                 "physical_path": str(p), "noise_seed": single_seed}); single_seed += 1
    for i, j in pairs:
        for mi, mj, reg in TEST_MAG_PAIRS:
            for si in (-1., 1.):
                for sj in (-1., 1.):
                    ai, aj = si * mi, sj * mj; p = pair_path(TESTDIR, i, j, ai, aj)
                    for k in range(N_NOISE):
                        rows.append({"case_type": "DOUBLE", "true_M": 2, "source_i": i, "source_j": j,
                                     "amplitude_i": ai, "amplitude_j": aj, "regime": reg,
                                     "physical_path": str(p), "noise_seed": pair_seed}); pair_seed += 1
    d = pd.DataFrame(rows); d.to_csv(RES / "load_multi_pilot_test_manifest.csv", index=False); return d


def additivity(dev, D, Q, qij, yn, idx, rows, L):
    rec = []
    for r in dev.itertuples():
        if not Path(r.physical_path).exists(): continue
        rr = response(Path(r.physical_path), yn, idx, rows).reshape(-1); i, j = int(r.source_i), int(r.source_j)
        ai, aj = float(r.amplitude_i), float(r.amplitude_j); ii, jj = BUSES.index(i), BUSES.index(j)
        A = ai * D[:, ii] + ai ** 2 * Q[:, ii] + aj * D[:, jj] + aj ** 2 * Q[:, jj]
        B = A + ai * aj * qij[(i, j)]
        wr = whiten(rr, L); wa = whiten(rr - A, L); wb = whiten(rr - B, L)
        rec.append({"source_i": i, "source_j": j, "regime": r.regime, "amplitude_i": ai, "amplitude_j": aj,
                    "physical_rel_A": np.linalg.norm(rr - A) / max(np.linalg.norm(rr), 1e-30),
                    "physical_rel_B": np.linalg.norm(rr - B) / max(np.linalg.norm(rr), 1e-30),
                    "whitened_rel_A": np.linalg.norm(wa) / max(np.linalg.norm(wr), 1e-30),
                    "whitened_rel_B": np.linalg.norm(wb) / max(np.linalg.norm(wr), 1e-30),
                    "whitened_error_A": wa @ wa, "whitened_error_B": wb @ wb})
    d = pd.DataFrame(rec); d.to_csv(RES / "load_multi_pilot_additivity.csv", index=False); return d


def conditional_info(Dw, Qw, qijw, pairs, amplitudes):
    out = []
    for i, j in pairs:
        for direction in ("i_to_j", "j_to_i"):
            known, target = ((i, j) if direction == "i_to_j" else (j, i))
            for ai in amplitudes:
                vi = Dw[:, BUSES.index(known)] + 2 * ai * Qw[:, BUSES.index(known)]
                vj = Dw[:, BUSES.index(target)] + ai * qijw[(i, j)]
                den = float(vi @ vi); cross = float(vi @ vj); raw = float(vj @ vj)
                floor = 1e-12 * max(float(np.linalg.norm(vi) ** 2), 1.0)
                deg = den <= floor; I = np.nan if deg else raw - cross ** 2 / den
                out.append({"source_i": i, "source_j": j, "direction": direction, "known_bus": known,
                            "target_bus": target, "amplitude_known": ai, "denominator": den,
                            "degenerate": deg, "I_j_given_i": I, "aj2_I": ai ** 2 * I if np.isfinite(I) else np.nan})
    d = pd.DataFrame(out)
    d.to_csv(RES / "load_multi_pilot_info.csv", index=False)
    return d


def quadrature(rw, Dw, Qw, qijw, grid1, grid2, pairs):
    """29-hypothesis quadrature: H0 + 16 singles + selected 12 doubles."""
    n = len(rw); const = n * np.log(2 * np.pi); rr = float(rw @ rw); out = []
    lp1 = -.5 * (grid1 / SIGMA_A) ** 2 - np.log(SIGMA_A * np.sqrt(2 * np.pi)); da = grid1[1] - grid1[0]
    out.append(("H0", (), -0.5 * (const + rr) + np.log(CARD_PRIOR[0]), None))
    for b in BUSES:
        d, q = Dw[:, BUSES.index(b)], Qw[:, BUSES.index(b)]
        M = grid1[:, None] * d[None, :] + grid1[:, None] ** 2 * q[None, :]
        ll = -.5 * (const + np.sum((rw[None, :] - M) ** 2, axis=1)) + lp1
        z = logsumexp(ll) + np.log(da * CARD_PRIOR[1] / 16)
        out.append(("single", (b,), z, ll))
    ai, aj = np.meshgrid(grid2, grid2, indexing="ij"); lp2 = (-.5 * (ai / SIGMA_A) ** 2 - .5 * (aj / SIGMA_A) ** 2
                                                       - 2 * np.log(SIGMA_A * np.sqrt(2 * np.pi)))
    da2 = (grid2[1] - grid2[0]) ** 2
    for i, j in pairs:
        d1, d2 = Dw[:, BUSES.index(i)], Dw[:, BUSES.index(j)]; q1, q2 = Qw[:, BUSES.index(i)], Qw[:, BUSES.index(j)]
        qx = qijw[(i, j)]; B = np.column_stack([d1, d2, q1, q2, qx]); G = B.T @ B; u = B.T @ rw
        C = np.column_stack([ai.ravel(), aj.ravel(), (ai * ai).ravel(), (aj * aj).ravel(), (ai * aj).ravel()])
        qn = rr - 2 * C @ u + np.einsum("ni,ij,nj->n", C, G, C)
        ll = (-.5 * (const + qn) + lp2.ravel()).reshape(ai.shape)
        z = logsumexp(ll) + np.log(da2 * CARD_PRIOR[2] / len(pairs))
        out.append(("pair", (i, j), z, ll))
    logs = np.asarray([x[2] for x in out]); pp = np.exp(logs - logsumexp(logs)); return pp, out


def quantile(x, w, q):
    o = np.argsort(x); xx, ww = np.asarray(x)[o], np.asarray(w)[o]; c = np.cumsum(ww) / max(np.sum(ww), 1e-300)
    return float(np.interp(q, c, xx))


def parse_support(value):
    """Parse tuple strings without evaluating numpy scalar reprs."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    import re
    # Numpy scalar reprs look like ``np.int64(7)``; extract the argument,
    # not the dtype suffix.  Plain tuples are handled by the second pattern.
    s = str(value)
    nums = re.findall(r"int\d+\((-?\d+)\)", s)
    if nums:
        return tuple(int(x) for x in nums)
    return tuple(int(x) for x in re.findall(r"-?\d+", s))


def infer(test, D, Q, qij, yn, idx, rows, L, pairs):
    Dw = np.column_stack([whiten(D[:, k], L) for k in range(16)]); Qw = np.column_stack([whiten(Q[:, k], L) for k in range(16)])
    qijw = {(i, j): whiten(q, L) for (i, j), q in qij.items()}
    g1, g2 = np.linspace(-.05, .05, 101), np.linspace(-.05, .05, 31)
    rec = []; t0 = time.perf_counter(); evals = 0
    for r in test.itertuples():
        if r.case_type == "H0": rr = np.zeros((30, 32)); ts = ()
        else: rr = response(Path(r.physical_path), yn, idx, rows); ts = tuple(sorted([int(r.source_i)] + ([int(r.source_j)] if r.true_M == 2 else [])))
        rw = whiten(rr + noise(int(r.noise_seed)), L); pp, meta = quadrature(rw, Dw, Qw, qijw, g1, g2, pairs); evals += len(g1) * 16 + len(g2) ** 2 * len(pairs)
        p0, p1, p2 = float(pp[0]), float(pp[1:17].sum()), float(pp[17:].sum()); k = int(np.argmax(pp)); predm = 0 if k == 0 else (1 if k < 17 else 2); predsup = meta[k][1] if k else ()
        pai = paj = np.nan; ai_lo = ai_hi = aj_lo = aj_hi = np.nan
        if predm == 1:
            ll = meta[k][3]; ww = np.exp(ll - logsumexp(ll)); pai = float(ww @ g1); ai_lo, ai_hi = quantile(g1, ww, .025), quantile(g1, ww, .975)
        elif predm == 2:
            ll = meta[k][3]; ww = np.exp(ll - logsumexp(ll)); aa, bb = np.meshgrid(g2, g2, indexing="ij"); pai, paj = float(np.sum(ww * aa)), float(np.sum(ww * bb)); ai_lo, ai_hi = quantile(aa.ravel(), ww.ravel(), .025), quantile(aa.ravel(), ww.ravel(), .975); aj_lo, aj_hi = quantile(bb.ravel(), ww.ravel(), .025), quantile(bb.ravel(), ww.ravel(), .975)
        incl = {b: float(sum(pp[q] for q, m in enumerate(meta) if b in (m[1] if isinstance(m[1], tuple) else (m[1],)))) for b in BUSES}
        ptrue = np.nan
        if r.true_M == 1: ptrue = float(pp[1 + BUSES.index(int(r.source_i))])
        elif r.true_M == 2 and ts in pairs: ptrue = float(pp[17 + pairs.index(ts)])
        support_probs = {"p_H0": float(pp[0])}
        for q, (kind, support, _, _) in enumerate(meta[1:], start=1):
            key = "p_H1_" + str(support[0]) if kind == "single" else "p_H2_" + str(support[0]) + "_" + str(support[1])
            support_probs[key] = float(pp[q])
        rec.append({"case_type": r.case_type, "regime": r.regime, "true_M": int(r.true_M), "source_i": int(r.source_i), "source_j": int(r.source_j), "amplitude_i": float(r.amplitude_i), "amplitude_j": float(r.amplitude_j), "noise_seed": int(r.noise_seed), "p_M0": p0, "p_M1": p1, "p_M2": p2, "pred_M": predm, "pred_support": str(predsup), "true_support": str(ts), "p_true_support": ptrue, "pred_amplitude_i": pai, "pred_amplitude_j": paj, "amp_i_lo95": ai_lo, "amp_i_hi95": ai_hi, "amp_j_lo95": aj_lo, "amp_j_hi95": aj_hi, "posterior_entropy": float(-np.sum(pp * np.log(np.maximum(pp, 1e-300)))), "nll": float(-np.log(max(ptrue if np.isfinite(ptrue) else pp[k], 1e-300))), **support_probs, **{f"p_include_{b}": incl[b] for b in BUSES}})
    df = pd.DataFrame(rec); df.to_parquet(RES / "load_multi_pilot_posterior.parquet", index=False); df.to_parquet(RES / "load_multi_pilot_source_posterior.parquet", index=False)
    pd.DataFrame([{"wall_seconds": time.perf_counter() - t0, "cases": len(df), "quadrature_evaluations": evals, "seconds_per_case": (time.perf_counter() - t0) / max(len(df), 1), "grid1_n": len(g1), "grid2_n": len(g2)}]).to_csv(RES / "load_multi_pilot_runtime.csv", index=False)
    return df, Dw, Qw, qijw


def metrics(df, pairs, info, add):
    # Detection and cardinality; all thresholds are descriptive, not fitted.
    ev = (df.true_M > 0).astype(int); score = 1 - df.p_M0
    fpr_n = int(np.sum(ev == 0)); fpr_k = int(np.sum((score >= .5) & (ev == 0)))
    fpr_lo = 0.0 if fpr_k == 0 else float(beta.ppf(.025, fpr_k, fpr_n - fpr_k + 1))
    fpr_hi = 1.0 if fpr_k == fpr_n else float(beta.ppf(.975, fpr_k + 1, fpr_n - fpr_k))
    det = [{"metric": "AUROC", "value": roc_auc_score(ev, score)}, {"metric": "AUPRC", "value": average_precision_score(ev, score)}, {"metric": "FPR_at_.5", "value": float(fpr_k / max(fpr_n, 1)), "n_H0": fpr_n, "binom95_lo": fpr_lo, "binom95_hi": fpr_hi}, {"metric": "FNR_at_.5", "value": float(np.mean((score < .5) & (ev == 1)))}]
    pd.DataFrame(det).to_csv(RES / "load_multi_pilot_detection.csv", index=False)
    crows = []
    for m in (0, 1, 2):
        g = df[df.true_M == m]; pred = g.pred_M.to_numpy(); crows.append({"true_M": m, "n": len(g), "accuracy": float(np.mean(pred == m)), "false_split": float(np.mean(pred == 2)) if m == 1 else np.nan, "false_merge": float(np.mean(pred == 1)) if m == 2 else np.nan, "mean_P_true": float(np.mean(g[["p_M0", "p_M1", "p_M2"][m]]))})
    pd.DataFrame(crows).to_csv(RES / "load_multi_pilot_cardinality.csv", index=False)
    pd.DataFrame([{"true_M": "ALL", "n": len(df), "mean_P_M0": float(df.p_M0.mean()), "mean_P_M1": float(df.p_M1.mean()), "mean_P_M2": float(df.p_M2.mean()), "H0_mean_P_M0": float(df[df.true_M == 0].p_M0.mean())}]).to_csv(RES / "load_multi_pilot_cardinality_summary.csv", index=False)
    d2 = df[df.true_M == 2].copy(); d2["pred_tuple"] = d2.pred_support.map(parse_support); d2["true_tuple"] = d2.true_support.map(parse_support); d2["exact"] = d2.apply(lambda x: set(x.pred_tuple) == set(x.true_tuple) and x.pred_M == 2, axis=1); d2["atleast"] = d2.apply(lambda x: bool(set(x.pred_tuple) & set(x.true_tuple)), axis=1)
    d2["false_split"] = False; d2["fusion"] = d2.pred_M == 1
    d2.to_csv(RES / "load_multi_pilot_support.csv", index=False)
    srows = []
    for p in pairs:
        g = d2[(d2.source_i == p[0]) & (d2.source_j == p[1])]; srows.append({"source_i": p[0], "source_j": p[1], "n": len(g), "exact_pair": float(g.exact.mean()), "at_least_one": float(g.atleast.mean()), "fusion": float(g.fusion.mean())})
    pd.DataFrame(srows).to_csv(RES / "load_multi_pilot_support_summary.csv", index=False)
    # Restricted-support top-k and cross-confusion diagnostics.
    pair_cols = [f"p_H2_{i}_{j}" for i, j in pairs]
    top_rows = []
    for _, x in d2.iterrows():
        vals = np.asarray([x.get(c, 0.0) for c in pair_cols], dtype=float)
        order = np.argsort(vals)[::-1]
        true_pair = tuple(sorted((int(x.source_i), int(x.source_j))))
        ranks = [(int(pairs[k][0]), int(pairs[k][1])) for k in order]
        top_rows.append({"source_i": x.source_i, "source_j": x.source_j, "true_pair": str(true_pair),
                         "top1_pair": str(ranks[0]), "top3_hit": true_pair in ranks[:3],
                         "top5_hit": true_pair in ranks[:5], "top1_probability": float(vals[order[0]])})
    top = pd.DataFrame(top_rows); top.to_csv(RES / "load_multi_pilot_support_topk.csv", index=False)
    conf = top.groupby(["true_pair", "top1_pair"], as_index=False).size().rename(columns={"size": "count"}); conf.to_csv(RES / "load_multi_pilot_pair_confusion.csv", index=False)
    # Inclusion probabilities, conditional and marginal amplitude diagnostics.
    inc = []
    for b in BUSES:
        truth = d2.apply(lambda x: b in x.true_tuple, axis=1); pred = d2[f"p_include_{b}"] >= .5; inc.append({"bus": b, "precision": float((pred & truth).sum() / max(pred.sum(), 1)), "recall": float((pred & truth).sum() / max(truth.sum(), 1)), "mean_probability": float(df[f"p_include_{b}"].mean())})
    pd.DataFrame(inc).to_csv(RES / "load_multi_pilot_inclusion.csv", index=False)
    # Both-amplitude errors only where MAP is a double support; support ambiguity is retained separately.
    arows = []
    for _, x in d2.iterrows():
        if x.pred_M == 2 and len(x.pred_tuple) == 2:
            aa, bb = sorted(x.pred_tuple); pa = x.pred_amplitude_i if aa == x.source_i else x.pred_amplitude_j; pb = x.pred_amplitude_j if bb == x.source_j else x.pred_amplitude_i
            arows.append({"regime": x.regime, "conditional_on_pred_double": True, "support_correct": bool(x.exact), "error_i": pa - x.amplitude_i, "error_j": pb - x.amplitude_j, "covered_i_95": x.amp_i_lo95 <= x.amplitude_i <= x.amp_i_hi95, "covered_j_95": x.amp_j_lo95 <= x.amplitude_j <= x.amp_j_hi95})
    ad = pd.DataFrame(arows); ad.to_csv(RES / "load_multi_pilot_amplitude.csv", index=False)
    if len(ad):
        ad.groupby("regime", as_index=False).agg(n=("error_i", "size"), bias_i=("error_i", "mean"), rmse_i=("error_i", lambda x: float(np.sqrt(np.mean(x * x)))), bias_j=("error_j", "mean"), rmse_j=("error_j", lambda x: float(np.sqrt(np.mean(x * x)))), coverage_i_95=("covered_i_95", "mean"), coverage_j_95=("covered_j_95", "mean"), support_correct=("support_correct", "mean")).to_csv(RES / "load_multi_pilot_amplitude_summary.csv", index=False)
    # Conditional information was frozen before TEST scoring; correlate prediction with target inclusion.
    pr = []
    for _, q in info.dropna(subset=["I_j_given_i"]).iterrows():
        i, j, direction = int(q.source_i), int(q.source_j), q.direction; target = int(q.target_bus); known = int(q.known_bus); ai = float(q.amplitude_known)
        g = d2[(d2.source_i == min(i, j)) & (d2.source_j == max(i, j)) & (np.sign(d2.amplitude_i) == np.sign(ai) if known == i else np.sign(d2.amplitude_j) == np.sign(ai))]
        # use target inclusion; matching exact amplitude is not required for monotone diagnostic bins
        if len(g): pr.append({"source_i": i, "source_j": j, "direction": direction, "amplitude_known": ai, "I_j_given_i": q.I_j_given_i, "aj2_I": q.aj2_I, "empirical_target_inclusion": float(np.mean(g[f"p_include_{target}"] >= .5)), "n": len(g)})
    pd.DataFrame(pr).to_csv(RES / "load_multi_pilot_detection_vs_info.csv", index=False)
    # Detection curve for the second source, retaining aj and aj^2 I bins.
    second = []
    for _, x in d2.iterrows():
        i, j = int(x.source_i), int(x.source_j); ai = float(x.amplitude_i); aj = float(x.amplitude_j)
        m = info[(info.source_i == i) & (info.source_j == j) & (info.direction == "i_to_j")]
        if len(m):
            q = m.iloc[np.argmin(np.abs(m.amplitude_known.to_numpy() - ai))]
            ii = float(q.I_j_given_i); second.append({"source_i": i, "source_j": j, "regime": x.regime,
                "amplitude_i": ai, "amplitude_j": aj, "abs_amplitude_j": abs(aj), "aj2_I": aj ** 2 * ii,
                "I_j_given_i": ii, "detected_second": bool(x[f"p_include_{j}"] >= .5), "p_include_target": float(x[f"p_include_{j}"])})
    sd = pd.DataFrame(second); sd.to_csv(RES / "load_multi_pilot_second_detection.csv", index=False)
    pred_rows = []
    if len(sd) > 3:
        for key in ("abs_amplitude_j", "aj2_I"):
            pred_rows.append({"predictor": key, "outcome": "detected_second", "spearman_rho": float(spearmanr(sd[key], sd.detected_second).statistic), "n": len(sd)})
    pd.DataFrame(pred_rows).to_csv(RES / "load_multi_pilot_info_predictive.csv", index=False)
    # Bus7/Bus12 selected case study.
    b = d2[((d2.source_i == 7) & (d2.source_j == 12)) | ((d2.source_i == 12) & (d2.source_j == 7))]; b.to_csv(RES / "load_multi_pilot_bus7_bus12.csv", index=False)
    # Frozen whitened geometry for the restricted support set.
    geom = []
    for i, j in pairs:
        # info carries no signatures; use the frozen geometry source.
        x = pd.read_csv(V1R / "load_multi_pair_geometry_whitened.csv"); z = x[(x.source_i == i) & (x.source_j == j)].iloc[0]
        geom.append(dict(source_i=i, source_j=j, coherence=float(z.coherence), principal_angle=float(z.principal_angle_rad), sigma_min=float(z.sigma_min), source="V1_FROZEN_WHITENED"))
    pd.DataFrame(geom).to_csv(RES / "load_multi_pilot_pair_geometry.csv", index=False)
    return d2, ad


def write_report(sel, qmf, dev, add, info, df, d2, ad, D):
    am = add[["physical_rel_A", "physical_rel_B", "whitened_rel_A", "whitened_rel_B"]].median()
    exact = float(d2.exact.mean()) if len(d2) else np.nan; at = float(d2.atleast.mean()) if len(d2) else np.nan
    text = {
        "HEAD_at_execution": "a59675b81 (verified before run; no rollback)",
        "label": "RESTRICTED_SUPPORT_PILOT", "selected_pairs": [list(x) for x in FROZEN_PAIRS],
        "selection_hash": hashlib.sha256(sel.to_csv(index=False).encode()).hexdigest(),
        "qij_success": int((qmf.status == "EXECUTED_SUCCESS").sum()), "qij_total": len(qmf),
        "dev_physical_unique": int(len(dev)), "dev_success": int(sum(Path(x).exists() for x in dev.physical_path)),
        "test_noise_rows": int(len(df)), "test_physical_unique": int(df[df.case_type != "H0"][["case_type", "source_i", "source_j", "amplitude_i", "amplitude_j"]].drop_duplicates().shape[0]),
        "additivity_median_rel_A": float(am.physical_rel_A), "additivity_median_rel_B": float(am.physical_rel_B),
        "whitened_median_rel_A": float(am.whitened_rel_A), "whitened_median_rel_B": float(am.whitened_rel_B),
        "exact_pair": exact, "at_least_one": at,
        "false_split": float(np.mean((df[df.true_M == 1].pred_M == 2))) if np.any(df.true_M == 1) else np.nan,
        "fusion": float(np.mean((df[df.true_M == 2].pred_M == 1))) if np.any(df.true_M == 2) else np.nan,
        "quadrature_seconds": float(pd.read_csv(RES / "load_multi_pilot_runtime.csv").wall_seconds.iloc[0]),
        "quadrature_seconds_per_case": float(pd.read_csv(RES / "load_multi_pilot_runtime.csv").seconds_per_case.iloc[0]),
        "qij_step_sensitivity_max_relative": float(pd.read_csv(RES / "load_multi_pilot_qij_step_sensitivity.csv").relative_to_h005.max()),
        "restricted_support_top3": float(pd.read_csv(RES / "load_multi_pilot_support_topk.csv").top3_hit.mean()),
        "restricted_support_top5": float(pd.read_csv(RES / "load_multi_pilot_support_topk.csv").top5_hit.mean()),
        "ANALYTIC_DAE_TANGENT": "PENDING",
    }
    det = pd.read_csv(RES / "load_multi_pilot_detection.csv")
    card = pd.read_csv(RES / "load_multi_pilot_cardinality.csv")
    ss = pd.read_csv(RES / "load_multi_pilot_support_summary.csv")
    ast = pd.read_csv(RES / "load_multi_pilot_amplitude_summary.csv") if (RES / "load_multi_pilot_amplitude_summary.csv").exists() else pd.DataFrame()
    sens = pd.read_csv(RES / "load_multi_pilot_qij_step_sensitivity.csv")
    sections = ["# LOAD-MULTI-PILOT — RESTRICTED_SUPPORT_PILOT\n",
                "## Frozen provenance\n\n" + json.dumps(text, indent=2),
                "## Physical interaction / Qij\n\n" + sens.to_markdown(index=False),
                "\nMedian DEV additivity errors (Model A vs B):\n\n" + add.groupby("regime")[["physical_rel_A", "physical_rel_B", "whitened_rel_A", "whitened_rel_B"]].median().to_markdown(),
                "## Detection and cardinality\n\n" + det.to_markdown(index=False) + "\n\n" + card.to_markdown(index=False),
                "## Restricted support\n\n" + ss.to_markdown(index=False),
                "\nTop-3 restricted-pair recovery: " + f"{pd.read_csv(RES / 'load_multi_pilot_support_topk.csv').top3_hit.mean():.3f}" + "; Top-5: " + f"{pd.read_csv(RES / 'load_multi_pilot_support_topk.csv').top5_hit.mean():.3f}",
                "## Amplitudes\n\n" + (ast.to_markdown(index=False) if len(ast) else "No case with a double MAP support in WEAK_WEAK; this is reported, not imputed."),
                "## Conditional second-event information\n\n" + info.head(24).to_markdown(index=False),
                "\nThe 12-support set was frozen from V1 evidence before pilot scoring. All pair events are simultaneous, true time-local callbacks at t=2 s with no reinitialization. V2 D/Q/Sigma0/priors and the numerical physical dictionary were read-only. This pilot is not a validation of the 137-support space. Analytic DAE tangent remains PENDING."]
    (REP / "load_multi_pilot_v1.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    pd.DataFrame([text]).to_csv(RES / "load_multi_pilot_summary.csv", index=False)


def main():
    # Freeze pair selection before any pilot physical result is inspected.
    sel = select_pairs(); pairs = [tuple(x) for x in sel[["source_i", "source_j"]].to_numpy()]
    D, Q, yn, idx, rows, L, S = frozen_physics()
    qij, qmf = qij_from_files(pairs, yn, idx, rows, L)
    qij_step_sensitivity(pairs, yn, idx, rows, L)
    dev = make_dev_manifest(pairs); test = make_test_manifest(pairs)
    # Preregister conditional information before opening any empirical test score.
    Dw = np.column_stack([whiten(D[:, k], L) for k in range(16)]); Qw = np.column_stack([whiten(Q[:, k], L) for k in range(16)])
    qijw = {(i, j): whiten(q, L) for (i, j), q in qij.items()}
    info = conditional_info(Dw, Qw, qijw, pairs, sorted({x[0] for x in DEV_MAG_PAIRS} | {x[0] for x in TEST_MAG_PAIRS}));
    # Physical interaction adequacy is DEV-only.  Do not use test for model choice.
    add = additivity(dev, D, Q, qij, yn, idx, rows, L)
    posterior_path = RES / "load_multi_pilot_posterior.parquet"
    df = pd.read_parquet(posterior_path) if posterior_path.exists() and not os.environ.get("PILOT_REINFER") else infer(test, D, Q, qij, yn, idx, rows, L, pairs)[0]
    d2, ad = metrics(df, pairs, info, add); write_report(sel, qmf, dev, add, info, df, d2, ad, D)
    # Materialize a canonical mirror alongside the campaign-specific folder.
    canonical = PD / "output" / "results"; canonical.mkdir(parents=True, exist_ok=True)
    for f in RES.glob("load_multi_pilot_*"):
        if f.is_file(): shutil.copy2(f, canonical / f.name)
    (PD / "output" / "reports").mkdir(parents=True, exist_ok=True)
    shutil.copy2(REP / "load_multi_pilot_v1.md", PD / "output" / "reports" / "load_multi_pilot_v1.md")
    print(json.dumps({"selected_pairs": len(pairs), "qij": len(qmf), "dev": len(dev), "test_noise_rows": len(df), "exact_pair": float(d2.exact.mean()), "fusion": float(d2.fusion.mean())}, indent=2))


if __name__ == "__main__":
    # The explicit switch is used to materialize the prospective selection
    # before any pilot trajectory is opened.
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-selection", action="store_true")
    args = parser.parse_args()
    if args.freeze_selection:
        sel = select_pairs(); print(json.dumps({"selection_hash": hashlib.sha256(sel.to_csv(index=False).encode()).hexdigest(), "pairs": [list(x) for x in FROZEN_PAIRS]}, indent=2))
    else:
        main()
