"""E06-I: causal two-timescale online recentering on fresh M6 streams.

The online path receives only the 32 frozen PMU channels and nominal model
metadata.  The E06-H nonlinear AC MAP implementation is imported unchanged;
only its PMU target is replaced by a causal slow estimate.  Truth is read in
the evaluation wrapper after the estimate has been produced.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from scripts import e06h_corrected_m6_static as h6  # noqa: E402

ROOT = HERE / "powerdynamics_ieee39"
RES = ROOT / "output" / "results"
REPORTS = ROOT / "output" / "reports"
PLOTS = ROOT / "output" / "plots"
CASES = RES / "e06i_cases_m6"
REPORTS.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

WINDOWS = [10, 30, 60]
CADENCES = [1, 3, 10, 30]
EXTRACTORS = ["TRAILING_MEAN", "HUBER_TRAILING_MEAN", "EMA"]
DT = 1.0 / 30.0
QK = 1e-6
RK = 1e-6
PK = 1e-2
DEN_FLOOR = 1e-6


def cplx(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if np.iscomplexobj(a):
        return a
    a = a.astype(float)
    return a[..., 0] + 1j * a[..., 1]


def wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    e = pred - truth
    tv = np.abs(e) / np.maximum(np.abs(truth), 1e-12)
    ang = np.rad2deg(wrap(np.angle(pred) - np.angle(truth)))
    return {
        "TVE_fraction": float(np.mean(tv)),
        "TVE_percent": float(100 * np.mean(tv)),
        "angle_RMSE_deg": float(np.sqrt(np.mean(ang**2))),
        "magnitude_RMSE": float(np.sqrt(np.mean((np.abs(pred) - np.abs(truth)) ** 2))),
        "complex_RMSE": float(np.sqrt(np.mean(np.abs(e) ** 2))),
        "p95_TVE_percent": float(100 * np.quantile(tv, 0.95)),
        "worst_bus_TVE_percent": float(100 * np.max(np.mean(tv, axis=0))),
    }


def load_model():
    A = expm(pd.read_csv(RES / "e04_A.csv").to_numpy(float) * DT)
    C = pd.read_csv(RES / "e04_C_pmu.csv").to_numpy(float)
    L = pd.read_csv(RES / "e04_C_hidden.csv").to_numpy(float)
    y0 = pd.read_csv(RES / "e04_y0_pmu.csv").iloc[:, 0].to_numpy(float)
    h0 = cplx(pd.read_csv(RES / "e04_pd_hidden0.csv").iloc[:, 0].to_numpy(float).reshape(31, 2))
    return A, C, L, y0, h0


def load_cases(manifest: str) -> list[tuple[pd.Series, np.ndarray, np.ndarray]]:
    out = []
    mf = pd.read_csv(RES / manifest)
    for _, row in mf.iterrows():
        path = CASES / f"{row.case_id}_trajectory.csv"
        if not path.exists():
            continue
        tr = pd.read_csv(path)
        y = tr[[f"pmu_{i}" for i in range(1, 33)]].to_numpy(float)
        truth = cplx(tr[[f"hidden_{i}" for i in range(1, 63)]].to_numpy(float).reshape(len(tr), 31, 2))
        out.append((row, y, truth))
    return out


def causal_center(history: np.ndarray, extractor: str, window: int) -> np.ndarray:
    """Return a causal slow PMU target; no future sample is accessed."""
    if len(history) == 0:
        raise ValueError("history must contain the current sample")
    w = min(window, len(history))
    z = np.asarray(history[-w:], float)
    if extractor == "TRAILING_MEAN":
        return z.mean(axis=0)
    if extractor == "HUBER_TRAILING_MEAN":
        mu = z.mean(axis=0)
        for _ in range(4):
            r = z - mu
            scale = np.median(np.abs(r), axis=0) * 1.4826 + 1e-9
            q = np.abs(r) / (1.5 * scale)
            wt = np.minimum(1.0, 1.0 / np.maximum(q, 1.0))
            mu = np.sum(wt * z, axis=0) / np.maximum(np.sum(wt, axis=0), 1e-12)
        return mu
    if extractor == "EMA":
        alpha = 2.0 / (window + 1.0)
        mu = history[0].copy()
        for zt in history[1:]:
            mu = alpha * zt + (1.0 - alpha) * mu
        return mu
    raise ValueError(f"unknown extractor {extractor}")


def kalman(A, C, L, y0, h0, y, centers, hidden_centers):
    x = np.zeros(A.shape[0]); P = np.eye(A.shape[0]) * PK
    pred, cov, innovations = [], [], []
    filter_time = 0.0
    Q = np.eye(A.shape[0]) * QK; Rk = np.eye(C.shape[0]) * RK
    for k, z in enumerate(y):
        t0 = time.perf_counter()
        xp = A @ x; Pp = A @ P @ A.T + Q
        S = C @ Pp @ C.T + Rk
        inn = (z - centers[k]) - C @ xp
        K = np.linalg.solve(S, C @ Pp).T
        x = xp + K @ inn
        I = np.eye(A.shape[0]); P = (I - K @ C) @ Pp @ (I - K @ C).T + K @ Rk @ K.T
        P = (P + P.T) / 2
        hv = hidden_centers[k] + cplx((L @ x).reshape(31, 2))
        pred.append(hv); cov.append(P.copy()); innovations.append(inn)
        filter_time += time.perf_counter() - t0
    return np.asarray(pred), np.asarray(cov), np.asarray(innovations), filter_time


def center_covariance(xmap: np.ndarray, ytarget: np.ndarray, vnom: np.ndarray, snom: np.ndarray,
                      ybus: np.ndarray, rows, qd: float, lam: float) -> np.ndarray:
    """Laplace covariance of hidden voltage center (nullspace-aware pinv)."""
    _, jac, _ = h6.map_residual_jac(xmap, ytarget, vnom, snom, ybus, rows, qd, lam)
    covx = np.linalg.pinv(jac.T @ jac, rcond=1e-10)
    v, _, _, _ = h6.unpack(xmap)
    jhid = h6.hidden_jacobian(v)
    return jhid @ covx[:78, :78] @ jhid.T


def online_case(A, C, L, y0, h0, y, qd, lam, extractor, window, cadence,
                vnom, snom, ybus, rows):
    """Causal R2-PF-ONLINE.  The truth arrays are deliberately not arguments."""
    history = []; center_v = vnom.copy(); center_pmu = h6.measurement(center_v, rows)
    center_hidden = center_v[h6.HIDDEN].copy(); center_cov = np.zeros((62, 62))
    centers = []; hcenters = []; covs = []; updates = []; logs = []
    total_map_time = 0.0
    for k, z in enumerate(y):
        history.append(z.copy())
        # First target is available only after enough causal history; prior
        # center is held between updates.  This creates a measurable startup
        # delay instead of using a non-causal centered filter.
        eligible = (k + 1) >= window and ((k + 1 - window) % cadence == 0)
        if eligible:
            yslow = causal_center(np.asarray(history), extractor, window)
            vhat, sol, info, dt = h6.run_map(yslow, vnom, snom, ybus, rows, qd, lam,
                                              vstart=center_v, max_nfev=80)
            total_map_time += dt
            accepted = bool(info["valid"])
            if accepted:
                center_v = vhat
                center_pmu = h6.measurement(center_v, rows)
                center_hidden = center_v[h6.HIDDEN].copy()
                xmap = np.r_[np.angle(center_v), np.log(np.abs(center_v)), np.zeros(43)]
                center_cov = center_covariance(xmap, yslow, vnom, snom, ybus, rows, qd, lam)
            logs.append({"frame": k, "yslow": yslow, "accepted": accepted, **info})
            updates.append(k)
        centers.append(center_pmu.copy()); hcenters.append(center_hidden.copy()); covs.append(center_cov.copy())
    pred, Pk, innovations, filter_time = kalman(A, C, L, y0, h0, y, np.asarray(centers), np.asarray(hcenters))
    return {"pred": pred, "state_cov": Pk, "innovations": innovations,
            "centers": np.asarray(centers), "hidden_centers": np.asarray(hcenters),
            "center_cov": np.asarray(covs), "updates": updates, "logs": logs,
            "map_time": total_map_time, "filter_time": filter_time}


def frozen_case(A, C, L, y0, h0, y):
    centers = np.repeat(y0[None, :], len(y), axis=0)
    hcenters = np.repeat(h0[None, :], len(y), axis=0)
    p, P, inn, rt = kalman(A, C, L, y0, h0, y, centers, hcenters)
    return {"pred": p, "state_cov": P, "innovations": inn, "centers": centers,
            "hidden_centers": hcenters, "center_cov": np.zeros((len(y), 62, 62)),
            "updates": [], "logs": [], "map_time": 0.0, "filter_time": rt}


def oracle_case(A, C, L, y0, h0, y, truth):
    # Evaluation-only oracle: true center is read by this wrapper, never by
    # online_case.  The PMU target is the first observed frame.
    centers = np.repeat(y[0][None, :], len(y), axis=0)
    hcenters = np.repeat(truth[0][None, :], len(y), axis=0)
    p, P, inn, rt = kalman(A, C, L, y0, h0, y, centers, hcenters)
    return {"pred": p, "state_cov": P, "innovations": inn, "centers": centers,
            "hidden_centers": hcenters, "center_cov": np.zeros((len(y), 62, 62)),
            "updates": [], "logs": [], "map_time": 0.0, "filter_time": rt}


def uncertainty(out, truth, L):
    vals = []; nll = []; nees = []
    cov = {0.50: [], 0.90: [], 0.95: []}; q = {0.50: 0.67449, 0.90: 1.64485, 0.95: 1.95996}
    for k in range(len(truth)):
        # Propagate Kalman deviation covariance and the last static-center
        # Laplace covariance.  Cross-covariance is conservatively omitted and
        # reported as an approximation in the report.
        vh = L @ out["state_cov"][k] @ L.T + out["center_cov"][k]
        e = out["pred"][k] - truth[k]
        er = np.r_[e.real, e.imag]
        var = np.maximum(np.diag(vh), 1e-12)
        z = np.abs(er) / np.sqrt(var)
        vals.extend(z.tolist()); nll.extend((0.5 * (er * er / var + np.log(2 * np.pi * var))).tolist())
        nees.append(float(er @ np.linalg.pinv(vh, rcond=1e-10) @ er / max(len(er), 1)))
        for p, qq in q.items(): cov[p].extend((z <= qq).tolist())
    return {"NLL": float(np.mean(nll)), "NEES_like": float(np.mean(nees)),
            "coverage50": float(np.mean(cov[0.50])), "coverage90": float(np.mean(cov[0.90])),
            "coverage95": float(np.mean(cov[0.95]))}


def bootstrap_ci(x: np.ndarray, seed=20260912):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x): return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed); b = np.median(rng.choice(x, (2000, len(x)), replace=True), axis=1)
    return float(np.median(x)), float(np.quantile(b, .025)), float(np.quantile(b, .975))


def main():
    A, C, L, y0, h0 = load_model()
    vnom, _s, _y, _br, meta = h6.load_nominal()
    ybus, _ = h6.build_pd_ybus(); snom = vnom * np.conj(ybus @ vnom)
    rows = h6.load_branch_rows(vnom, y0)
    dev = load_cases("e06i_dev_manifest.csv"); test = load_cases("e06i_test_manifest.csv")
    if len(dev) == 0 or len(test) == 0:
        raise RuntimeError("E06-I manifests/cases are missing")

    # Selection is DEV-only and uses one trajectory per scale to keep the
    # preregistered extractor/cadence grid independent of TEST.
    dev_subset = []
    ddf = pd.DataFrame(dev[0][0:1]) if False else None
    seen = set()
    for case in dev:
        m = float(case[0].m)
        if m not in seen:
            dev_subset.append(case); seen.add(m)
    sel_rows = []
    for ex in EXTRACTORS:
        for w in WINDOWS:
            for cdc in CADENCES:
                vals = []; delays = []; runtimes = []
                for row, y, truth in dev_subset:
                    out = online_case(A, C, L, y0, h0, y, 1.0, 1e-2, ex, w, cdc, vnom, snom, ybus, rows)
                    vals.append(metrics(out["pred"], truth)["TVE_fraction"])
                    delays.append(next((u["frame"] for u in out["logs"] if u["accepted"]), len(y)))
                    runtimes.append(1000 * out["map_time"] / max(len(y), 1))
                sel_rows.append({"extractor": ex, "window": w, "cadence": cdc,
                                 "n_cases": len(vals), "median_tve_fraction": float(np.median(vals)),
                                 "median_first_update_frame": float(np.median(delays)),
                                 "median_map_ms_per_frame": float(np.median(runtimes))})
    sdf = pd.DataFrame(sel_rows)
    # Prefer reconstruction and then causal latency/runtime.  This is not
    # tuned on hidden TEST outcomes.
    chosen = sdf.sort_values(["median_tve_fraction", "median_first_update_frame", "median_map_ms_per_frame"]).iloc[0]
    ex, w, cdc = str(chosen.extractor), int(chosen.window), int(chosen.cadence)
    sdf["selected"] = (sdf.extractor == ex) & (sdf.window == w) & (sdf.cadence == cdc)
    sdf.to_csv(RES / "e06i_dev_selection.csv", index=False)

    rows_out = []; closure_rows = []; center_rows = []; tr_rows = []; unc_rows = []; run_rows = []; reject_rows = []
    for row, y, truth in test:
        r0 = frozen_case(A, C, L, y0, h0, y)
        r1 = oracle_case(A, C, L, y0, h0, y, truth)
        r2 = online_case(A, C, L, y0, h0, y, 1.0, 1e-2, ex, w, cdc, vnom, snom, ybus, rows)
        outs = [("R0_FROZEN_B2", r0), ("R1_ORACLE_CENTER", r1), ("R2_PF_ONLINE", r2)]
        mets = {}
        for method, out in outs:
            met = metrics(out["pred"], truth); mets[method] = met
            rows_out.append({"case_id": row.case_id, "m": row.m, "seed": row.seed, "method": method,
                             "extractor": ex if method == "R2_PF_ONLINE" else "NA",
                             "window": w if method == "R2_PF_ONLINE" else 0,
                             "cadence": cdc if method == "R2_PF_ONLINE" else 0, **met,
                             "updates": len(out["updates"]), "accepted_updates": sum(bool(x.get("accepted", False)) for x in out["logs"]),
                             "convergence_rate": (np.mean([x.get("accepted", False) for x in out["logs"]]) if out["logs"] else 1.0)})
            um = uncertainty(out, truth, L) if method == "R2_PF_ONLINE" else {"NLL": np.nan, "NEES_like": np.nan, "coverage50": np.nan, "coverage90": np.nan, "coverage95": np.nan}
            if method == "R2_PF_ONLINE":
                unc_rows.append({"case_id": row.case_id, "m": row.m, **um})
                run_rows.append({"case_id": row.case_id, "m": row.m, "map_update_ms": 1000 * out["map_time"] / max(len(out["updates"]), 1),
                                 "filter_ms": 1000 * out["filter_time"] / len(y), "blocking_ms_on_update": 1000 * out["map_time"] / max(len(out["updates"]), 1),
                                 "amortized_ms_per_frame": 1000 * out["filter_time"] / len(y) + 1000 * out["map_time"] / len(y),
                                 "n_updates": len(out["updates"])})
                reject_rows.extend({"case_id": row.case_id, "m": row.m, **x} for x in out["logs"] if not x.get("accepted", False))
        den = mets["R0_FROZEN_B2"]["TVE_fraction"] - mets["R1_ORACLE_CENTER"]["TVE_fraction"]
        num = mets["R0_FROZEN_B2"]["TVE_fraction"] - mets["R2_PF_ONLINE"]["TVE_fraction"]
        closure_rows.append({"case_id": row.case_id, "m": row.m, "R0_TVE_percent": 100 * mets["R0_FROZEN_B2"]["TVE_fraction"],
                             "R1_TVE_percent": 100 * mets["R1_ORACLE_CENTER"]["TVE_fraction"], "R2_TVE_percent": 100 * mets["R2_PF_ONLINE"]["TVE_fraction"],
                             "oracle_gap": den, "closure_online": num / den if den > DEN_FLOOR else np.nan,
                             "status": "OK" if den > DEN_FLOOR else "LOW_ORACLE_GAP"})
        final = -1
        center_rows.append({"case_id": row.case_id, "m": row.m, "observed_center_RMSE_final": float(np.sqrt(np.mean(np.abs(cplx(r2["centers"][final][:16].reshape(8, 2))) - cplx(y[0, :16].reshape(8, 2))) ** 2)),
                            "hidden_center_TVE_percent_final": metrics(r2["hidden_centers"][final], truth[0])["TVE_percent"],
                            "hidden_center_angle_RMSE_deg_final": metrics(r2["hidden_centers"][final], truth[0])["angle_RMSE_deg"],
                            "first_accepted_update_frame": next((u["frame"] for u in r2["logs"] if u.get("accepted", False)), np.nan),
                            "n_updates": len(r2["updates"])})
        for name, out in outs:
            for label, sl in [("0_0p5s", slice(0, 16)), ("0p5_1p5s", slice(16, 46)), ("post_settling", slice(46, None))]:
                tr_rows.append({"case_id": row.case_id, "m": row.m, "method": name, "interval": label,
                                **metrics(out["pred"][sl], truth[sl])})

    per = pd.DataFrame(rows_out); per.to_csv(RES / "e06i_test_per_case.csv", index=False)
    cdf = pd.DataFrame(closure_rows); cdf.to_csv(RES / "e06i_oracle_closure.csv", index=False)
    pd.DataFrame(center_rows).to_csv(RES / "e06i_center_tracking.csv", index=False)
    pd.DataFrame(tr_rows).to_csv(RES / "e06i_time_resolved.csv", index=False)
    pd.DataFrame(unc_rows).to_csv(RES / "e06i_uncertainty.csv", index=False)
    pd.DataFrame(run_rows).to_csv(RES / "e06i_runtime.csv", index=False)
    pd.DataFrame(reject_rows).to_csv(RES / "e06i_rejections.csv", index=False)

    valid = per[per.method == "R2_PF_ONLINE"].convergence_rate
    focus = cdf[(cdf.m > 0) & np.isfinite(cdf.closure_online)]
    by_scale = []
    for m, g in focus.groupby("m"):
        med, lo, hi = bootstrap_ci(g.closure_online.to_numpy())
        by_scale.append({"m": m, "n": len(g), "closure_median": med, "closure_ci95_low": lo, "closure_ci95_high": hi})
    by_scale_df = pd.DataFrame(by_scale); by_scale_df.to_csv(RES / "e06i_closure.csv", index=False)
    closure = float(np.median(focus.closure_online)) if len(focus) else np.nan
    r2_nom = per[(per.method == "R2_PF_ONLINE") & (per.m == 0)].TVE_fraction.median()
    r0_nom = per[(per.method == "R0_FROZEN_B2") & (per.m == 0)].TVE_fraction.median()
    nominal = r2_nom / max(r0_nom, 1e-12) - 1.0
    nominal_delta_percent = 100.0 * (r2_nom - r0_nom)
    nominal_center = pd.DataFrame(center_rows).query("m == 0").hidden_center_TVE_percent_final.mean()
    status_oracle = "STRONG" if closure >= .8 else ("MODERATE" if closure >= .5 else "WEAK")
    status_online = "PASS" if closure >= .8 and float(valid.mean()) >= .95 else ("PARTIAL" if closure >= .5 else "FAIL")
    status_safe = "PASS" if abs(nominal) < .10 and nominal_center < .05 else "FAIL"
    rt = pd.DataFrame(run_rows)
    summary = {"dev_cases": len(dev), "test_cases": len(test), "selected_extractor": ex, "selected_window": w, "selected_cadence": cdc,
               "test_convergence_rate": float(valid.mean()), "closure_online_median": closure, "nominal_relative_tve_change": float(nominal),
               "nominal_absolute_tve_delta_percent": float(nominal_delta_percent),
               "nominal_center_drift_tve_percent": float(nominal_center), "median_map_update_ms": float(rt.map_update_ms.median()),
               "p95_map_update_ms": float(rt.map_update_ms.quantile(.95)), "median_amortized_ms_per_frame": float(rt.amortized_ms_per_frame.median()),
               "ONLINE_M6_RECENTERING": status_online, "ORACLE_RECOVERY_CAPTURED": status_oracle,
               "CAUSAL_CENTER_EXTRACTION": ("PASS" if closure >= .8 and float(valid.mean()) >= .95 else ("PARTIAL" if closure >= .5 and float(valid.mean()) >= .95 else "FAIL")),
               "NOMINAL_SAFETY": status_safe, "REALTIME_AVERAGE": "PASS" if float(rt.amortized_ms_per_frame.median()) <= 33.333 else "FAIL",
               "HARD_REALTIME_BLOCKING": "PASS" if float(rt.blocking_ms_on_update.median()) <= 33.333 else "FAIL",
               "UNCERTAINTY": "PARTIAL", "NONLINEAR_DAE_FIXED_LAG_NEEDED": "NOT_YET_JUSTIFIED"}
    pd.DataFrame([summary]).to_csv(RES / "e06i_summary.csv", index=False)

    # Compact plots required for the dossier.
    g = per.groupby(["m", "method"], as_index=False).TVE_percent.median(); plt.figure(figsize=(8, 4))
    for method, z in g.groupby("method"): plt.plot(z.m, z.TVE_percent, "o-", label=method)
    plt.xlabel("mismatch scale"); plt.ylabel("median hidden TVE (%)"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(PLOTS / "e06i_tve_frozen_oracle_online.png", dpi=150); plt.close()
    if len(by_scale_df):
        plt.figure(figsize=(6, 4)); plt.errorbar(by_scale_df.m, by_scale_df.closure_median, yerr=[by_scale_df.closure_median - by_scale_df.closure_ci95_low, by_scale_df.closure_ci95_high - by_scale_df.closure_median], fmt="o-"); plt.axhline(.8, ls="--", c="g"); plt.axhline(.5, ls="--", c="orange"); plt.ylabel("online oracle closure"); plt.xlabel("mismatch scale"); plt.tight_layout(); plt.savefig(PLOTS / "e06i_oracle_closure.png", dpi=150); plt.close()
    ce = pd.DataFrame(center_rows); plt.figure(figsize=(7, 4)); ce.groupby("m").hidden_center_TVE_percent_final.median().plot(marker="o"); plt.ylabel("final hidden-center TVE (%)"); plt.xlabel("mismatch scale"); plt.tight_layout(); plt.savefig(PLOTS / "e06i_center_tracking.png", dpi=150); plt.close()
    if len(rt):
        plt.figure(figsize=(7, 4)); rt.groupby("n_updates").amortized_ms_per_frame.median().plot(marker="o"); plt.axhline(33.333, ls="--", c="k"); plt.xlabel("updates per trajectory"); plt.ylabel("amortized ms/frame"); plt.tight_layout(); plt.savefig(PLOTS / "e06i_runtime.png", dpi=150); plt.close()

    report = f"""# E06-I — causal online M6 operating-point recentering

## Scope and leakage contract

Fresh PowerDynamics-native M6 trajectories were frozen before selection: **{len(dev)} DEV** and **{len(test)} TEST** cases (m=0, 0.5, 1.0, 1.5), disjoint from E04/E06-D/E06-E/E06-F/E06-G/E06-H. The R2 path receives only the streaming 32-channel PMU vector, nominal matrices, and its previous estimate. It never receives hidden truth, changed dispatch, mismatch labels, future samples, or an oracle equilibrium.

## Selected causal configuration

DEV selected `{ex}`, window **{w} frames** ({w * DT:.3f} s), and update cadence **every {cdc} frames**. The causal target is formed only from samples up to the current frame. At each accepted update, the frozen E06-H physical 43-coordinate AC MAP is warm-started from the previous center; the nominal B2 Kalman filter estimates fast deviations between updates.

## TEST results

The online median oracle closure is **{closure:.3f}** (per-scale bootstrap intervals are in `e06i_closure.csv`). R0/R1/R2 distributions are in `e06i_test_per_case.csv`; time-resolved intervals (0–0.5 s, 0.5–1.5 s, post-settling) are in `e06i_time_resolved.csv`. TEST static MAP convergence rate is **{float(valid.mean()):.3f}**; rejected updates are explicit in `e06i_rejections.csv`.

Nominal m=0 TVE changes from {100*r0_nom:.4f}% (R0) to {100*r2_nom:.4f}% (R2), an absolute increase of **{nominal_delta_percent:.4f} percentage points** (relative ratio {100 * nominal:.1f}%). Final center drift is **{nominal_center:.4f}%**; the nominal safety gate is therefore **{status_safe}**.

## Causal delay, uncertainty, and runtime

Center tracking and first accepted-update delay are in `e06i_center_tracking.csv`; the selected window imposes a causal startup delay of approximately **{w * DT:.3f} s** before the first update. Laplace center covariance plus B2 covariance are propagated with center/state cross-covariance omitted; uncertainty remains **PARTIAL** (`e06i_uncertainty.csv`). Median MAP update time is **{float(rt.map_update_ms.median()):.2f} ms**, p95 **{float(rt.map_update_ms.quantile(.95)):.2f} ms**, and median amortized cost **{float(rt.amortized_ms_per_frame.median()):.2f} ms/frame** (`e06i_runtime.csv`). Blocking update time is reported separately; this is not yet a hard-real-time claim.

## Statuses

`ONLINE_M6_RECENTERING = {status_online}`  
`ORACLE_RECOVERY_CAPTURED = {status_oracle}`  
`CAUSAL_CENTER_EXTRACTION = {summary['CAUSAL_CENTER_EXTRACTION']}`  
`NOMINAL_SAFETY = {status_safe}`  
`REALTIME_AVERAGE = {summary['REALTIME_AVERAGE']}`  
`HARD_REALTIME_BLOCKING = {summary['HARD_REALTIME_BLOCKING']}`  
`UNCERTAINTY = PARTIAL`  
`NONLINEAR_DAE_FIXED_LAG_NEEDED = NOT_YET_JUSTIFIED`

No E04-B, events, ML, adaptive activation, M1/M7 transfer, or network-parameter estimation was run. No push was performed.
"""
    (REPORTS / "e06i_online_causal_recentering.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
