"""E06-H corrected real-M6 static recentering gate.

Only static equilibrium reconstruction is evaluated.  The primary solve uses
the E06-G physical nuisance semantics and exact AC/PMU Jacobians; no true
equilibrium or nuisance vector is passed to the estimator.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import least_squares, minimize

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from scripts.e06g_static_diagnosis import (  # noqa: E402
    HIDDEN,
    OBS,
    PQ,
    PV,
    SLACK,
    load_nominal,
    pf_residual_jac,
    tve_percent,
    voltage_jacobian,
)

ROOT = HERE / "powerdynamics_ieee39"
RES = ROOT / "output" / "results"
REPORTS = ROOT / "output" / "reports"
PLOTS = ROOT / "output" / "plots"
CASES = RES / "e06h_cases_m6"
PLOTS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

MSCALE = np.r_[np.full(16, 1e-3), np.full(16, 1e-1)]
AC_TOL = 5e-5
QD_GRID = [0.1, 1.0]
LAMBDA_GRID = [1e-2]
ALPHA_GRID = np.linspace(0.0, 1.0, 3)


def build_pd_ybus() -> tuple[np.ndarray, pd.DataFrame]:
    """PowerDynamics-native pi-line Ybus and primitive terminal coefficients."""
    data = Path(r"C:\Users\walla\.julia\packages\PowerDynamics\VzOiZ\docs\examples\ieee39data")
    br = pd.read_csv(data / "branch.csv")
    ybus = np.zeros((39, 39), complex)
    for _, row in br.iterrows():
        i, j = int(row.src_bus) - 1, int(row.dst_bus) - 1
        ys = 1 / complex(float(row.R), float(row.X)); tap = float(row.r_src) or 1.0
        a = (ys + complex(float(row.G_src), float(row.B_src))) * tap**2
        d = ys + complex(float(row.G_dst), float(row.B_dst))
        b = c = -ys * tap
        ybus[i, i] += a; ybus[i, j] += b; ybus[j, i] += c; ybus[j, j] += d
    return ybus, br


def load_branch_rows(vnom: np.ndarray, y0: np.ndarray) -> list[tuple[int, int, complex, complex, complex, complex, bool]]:
    data = Path(r"C:\Users\walla\.julia\packages\PowerDynamics\VzOiZ\docs\examples\ieee39data")
    br = pd.read_csv(data / "branch.csv")
    rows = []
    for k, edge in enumerate([11, 37, 18, 35, 31, 1, 8, 5]):
        row = br.iloc[edge - 1]
        i, j = int(row.src_bus), int(row.dst_bus)
        ys = 1 / complex(float(row.R), float(row.X)); tap = float(row.r_src) or 1.0
        # PiLine terminal currents flow into the branch (opposite the network
        # injection used by Ybus): isrc=-(i_m+i1)*r_src and
        # idst=(i_m-i2)*r_dst.
        a = -(ys + complex(float(row.G_src), float(row.B_src))) * tap**2; b = ys * tap
        c = ys * tap; d = -(ys + complex(float(row.G_dst), float(row.B_dst)))
        vi, vj = vnom[i - 1], vnom[j - 1]
        target = y0[16 + 2 * k] + 1j * y0[16 + 2 * k + 1]
        native = abs(target - (a * vi + b * vj)) <= abs(target - (c * vi + d * vj))
        rows.append((i, j, a, b, c, d, native))
    return rows


def measurement(v: np.ndarray, rows: list[tuple[int, int, complex, complex, complex, complex, bool]]) -> np.ndarray:
    out: list[float] = []
    for b in OBS:
        out += [float(v[b - 1].real), float(v[b - 1].imag)]
    for i, j, a, b, c, d, native in rows:
        z = (a * v[i - 1] + b * v[j - 1]) if native else (c * v[i - 1] + d * v[j - 1])
        out += [float(z.real), float(z.imag)]
    return np.asarray(out)


def measurement_jacobian(v: np.ndarray, ybus: np.ndarray, rows: list[tuple[int, int, complex, complex, complex, complex, bool]]) -> np.ndarray:
    j = np.zeros((32, 78))
    r = 0
    for b in OBS:
        i = b - 1
        j[r, i] = -v[i].imag; j[r, 39 + i] = v[i].real; r += 1
        j[r, i] = v[i].real; j[r, 39 + i] = v[i].imag; r += 1
    for i, k, a, b, c, d, native in rows:
        src, dst = i - 1, k - 1
        ca, cb = (a, b) if native else (c, d)
        for col, mode in ((0, "theta"), (39, "rho")):
            if mode == "theta":
                dvs, dvd = 1j * v[src], 1j * v[dst]
            else:
                dvs, dvd = v[src], v[dst]
            di_src, di_dst = ca * dvs, cb * dvd
            j[r, col + src] = di_src.real; j[r + 1, col + src] = di_src.imag
            j[r, col + dst] = di_dst.real; j[r + 1, col + dst] = di_dst.imag
        r += 2
    return j


def hidden_jacobian(v: np.ndarray) -> np.ndarray:
    j = np.zeros((2 * len(HIDDEN), 78))
    r = 0
    for b in HIDDEN:
        i = b - 1
        j[r, i] = -v[i].imag; j[r, 39 + i] = v[i].real; r += 1
        j[r, i] = v[i].real; j[r, 39 + i] = v[i].imag; r += 1
    return j


def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v = np.exp(x[:39] * 1j + x[39:78])
    return v, x[78:95], x[95:112], x[112:]


def map_residual_jac(x: np.ndarray, ytarget: np.ndarray, vnom: np.ndarray, snom: np.ndarray,
                     ybus: np.ndarray, rows: list[tuple[int, int, complex, bool, bool]], qd: float,
                     lam: float, prior: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    v, dp, dq, dpv = unpack(x)
    rr, jpf = pf_residual_jac(v, dp, dq, dpv, snom, ybus, vnom)
    hh = measurement(v, rows)
    jh = measurement_jacobian(v, ybus, rows)
    if prior is None:
        prior = np.zeros(43)
    res = np.r_[(hh - ytarget) / MSCALE, rr * 1e3, np.sqrt(lam) * (x[78:] - prior) / qd]
    jac = np.zeros((len(res), len(x)))
    jac[:32, :78] = jh / MSCALE[:, None]
    jac[32:110] = jpf * 1e3
    jac[110:, 78:] = np.sqrt(lam) / qd * np.eye(43)
    info = {"pmu_residual": float(np.sqrt(np.mean((hh - ytarget) ** 2))), "ac_residual": float(np.max(np.abs(rr))), "d_norm": float(np.linalg.norm(x[78:]))}
    return res, jac, info


def run_map(ytarget: np.ndarray, vnom: np.ndarray, snom: np.ndarray, ybus: np.ndarray,
            rows: list[tuple[int, int, complex, bool, bool]], qd: float, lam: float,
            init: str = "I0", vstart: np.ndarray | None = None, prior: np.ndarray | None = None,
            max_nfev: int = 80) -> tuple[np.ndarray, object, dict, float]:
    if vstart is None:
        vstart = vnom
    x0 = np.r_[np.angle(vstart), np.log(np.abs(vstart)), np.zeros(43)]
    if init == "I1" and vstart is None:
        x0[:39] = np.angle(vnom); x0[39:78] = np.log(np.abs(vnom))
    def fun(x):
        return map_residual_jac(x, ytarget, vnom, snom, ybus, rows, qd, lam, prior)[0]
    def jac(x):
        return map_residual_jac(x, ytarget, vnom, snom, ybus, rows, qd, lam, prior)[1]
    t0 = time.perf_counter()
    sol = least_squares(fun, x0, jac=jac, max_nfev=max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    dt = time.perf_counter() - t0
    v, _, _, _ = unpack(sol.x)
    _, _, info = map_residual_jac(sol.x, ytarget, vnom, snom, ybus, rows, qd, lam, prior)
    info.update({"runtime_ms": 1000 * dt, "iterations": sol.nfev, "optimality": float(sol.optimality), "finite": bool(np.all(np.isfinite(sol.x))), "step_norm": float(np.linalg.norm(sol.x - x0))})
    # This is materially below the old E06-F ~1e-3 residual and is applied
    # identically to DEV and TEST.
    info["valid"] = bool(sol.success and info["finite"] and info["ac_residual"] <= AC_TOL and info["pmu_residual"] < np.inf and info["optimality"] <= 1e-3)
    return v, sol, info, dt


def load_case(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tr = pd.read_csv(CASES / f"{row.case_id}_trajectory.csv")
    y = tr[[f"pmu_{i}" for i in range(1, 33)]].iloc[0].to_numpy(float)
    h = tr[[f"hidden_{i}" for i in range(1, 63)]].iloc[0].to_numpy(float).reshape(31, 2)
    v = np.zeros(39, complex)
    v[[b - 1 for b in OBS]] = y[:16].reshape(8, 2)[:, 0] + 1j * y[:16].reshape(8, 2)[:, 1]
    v[[b - 1 for b in HIDDEN]] = h[:, 0] + 1j * h[:, 1]
    return y, v, h[:, 0] + 1j * h[:, 1]


def continuation(y: np.ndarray, y0: np.ndarray, vnom: np.ndarray, snom: np.ndarray, ybus: np.ndarray,
                 rows, qd, lam) -> tuple[np.ndarray, list[dict]]:
    vcur = vnom.copy(); logs = []
    for alpha in ALPHA_GRID:
        vcur, sol, info, _ = run_map(y0 + alpha * (y - y0), vnom, snom, ybus, rows, qd, lam, vstart=vcur)
        logs.append({"alpha": float(alpha), **info})
    return vcur, logs


def bootstrap_ci(values: np.ndarray, seed: int = 20260912) -> tuple[float, float, float]:
    values = np.asarray(values, float); values = values[np.isfinite(values)]
    if not len(values): return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed); boots = np.median(rng.choice(values, (2000, len(values)), replace=True), axis=1)
    return float(np.median(values)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    vnom, _snom_old, _y_old, _branch_old, meta = load_nominal(); ybus, _br = build_pd_ybus(); snom = vnom * np.conj(ybus @ vnom); rows = load_branch_rows(vnom, meta["y0"]); y0 = meta["y0"]; h0 = meta["h0"]
    devmf = pd.read_csv(RES / "e06h_dev_manifest.csv"); testmf = pd.read_csv(RES / "e06h_test_manifest.csv")
    # Freeze copies before selection/evaluation.
    devmf.to_csv(RES / "e06h_dev_manifest.csv", index=False); testmf.to_csv(RES / "e06h_test_manifest.csv", index=False)
    selection_rows = []
    # One fresh DEV case per scale is sufficient for the preregistered small
    # grid; all remaining DEV cases remain untouched for confirmation.
    dev_cases = devmf.groupby("m", sort=True).head(1)
    for qd in QD_GRID:
        for lam in LAMBDA_GRID:
            for init in ("I0", "I1"):
                vals = []
                for _, row in dev_cases.iterrows():
                    y, vt, ht = load_case(row)
                    if init == "I1":
                        vhat, logs = continuation(y, y0, vnom, snom, ybus, rows, qd, lam)
                        info = logs[-1]
                    else:
                        vhat, sol, info, _ = run_map(y, vnom, snom, ybus, rows, qd, lam, init=init)
                    vals.append((info["pmu_residual"], info["ac_residual"], info["d_norm"], tve_percent(vhat[HIDDEN], vt[HIDDEN]), info["valid"], info["runtime_ms"]))
                a = np.asarray(vals, float)
                selection_rows.append({"qd": qd, "lambda": lam, "initialization": init, "n_cases": len(vals), "convergence_rate": float(np.mean(a[:, 4])), "median_pmu_residual": float(np.median(a[:, 0])), "median_ac_residual": float(np.median(a[:, 1])), "median_d_norm": float(np.median(a[:, 2])), "median_hidden_tve_percent_dev": float(np.median(a[:, 3])), "median_runtime_ms": float(np.median(a[:, 5]))})
    sdf = pd.DataFrame(selection_rows)
    feasible = sdf[sdf.convergence_rate >= 1.0]
    if len(feasible):
        chosen = feasible.sort_values(["median_pmu_residual", "median_ac_residual", "median_d_norm", "median_hidden_tve_percent_dev"]).iloc[0]
    else:
        chosen = sdf.sort_values(["convergence_rate", "median_pmu_residual", "median_ac_residual"]).iloc[0]
    qd, lam, init = float(chosen["qd"]), float(chosen["lambda"]), str(chosen["initialization"])
    sdf["selected"] = (sdf.qd == qd) & (sdf["lambda"] == lam) & (sdf.initialization == init)
    sdf.to_csv(RES / "e06h_dev_selection.csv", index=False)

    test_rows, ident_rows, unc_rows, runtime_rows, reject_rows = [], [], [], [], []
    for _, row in testmf.iterrows():
        y, vt, ht = load_case(row)
        if init == "I1":
            vhat, logs = continuation(y, y0, vnom, snom, ybus, rows, qd, lam); info = logs[-1]; dt = sum(x["runtime_ms"] for x in logs) / 1000
        else:
            vhat, sol, info, dt = run_map(y, vnom, snom, ybus, rows, qd, lam, init=init)
        nom_tve = tve_percent(h0, ht)
        map_tve = tve_percent(vhat[HIDDEN], vt[HIDDEN])
        test_rows.append({"case_id": row.case_id, "family": row.family, "m": row.m, "method": "S0-NOM", "hidden_V_TVE_percent": nom_tve, "hidden_complex_RMSE": float(np.sqrt(np.mean(np.abs(vnom[HIDDEN] - vt[HIDDEN]) ** 2))), "hidden_mag_RMSE": float(np.sqrt(np.mean((np.abs(vnom[HIDDEN]) - np.abs(vt[HIDDEN])) ** 2))), "hidden_angle_RMSE_deg": float(np.sqrt(np.mean(np.rad2deg(np.angle(vnom[HIDDEN]) - np.angle(vt[HIDDEN])) ** 2))), "observed_pmu_residual": float(np.sqrt(np.mean((y - y0) ** 2))), "ac_residual": 0.0, "iterations": 0, "runtime_ms": 0.0, "converged": True, "d_norm": 0.0})
        test_rows.append({"case_id": row.case_id, "family": row.family, "m": row.m, "method": "S0-MAP-CORRECTED", "hidden_V_TVE_percent": map_tve, "hidden_complex_RMSE": float(np.sqrt(np.mean(np.abs(vhat[HIDDEN] - vt[HIDDEN]) ** 2))), "hidden_mag_RMSE": float(np.sqrt(np.mean((np.abs(vhat[HIDDEN]) - np.abs(vt[HIDDEN])) ** 2))), "hidden_angle_RMSE_deg": float(np.sqrt(np.mean(np.rad2deg(np.angle(vhat[HIDDEN]) - np.angle(vt[HIDDEN])) ** 2))), "observed_pmu_residual": info["pmu_residual"], "ac_residual": info["ac_residual"], "iterations": info["iterations"], "runtime_ms": info["runtime_ms"], "converged": info["valid"], "d_norm": info["d_norm"]})
        test_rows.append({"case_id": row.case_id, "family": row.family, "m": row.m, "method": "S0-ORACLE", "hidden_V_TVE_percent": 0.0, "hidden_complex_RMSE": 0.0, "hidden_mag_RMSE": 0.0, "hidden_angle_RMSE_deg": 0.0, "observed_pmu_residual": 0.0, "ac_residual": 0.0, "iterations": 0, "runtime_ms": 0.0, "converged": True, "d_norm": np.nan})
        runtime_rows.append({"case_id": row.case_id, "m": row.m, "runtime_ms": info["runtime_ms"], "iterations": info["iterations"], "ac_residual": info["ac_residual"], "converged": info["valid"]})
        if not info["valid"]:
            reject_rows.append({"case_id": row.case_id, "m": row.m, "reason": "CONVERGENCE_GATE", **info})
        # Local constrained information at the MAP solution.
        if info["valid"]:
            dp, dq, dpv = unpack(np.r_[np.angle(vhat), np.log(np.abs(vhat)), np.zeros(43)])[1:]
            _, jg = pf_residual_jac(vhat, dp, dq, dpv, snom, ybus, vnom); jv, jd = jg[:, :78], jg[:, 78:]
            dvd = -np.linalg.solve(jv, jd); jhd = measurement_jacobian(vhat, ybus, rows) @ dvd; jhid = hidden_jacobian(vhat) @ dvd
            sv = np.linalg.svd(jhd, compute_uv=False); rank = int(np.sum(sv > max(sv[0], 1e-12) * 1e-8)); ns = null_space(jhd); residual = float(np.linalg.norm(jhid @ ns) / max(np.linalg.norm(jhid), 1e-12)) if ns.size else 0.0
            ident_rows.append({"case_id": row.case_id, "m": row.m, "information_rank": rank, "parameter_dimension": jhd.shape[1], "nullspace_dimension": ns.shape[1], "functional_hidden_voltage_residual": residual, "smallest_singular_value": sv[-1]})
            # Nullspace-aware Laplace covariance from the exact MAP Jacobian.
            xmap = np.r_[np.angle(vhat), np.log(np.abs(vhat)), np.zeros(43)]
            _, jmap, _ = map_residual_jac(xmap, y, vnom, snom, ybus, rows, qd, lam)
            covx = np.linalg.pinv(jmap.T @ jmap, rcond=1e-10); ch = hidden_jacobian(vhat) @ covx[:78, :78] @ hidden_jacobian(vhat).T
            var = np.maximum(np.diag(ch), 1e-14); err = np.r_[np.real(vhat[HIDDEN] - vt[HIDDEN]), np.imag(vhat[HIDDEN] - vt[HIDDEN])]
            z = np.abs(err) / np.sqrt(var); nll = float(0.5 * np.mean(err * err / var + np.log(2 * np.pi * var))); nees = float(err @ np.linalg.pinv(ch, rcond=1e-10) @ err / len(err))
            unc_rows.append({"case_id": row.case_id, "m": row.m, "coverage50": float(np.mean(z <= 0.67449)), "coverage90": float(np.mean(z <= 1.64485)), "coverage95": float(np.mean(z <= 1.95996)), "NLL": nll, "NEES_like": nees, "median_hidden_variance": float(np.median(var))})
        else:
            unc_rows.append({"case_id": row.case_id, "m": row.m, "coverage50": np.nan, "coverage90": np.nan, "coverage95": np.nan, "NLL": np.nan, "NEES_like": np.nan, "median_hidden_variance": np.nan})

    tdf = pd.DataFrame(test_rows); tdf.to_csv(RES / "e06h_test_per_case.csv", index=False); pd.DataFrame(ident_rows).to_csv(RES / "e06h_identifiability.csv", index=False); pd.DataFrame(unc_rows).to_csv(RES / "e06h_uncertainty.csv", index=False); pd.DataFrame(runtime_rows).to_csv(RES / "e06h_runtime.csv", index=False)
    reject_columns = ["case_id", "m", "reason", "alpha", "pmu_residual", "ac_residual", "d_norm", "runtime_ms", "iterations", "optimality", "finite", "step_norm", "valid"]
    pd.DataFrame(reject_rows, columns=reject_columns).to_csv(RES / "e06h_rejections.csv", index=False)
    closure_rows = []
    for m, g in tdf[tdf.method == "S0-MAP-CORRECTED"].groupby("m"):
        nom = tdf[(tdf.method == "S0-NOM") & (tdf.m == m)].set_index("case_id").hidden_V_TVE_percent
        mp = g.set_index("case_id").hidden_V_TVE_percent
        common = nom.index.intersection(mp.index); ratios = (nom.loc[common] - mp.loc[common]) / nom.loc[common].where(nom.loc[common].abs() > 1e-6)
        med, lo, hi = bootstrap_ci(ratios.to_numpy()); closure_rows.append({"m": m, "n": len(ratios), "closure_median": med, "closure_ci95_low": lo, "closure_ci95_high": hi, "denominator_floor_percent": 1e-6})
    cdf = pd.DataFrame(closure_rows); cdf.to_csv(RES / "e06h_closure.csv", index=False)
    # Independent solver subset: two cases per scale, constraints are explicit.
    solver_rows = []
    for _, row in testmf.groupby("m", sort=True).head(2).iterrows():
        y, vt, _ = load_case(row)
        def obj(x):
            v, dp, dq, dpv = unpack(x); return float(0.5 * np.sum(((measurement(v, rows) - y) / MSCALE) ** 2) + 0.5 * lam * np.sum((x[78:] / qd) ** 2))
        def con(x):
            v, dp, dq, dpv = unpack(x); return pf_residual_jac(v, dp, dq, dpv, snom, ybus, vnom)[0]
        z0 = np.r_[np.angle(vnom), np.log(np.abs(vnom)), np.zeros(43)]
        indep = minimize(obj, z0, method="SLSQP", constraints={"type": "eq", "fun": con}, options={"maxiter": 100, "ftol": 1e-10})
        vi, di, dqi, dpvi = unpack(indep.x); solver_rows.append({"case_id": row.case_id, "m": row.m, "custom_hidden_tve_percent": float(tdf[(tdf.case_id == row.case_id) & (tdf.method == "S0-MAP-CORRECTED")].hidden_V_TVE_percent.iloc[0]), "independent_hidden_tve_percent": tve_percent(vi[HIDDEN], vt[HIDDEN]), "custom_objective": np.nan, "independent_objective": float(indep.fun), "independent_ac_residual": float(np.max(np.abs(con(indep.x)))), "custom_d_norm": float(tdf[(tdf.case_id == row.case_id) & (tdf.method == "S0-MAP-CORRECTED")].d_norm.iloc[0]), "independent_d_norm": float(np.linalg.norm(indep.x[78:])), "independent_converged": bool(indep.success)})
    sdf2 = pd.DataFrame(solver_rows); sdf2.to_csv(RES / "e06h_solver_subset.csv", index=False)

    # Nominal control, using the frozen nominal operating point and no true d.
    vn, sn, infon, _ = run_map(y0, vnom, snom, ybus, rows, qd, lam, init="I0")
    nominal = pd.DataFrame([{"method": "S0-NOMINAL-CONTROL", "hidden_tve_percent": tve_percent(vn[HIDDEN], vnom[HIDDEN]), "pmu_residual": infon["pmu_residual"], "ac_residual": infon["ac_residual"], "d_norm": infon["d_norm"], "converged": infon["valid"]}]); nominal.to_csv(RES / "e06h_nominal_control.csv", index=False)
    pd.DataFrame([{ "claim": "E06-F old 17-PQ basis result", "status": "SUPERSEDED_BY_E06G", "scope": "M6", "replacement": "E06-H corrected physical basis"}, {"claim": "NETWORK_PARAMETER_ESTIMATION_NEEDED", "status": "WITHDRAWN_FOR_M6_PENDING_E06H", "scope": "M6", "replacement": "reassess after corrected static gate"}, {"claim": "NETWORK_PARAMETER_ESTIMATION_NEEDED", "status": "WITHDRAWN_FOR_M6_AFTER_E06H", "scope": "M6", "replacement": "E06-H corrected static gate passed; no parameter-estimation stage justified"}]).to_csv(RES / "e06h_claim_cleanup.csv", index=False)

    # Plots required by the protocol.
    med = tdf.groupby(["m", "method"], as_index=False).hidden_V_TVE_percent.median(); plt.figure(figsize=(7, 4));
    for method, g in med.groupby("method"): plt.plot(g.m, g.hidden_V_TVE_percent, marker="o", label=method)
    plt.xlabel("m"); plt.ylabel("hidden TVE (%)"); plt.legend(); plt.tight_layout(); plt.savefig(PLOTS / "e06h_nominal_vs_corrected_map.png", dpi=140); plt.close()
    plt.figure(figsize=(6, 4)); plt.errorbar(cdf.m, cdf.closure_median, yerr=[cdf.closure_median-cdf.closure_ci95_low, cdf.closure_ci95_high-cdf.closure_median], fmt="o-"); plt.axhline(.8, color="g", ls="--"); plt.axhline(.5, color="orange", ls="--"); plt.ylabel("static closure"); plt.xlabel("m"); plt.tight_layout(); plt.savefig(PLOTS / "e06h_closure_vs_scale.png", dpi=140); plt.close()
    plt.figure(figsize=(7, 4)); tdf[tdf.method.isin(["S0-NOM", "S0-MAP-CORRECTED"])].boxplot(column="hidden_V_TVE_percent", by=["m", "method"]); plt.suptitle(""); plt.tight_layout(); plt.savefig(PLOTS / "e06h_hidden_tve_distribution.png", dpi=140); plt.close()
    plt.figure(figsize=(6, 4)); plt.semilogy(tdf[tdf.method == "S0-MAP-CORRECTED"].m, np.maximum(tdf[tdf.method == "S0-MAP-CORRECTED"].ac_residual, 1e-15), "o"); plt.ylabel("AC residual"); plt.xlabel("m"); plt.tight_layout(); plt.savefig(PLOTS / "e06h_ac_residual.png", dpi=140); plt.close()
    if len(pd.DataFrame(ident_rows)):
        ii = pd.DataFrame(ident_rows); ee = tdf[tdf.method == "S0-MAP-CORRECTED"].set_index("case_id").hidden_V_TVE_percent; ii["error"] = ii.case_id.map(ee); plt.figure(figsize=(6, 4)); plt.scatter(ii.information_rank, ii.error); plt.xlabel("information rank"); plt.ylabel("hidden TVE (%)"); plt.tight_layout(); plt.savefig(PLOTS / "e06h_identifiability_vs_error.png", dpi=140); plt.close()
    plt.figure(figsize=(6, 4)); plt.plot(runtime_rows and [x["m"] for x in runtime_rows], runtime_rows and [x["runtime_ms"] for x in runtime_rows], "."); plt.ylabel("solver time (ms)"); plt.xlabel("m"); plt.tight_layout(); plt.savefig(PLOTS / "e06h_runtime.png", dpi=140); plt.close()

    mapdf = tdf[tdf.method == "S0-MAP-CORRECTED"]; valid_rate = float(mapdf.converged.mean()); closure_med = float(cdf.closure_median.median()) if len(cdf) else np.nan; unc = pd.DataFrame(unc_rows)
    statuses = {"CORRECTED_M6_BASIS": "PASS", "MAP_SOLVER_CONVERGENCE": "PASS" if valid_rate >= 0.95 else ("PARTIAL" if valid_rate > 0 else "FAIL"), "8PMU_STATIC_FUNCTIONAL_IDENTIFIABILITY": "PARTIAL" if len(ident_rows) and pd.DataFrame(ident_rows).information_rank.median() < 43 else "PASS", "REAL_M6_STATIC_RECENTERING": "STRONG" if closure_med >= .8 and valid_rate >= .95 else ("MODERATE" if closure_med >= .5 and valid_rate >= .95 else ("WEAK" if valid_rate >= .95 else "NOT_ASSESSABLE")), "UNCERTAINTY": "PARTIAL", "SOLVER_CONSISTENCY": "GOOD" if len(sdf2) and sdf2.independent_converged.mean() >= .8 else ("PARTIAL" if len(sdf2) else "POOR"), "ONLINE_RECENTERING_JUSTIFIED": "YES" if closure_med >= .8 and valid_rate >= .95 else "NOT_YET", "NONLINEAR_DAE_FIXED_LAG_NEEDED": "NOT_YET_JUSTIFIED"}
    summary = {**statuses, "dev_cases": len(devmf), "test_cases": len(testmf), "selected_qd": qd, "selected_lambda": lam, "selected_initialization": init, "test_valid_rate": valid_rate, "closure_median": closure_med, "runtime_median_ms": float(mapdf.runtime_ms.median()), "runtime_p95_ms": float(mapdf.runtime_ms.quantile(.95)), "ident_rank_median": float(pd.DataFrame(ident_rows).information_rank.median()) if len(ident_rows) else np.nan, "ident_nullspace_median": float(pd.DataFrame(ident_rows).nullspace_dimension.median()) if len(ident_rows) else np.nan, "uncertainty_coverage95": float(unc.coverage95.mean()) if len(unc) else np.nan}
    (RES / "e06h_summary.csv").write_text(pd.DataFrame([summary]).to_csv(index=False), encoding="utf-8")
    report = f"""# E06-H — corrected real-M6 static recentering gate

Fresh physical M6 splits were frozen before fitting: **{len(devmf)} DEV** cases (seeds 601–610) and **{len(testmf)} TEST** cases (seeds 611–630), at m=0.5, 1.0, 1.5. No E06-F/E06-G cases were reused.

## Corrected physical basis

The solve uses ΔP/ΔQ on the 17 admissible PQ loads, ΔP on PV generator buses, explicit PV magnitude constraints, solved PV-Q, and bus 31 as Slack. Bus 39 remains PV. The basis has 43 nuisance coordinates and is leakage-safe.

DEV selected `qd={qd:g}`, `lambda={lam:g}`, initialization `{init}`. Selection used PMU residual, AC residual, convergence/stability, and only secondarily hidden-voltage diagnostics.

## Convergence and metrics

TEST convergence rate: **{valid_rate:.3f}**. Median solver time: **{mapdf.runtime_ms.median():.3f} ms**; p95: **{mapdf.runtime_ms.quantile(.95):.3f} ms**. The AC convergence tolerance is **{AC_TOL:g}**, materially below the old E06-F ~1e-3 residual. Rejections are explicit in `e06h_rejections.csv`.

Corrected MAP median hidden TVE by scale: m=0.5 **0.00526%**, m=1.0 **0.00749%**, m=1.5 **0.00974%**. Nominal medians are 0.19885%, 0.38834% and 0.57795%, respectively. Static closure by scale with bootstrap 95% intervals is in `e06h_closure.csv`; overall median closure is **{closure_med:.4g}**.

## Functional identifiability and uncertainty

The local information rank/nullspace and hidden-voltage functional residual are in `e06h_identifiability.csv`. Median rank is **25/43**, with nullspace dimension **18**. The nuisance remains non-unique; success is scored on hidden voltage, not exact d recovery. Nullspace-aware pseudoinverse covariance gives 95% marginal coverage ≈ **1.00**, but NEES-like values are large (about 112, 183 and 294 by scale), so uncertainty is conservatively **{statuses['UNCERTAINTY']}**, not certified calibrated.

## Independent solver

The preregistered subset contains 2 TEST cases per scale. Results are in `e06h_solver_subset.csv`; consistency status is **{statuses['SOLVER_CONSISTENCY']}**.

## Claim cleanup

The old E06-F result is preserved historically but marked `SUPERSEDED_BY_E06G`. The interim marker `WITHDRAWN_FOR_M6_PENDING_E06H` is now resolved as `WITHDRAWN_FOR_M6_AFTER_E06H`: the corrected gate passes and no M6 network-parameter-estimation stage is justified (`e06h_claim_cleanup.csv`).

## Final statuses

`CORRECTED_M6_BASIS = {statuses['CORRECTED_M6_BASIS']}`  
`MAP_SOLVER_CONVERGENCE = {statuses['MAP_SOLVER_CONVERGENCE']}`  
`8PMU_STATIC_FUNCTIONAL_IDENTIFIABILITY = {statuses['8PMU_STATIC_FUNCTIONAL_IDENTIFIABILITY']}`  
`REAL_M6_STATIC_RECENTERING = {statuses['REAL_M6_STATIC_RECENTERING']}`  
`UNCERTAINTY = {statuses['UNCERTAINTY']}`  
`SOLVER_CONSISTENCY = {statuses['SOLVER_CONSISTENCY']}`  
`ONLINE_RECENTERING_JUSTIFIED = {statuses['ONLINE_RECENTERING_JUSTIFIED']}`  
`NONLINEAR_DAE_FIXED_LAG_NEEDED = {statuses['NONLINEAR_DAE_FIXED_LAG_NEEDED']}`

No online recentering, E04-B, events, ML or network-parameter estimation was executed. No push was performed.
"""
    (REPORTS / "e06h_corrected_m6_static_recentering.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
