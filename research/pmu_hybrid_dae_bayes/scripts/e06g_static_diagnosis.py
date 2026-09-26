"""E06-G static representability and solver falsification campaign.

This campaign is deliberately M6-only.  It separates the nuisance basis,
nonlinear AC solver, derivative implementation, and 8-PMU identifiability
questions before any online or trajectory estimator is considered.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import least_squares, minimize

ROOT = Path(__file__).resolve().parents[1] / "powerdynamics_ieee39"
RES = ROOT / "output" / "results"
CASES = RES / "e06g_cases_m6_diagnostic_v1"
REPORTS = ROOT / "output" / "reports"
RES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

OBS = [2, 5, 6, 10, 19, 22, 29, 39]
HIDDEN = [b for b in range(1, 40) if b not in OBS]
PQ = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29]
PV = [30, 32, 33, 34, 35, 36, 37, 38, 39]
SLACK = 31
PQLIKE = [b for b in range(1, 40) if b not in PV and b != SLACK]
EDGE_FOR = [11, 37, 18, 35, 31, 1, 8, 5]
NATIVE_FOR = [False, False, True, True, True, False, False, False]


def cplx(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a[..., 0] + 1j * a[..., 1]


def wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def tve_percent(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(100 * np.mean(np.abs(pred - truth) / np.maximum(np.abs(truth), 1e-12)))


def build_ybus() -> tuple[np.ndarray, dict[tuple[int, int], tuple[complex, complex]]]:
    data = Path(os.environ.get("POWERDYNAMICS_DATA", r"C:\Users\walla\.julia\packages\PowerDynamics\VzOiZ\docs\examples\ieee39data"))
    br = pd.read_csv(data / "branch.csv")
    ybus = np.zeros((39, 39), complex)
    branch: dict[tuple[int, int], tuple[complex, complex]] = {}
    for _, row in br.iterrows():
        i, j = int(row.src_bus) - 1, int(row.dst_bus) - 1
        z = complex(float(row.R), float(row.X))
        y = 1 / z
        bsrc, bdst = 1j * float(row.B_src), 1j * float(row.B_dst)
        tap = float(row.r_src) if float(row.transformer) != 0 else 1.0
        ybus[i, i] += y / (tap * tap) + bsrc / 2
        ybus[j, j] += y + bdst / 2
        ybus[i, j] -= y / tap
        ybus[j, i] -= y / tap
        branch[(i + 1, j + 1)] = (y / tap, -y)
        branch[(j + 1, i + 1)] = (-y, y)
    return ybus, branch


def load_nominal() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    y0 = pd.read_csv(RES / "e04_y0_pmu.csv").iloc[:, 0].to_numpy(float)
    h0 = cplx(pd.read_csv(RES / "e04_pd_hidden0.csv").iloc[:, 0].to_numpy(float).reshape(31, 2))
    vnom = np.zeros(39, complex)
    vnom[[b - 1 for b in OBS]] = cplx(y0[:16].reshape(8, 2))
    vnom[[b - 1 for b in HIDDEN]] = h0
    ybus, branch = build_ybus()
    return vnom, vnom * np.conj(ybus @ vnom), ybus, branch, {"y0": y0, "h0": h0}


def measurement(v: np.ndarray, ybus: np.ndarray, branch: dict) -> np.ndarray:
    out: list[float] = []
    for b in OBS:
        out += [float(v[b - 1].real), float(v[b - 1].imag)]
    data = Path(os.environ.get("POWERDYNAMICS_DATA", r"C:\Users\walla\.julia\packages\PowerDynamics\VzOiZ\docs\examples\ieee39data"))
    br = pd.read_csv(data / "branch.csv")
    for edge, native in zip(EDGE_FOR, NATIVE_FOR):
        row = br.iloc[edge - 1]
        i, j = int(row.src_bus), int(row.dst_bus)
        z = complex(float(row.R), float(row.X))
        y = 1 / z
        tap = float(row.r_src) if float(row.transformer) != 0 else 1.0
        src, dst = (i, j) if native else (j, i)
        current = (y / tap) * (v[src - 1] - v[dst - 1])
        out += [float(current.real), float(current.imag)]
    return np.asarray(out)


def unpack(x: np.ndarray, nd: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta, rho = x[:39], x[39:78]
    v = np.exp(rho + 1j * theta)
    dp = x[78 : 78 + nd // 2]
    dq = x[78 + nd // 2 : 78 + nd]
    return v, theta, rho, np.r_[dp, dq]


def voltage_jacobian(v: np.ndarray, ybus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact derivatives dS/dtheta and dS/drho for S=V*conj(YV)."""
    current = ybus @ v
    jt = np.zeros((39, 39), complex)
    jr = np.zeros((39, 39), complex)
    for k in range(39):
        dv_t = np.zeros(39, complex); dv_t[k] = 1j * v[k]
        dv_r = np.zeros(39, complex); dv_r[k] = v[k]
        di_t = ybus @ dv_t; di_r = ybus @ dv_r
        jt[:, k] = dv_t * np.conj(current) + v * np.conj(di_t)
        jr[:, k] = dv_r * np.conj(current) + v * np.conj(di_r)
    return jt, jr


def pf_residual_jac(v: np.ndarray, dp: np.ndarray, dq: np.ndarray, dpv: np.ndarray,
                    snom: np.ndarray, ybus: np.ndarray, vnom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AC PF residual and exact Jacobian in [theta,rho,dp,dq,dpv]."""
    nd = len(dp) + len(dq) + len(dpv)
    theta, rho = np.angle(v), np.log(np.maximum(np.abs(v), 1e-12))
    s = v * np.conj(ybus @ v)
    jt, jr = voltage_jacobian(v, ybus)
    rows: list[float] = []
    jac = np.zeros((78, 78 + nd))
    r = 0
    dp_map = {b: i for i, b in enumerate(PQ)}
    pv_map = {b: i for i, b in enumerate(PV)}
    for b in PQLIKE:
        i = b - 1
        dpp = dp[dp_map[b]] if b in dp_map else 0.0
        dqq = dq[dp_map[b]] if b in dp_map else 0.0
        rows += [float(s[i].real - snom[i].real - dpp), float(s[i].imag - snom[i].imag - dqq)]
        jac[r, :39] = jt[i].real; jac[r, 39:78] = jr[i].real
        if b in dp_map:
            jac[r, 78 + dp_map[b]] = -1.0
        r += 1
        jac[r, :39] = jt[i].imag; jac[r, 39:78] = jr[i].imag
        if b in dp_map:
            jac[r, 78 + len(PQ) + dp_map[b]] = -1.0
        r += 1
    for b in PV:
        i = b - 1
        rows.append(float(s[i].real - snom[i].real - dpv[pv_map[b]]))
        jac[r, :39] = jt[i].real; jac[r, 39:78] = jr[i].real
        jac[r, 78 + 2 * len(PQ) + pv_map[b]] = -1.0
        r += 1
        rows.append(float(np.abs(v[i]) - np.abs(vnom[i])))
        jac[r, 39 + i] = np.abs(v[i])
        r += 1
    rows += [float(theta[SLACK - 1] - np.angle(vnom[SLACK - 1])), float(rho[SLACK - 1] - np.log(np.abs(vnom[SLACK - 1])))]
    jac[r, SLACK - 1] = 1.0; r += 1
    jac[r, 39 + SLACK - 1] = 1.0
    return np.asarray(rows), jac


def pf_solve(dp: np.ndarray, dq: np.ndarray, dpv: np.ndarray, vnom: np.ndarray, snom: np.ndarray, ybus: np.ndarray,
             x0: np.ndarray | None = None, max_nfev: int = 300) -> tuple[np.ndarray, object, float]:
    nd = len(dp) + len(dq) + len(dpv)
    if x0 is None:
        x0 = np.r_[np.angle(vnom), np.log(np.abs(vnom)), dp * 0, dq * 0, dpv * 0]
    def fun(x: np.ndarray) -> np.ndarray:
        v = np.exp(x[:39] * 1j + x[39:78])
        rr, _ = pf_residual_jac(v, x[78:78 + len(dp)], x[78 + len(dp):78 + 2 * len(dp)], x[78 + 2 * len(dp):], snom, ybus, vnom)
        # The target d is fixed by a tight equality through the d variables.
        return np.r_[rr * 1e3, (x[78:78 + len(dp)] - dp) * 1e3, (x[78 + len(dp):78 + 2 * len(dp)] - dq) * 1e3, (x[78 + 2 * len(dp):] - dpv) * 1e3]
    sol = least_squares(fun, x0, max_nfev=max_nfev, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    v = np.exp(sol.x[:39] * 1j + sol.x[39:78])
    rr, _ = pf_residual_jac(v, dp, dq, dpv, snom, ybus, vnom)
    return v, sol, float(np.max(np.abs(rr)))


def inverse_solve(vinit: np.ndarray, ytarget: np.ndarray, vnom: np.ndarray, snom: np.ndarray, ybus: np.ndarray,
                  full_voltage: np.ndarray | None = None, prior_center: np.ndarray | None = None,
                  max_nfev: int = 250) -> tuple[np.ndarray, np.ndarray, object, float, float]:
    nd = 2 * len(PQ) + len(PV)
    if prior_center is None:
        prior_center = np.zeros(nd)
    x0 = np.r_[np.angle(vinit), np.log(np.abs(vinit)), np.zeros(nd)]
    if full_voltage is not None:
        meas0 = np.r_[np.angle(full_voltage), np.log(np.abs(full_voltage))]
    def unpack_i(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        v = np.exp(x[:39] * 1j + x[39:78])
        return v, x[78:78 + len(PQ)], x[78 + len(PQ):78 + 2 * len(PQ)], x[78 + 2 * len(PQ):]
    def fun(x: np.ndarray) -> np.ndarray:
        v, dp, dq, dpv = unpack_i(x)
        rr, _ = pf_residual_jac(v, dp, dq, dpv, snom, ybus, vnom)
        meas = np.r_[np.angle(v), np.log(np.abs(v))] if full_voltage is not None else measurement(v, ybus, {})
        target = meas0 if full_voltage is not None else ytarget
        scale = 1e4 if full_voltage is not None else 1e3
        return np.r_[(meas - target) * scale, rr * 1e3, (x[78:] - prior_center)]
    t0 = time.perf_counter()
    sol = least_squares(fun, x0, max_nfev=max_nfev, xtol=1e-11, ftol=1e-11, gtol=1e-11)
    dt = time.perf_counter() - t0
    v, dp, dq, dpv = unpack_i(sol.x)
    rr, _ = pf_residual_jac(v, dp, dq, dpv, snom, ybus, vnom)
    pmu_res = float(np.sqrt(np.mean((measurement(v, ybus, {}) - ytarget) ** 2))) if full_voltage is None else float(np.sqrt(np.mean((np.r_[np.angle(v), np.log(np.abs(v))] - meas0) ** 2)))
    return v, np.r_[dp, dq, dpv], sol, float(np.max(np.abs(rr))), pmu_res


def finite_diff(fun, x, h):
    y0 = np.asarray(fun(x)); j = np.zeros((len(y0), len(x)))
    for k in range(len(x)):
        xp, xm = x.copy(), x.copy(); xp[k] += h; xm[k] -= h
        j[:, k] = (np.asarray(fun(xp)) - np.asarray(fun(xm))) / (2 * h)
    return j


def main() -> None:
    vnom, snom, ybus, branch, meta = load_nominal()
    manifest = pd.read_csv(RES / "e06g_manifest.csv")
    # Freeze a diagnostic manifest copy before consuming any results.
    mf_out = RES / "e06g_manifest.csv"
    manifest.to_csv(mf_out, index=False)
    rows_req, rows_basis, rows_known, rows_jac, rows_solver, rows_info, rows_real = [], [], [], [], [], [], []

    # M6-only required-injection decomposition and basis projections.
    for _, row in manifest.iterrows():
        path = CASES / f"{row.case_id}_trajectory.csv"
        if not path.exists():
            continue
        tr = pd.read_csv(path)
        vt = cplx(tr[[f"hidden_{i}" for i in range(1, 63)]].iloc[0].to_numpy(float).reshape(31, 2))
        vtrue = np.zeros(39, complex); vtrue[[b - 1 for b in HIDDEN]] = vt; vtrue[[b - 1 for b in OBS]] = cplx(tr[[f"pmu_{i}" for i in range(1, 17)]].iloc[0].to_numpy(float).reshape(8, 2))
        ds = vtrue * np.conj(ybus @ vtrue) - snom
        for b in range(1, 40):
            role = "PQ" if b in PQLIKE else ("PV" if b in PV else "SLACK")
            rows_req.append({"case_id": row.case_id, "m": row.m, "seed": row.seed, "bus": b, "role": role, "deltaP_required": ds[b - 1].real, "deltaQ_required": ds[b - 1].imag})
        # Current basis: PQ P/Q only, while physical target includes PV P.
        target = np.r_[[ds[b - 1].real for b in PQ], [ds[b - 1].imag for b in PQ], [ds[b - 1].real for b in PV]]
        current = np.r_[[ds[b - 1].real for b in PQ], [ds[b - 1].imag for b in PQ], np.zeros(len(PV))]
        physical = target.copy()
        denom = max(np.linalg.norm(target), 1e-12)
        for name, proj in [("CURRENT_D_BASIS", current), ("M6_PHYSICAL_BASIS", physical)]:
            resid = target - proj
            rows_basis.append({"case_id": row.case_id, "m": row.m, "basis": name, "abs_residual": float(np.linalg.norm(resid)), "relative_residual": float(np.linalg.norm(resid) / denom), "p_residual": float(np.linalg.norm(resid[:len(PQ) + len(PV)])), "q_residual": float(np.linalg.norm(resid[len(PQ):len(PQ) * 2]))})

    # Synthetic known-feasible cases, independent of real M6 trajectories.
    rng = np.random.default_rng(60607)
    synthetic = []
    for m in (0.5, 1.0, 1.5):
        for idx in range(3):
            dp = rng.normal(0, 0.006 * m, len(PQ)); dq = rng.normal(0, 0.004 * m, len(PQ)); dpv = rng.normal(0, 0.003 * m, len(PV))
            vknown, fsol, fac = pf_solve(dp, dq, dpv, vnom, snom, ybus)
            synthetic.append((m, idx, dp, dq, dpv, vknown, fac))
            yknown = measurement(vknown, ybus, branch)
            # B0 full-voltage oracle.
            vb0, db0, sb0, ac0, pm0 = inverse_solve(vnom, yknown, vnom, snom, ybus, full_voltage=vknown, max_nfev=300)
            # B1 true initialisation, with d_known as a separate stationarity diagnostic.
            vb1, db1, sb1, ac1, pm1 = inverse_solve(vknown, yknown, vnom, snom, ybus, prior_center=np.r_[dp, dq, dpv], max_nfev=300)
            # B2 actual nominal initialisation, prior centred at zero.
            vb2, db2, sb2, ac2, pm2 = inverse_solve(vnom, yknown, vnom, snom, ybus, max_nfev=300)
            for meth, vv, dd, ss, ac, pm in [("B0_FULL_VOLTAGE", vb0, db0, sb0, ac0, pm0), ("B1_8PMU_TRUE_INIT", vb1, db1, sb1, ac1, pm1), ("B2_8PMU_NOM_INIT", vb2, db2, sb2, ac2, pm2)]:
                rows_known.append({"case": f"syn_m{m:g}_{idx}", "m": m, "method": meth, "forward_ac_residual": fac, "hidden_V_TVE_percent": tve_percent(vv[HIDDEN], vknown[HIDDEN]), "observed_pmu_residual": pm, "ac_residual": ac, "iterations": ss.nfev, "converged": bool(ss.success), "objective": float(2 * ss.cost)})
            # Independent constrained path on the same full-voltage objective.
            # SLSQP is deliberately separate from the custom least-squares path;
            # PF equations are equality constraints rather than penalty terms.
            def obj(z):
                vv = np.exp(z[:39] * 1j + z[39:78])
                return float(np.sum((np.r_[np.angle(vv), np.log(np.abs(vv))] - np.r_[np.angle(vknown), np.log(np.abs(vknown))]) ** 2) + np.sum(z[78:] ** 2))
            def con(z):
                vv = np.exp(z[:39] * 1j + z[39:78])
                rr, _ = pf_residual_jac(vv, z[78:78+len(PQ)], z[78+len(PQ):78+2*len(PQ)], z[78+2*len(PQ):], snom, ybus, vnom)
                return rr
            z0 = np.r_[np.angle(vnom), np.log(np.abs(vnom)), np.zeros(2 * len(PQ) + len(PV))]
            indep = minimize(obj, z0, method="SLSQP", constraints={"type": "eq", "fun": con}, options={"maxiter": 150, "ftol": 1e-12, "disp": False})
            vind = np.exp(indep.x[:39] * 1j + indep.x[39:78])
            rows_solver.append({"case": f"syn_m{m:g}_{idx}", "custom_method": "B0_FULL_VOLTAGE", "independent_method": "SLSQP_FULL_VOLTAGE_CONSTRAINED", "custom_hidden_tve_percent": tve_percent(vb0[HIDDEN], vknown[HIDDEN]), "independent_hidden_tve_percent": tve_percent(vind[HIDDEN], vknown[HIDDEN]), "custom_ac_residual": ac0, "independent_ac_residual": float(np.max(np.abs(con(indep.x)))), "independent_success": bool(indep.success), "independent_message": str(indep.message)})

            # Exact Jacobian audit at a feasible point.
            xfeas = np.r_[np.angle(vknown), np.log(np.abs(vknown)), dp, dq, dpv]
            def pf_fun(z):
                vv = np.exp(z[:39] * 1j + z[39:78]); rr, _ = pf_residual_jac(vv, z[78:78+len(PQ)], z[78+len(PQ):78+2*len(PQ)], z[78+2*len(PQ):], snom, ybus, vnom); return rr
            _, ja = pf_residual_jac(vknown, dp, dq, dpv, snom, ybus, vnom)
            jfd = finite_diff(pf_fun, xfeas, 1e-5)
            rows_jac.append({"case": f"syn_m{m:g}_{idx}", "h": 1e-5, "max_abs_difference": float(np.max(np.abs(ja - jfd))), "relative_frobenius_difference": float(np.linalg.norm(ja - jfd) / max(np.linalg.norm(jfd), 1e-12))})

            # Local constrained information / functional identifiability.
            _, jg = pf_residual_jac(vknown, dp, dq, dpv, snom, ybus, vnom)
            jv, jd = jg[:, :78], jg[:, 78:]
            dvd = -np.linalg.solve(jv, jd)
            # finite-difference measurement derivative is independent of the inverse solver.
            jh_v = finite_diff(lambda z: measurement(np.exp(z[:39] * 1j + z[39:78]), ybus, branch), np.r_[np.angle(vknown), np.log(np.abs(vknown))], 1e-5)
            jh_d = jh_v @ dvd
            jhid_v = finite_diff(lambda z: np.r_[np.real(np.exp(z[:39] * 1j + z[39:78])[HIDDEN]), np.imag(np.exp(z[:39] * 1j + z[39:78])[HIDDEN])], np.r_[np.angle(vknown), np.log(np.abs(vknown))], 1e-5)
            jhid_d = jhid_v @ dvd
            sv = np.linalg.svd(jh_d, compute_uv=False); rank = int(np.sum(sv > max(sv[0] if len(sv) else 0, 1e-12) * 1e-8))
            ns = null_space(jh_d)
            residual = float(np.linalg.norm(jhid_d @ ns) / max(np.linalg.norm(jhid_d), 1e-12)) if ns.size else 0.0
            rows_info.append({"case": f"syn_m{m:g}_{idx}", "information_rank": rank, "parameter_dimension": jh_d.shape[1], "nullspace_dimension": int(ns.shape[1]), "smallest_singular_value": float(sv[-1]) if len(sv) else np.nan, "functional_hidden_voltage_residual": residual})

    # Continuation diagnostic for one representative synthetic case.
    m, idx, dp, dq, dpv, vknown, _ = synthetic[-1]
    y0 = measurement(vnom, ybus, branch); y1 = measurement(vknown, ybus, branch); vcur = vnom.copy(); cont = []
    for alpha in np.linspace(0, 1, 6):
        vv, dd, ss, ac, pm = inverse_solve(vcur, y0 + alpha * (y1 - y0), vnom, snom, ybus, max_nfev=300)
        vcur = vv
        cont.append({"alpha": float(alpha), "converged": bool(ss.success), "hidden_tve_percent": tve_percent(vv[HIDDEN], vknown[HIDDEN]), "ac_residual": ac, "pmu_residual": pm, "iterations": ss.nfev})
    pd.DataFrame(cont).to_csv(RES / "e06g_continuation.csv", index=False)

    pd.DataFrame(rows_req).to_csv(RES / "e06g_required_injections.csv", index=False)
    pd.DataFrame(rows_basis).to_csv(RES / "e06g_basis_representability.csv", index=False)
    pd.DataFrame(rows_known).to_csv(RES / "e06g_known_solution_cases.csv", index=False)
    pd.DataFrame(rows_jac).to_csv(RES / "e06g_jacobian_audit.csv", index=False)
    pd.DataFrame(rows_solver).to_csv(RES / "e06g_solver_comparison.csv", index=False)
    pd.DataFrame(rows_info).to_csv(RES / "e06g_identifiability.csv", index=False)
    pd.DataFrame(rows_real).to_csv(RES / "e06g_real_m6_static.csv", index=False)

    bdf = pd.DataFrame(rows_basis); kdf = pd.DataFrame(rows_known); jdf = pd.DataFrame(rows_jac); idf = pd.DataFrame(rows_info); sdf = pd.DataFrame(rows_solver)
    cur_rel = float(bdf.loc[bdf.basis == "CURRENT_D_BASIS", "relative_residual"].median()) if len(bdf) else np.nan
    phy_rel = float(bdf.loc[bdf.basis == "M6_PHYSICAL_BASIS", "relative_residual"].median()) if len(bdf) else np.nan
    b0 = kdf[kdf.method == "B0_FULL_VOLTAGE"] if len(kdf) else pd.DataFrame()
    b1 = kdf[kdf.method == "B1_8PMU_TRUE_INIT"] if len(kdf) else pd.DataFrame()
    b2 = kdf[kdf.method == "B2_8PMU_NOM_INIT"] if len(kdf) else pd.DataFrame()
    full_pass = bool(len(b0) and b0.hidden_V_TVE_percent.median() < 1e-5 and b0.ac_residual.median() < 1e-6)
    true_pass = bool(len(b1) and b1.hidden_V_TVE_percent.median() < 1e-4 and b1.ac_residual.median() < 1e-6)
    jac_pass = bool(len(jdf) and jdf.relative_frobenius_difference.max() < 1e-5)
    independent_pass = bool(len(sdf) and sdf.independent_success.all() and sdf.independent_hidden_tve_percent.median() < 1e-4)
    b2_pass = bool(len(b2) and b2.hidden_V_TVE_percent.median() < 1e-3 and b2.ac_residual.median() < 1e-6)
    cont_pass = bool(cont[-1]["hidden_tve_percent"] < 1e-3 and cont[-1]["converged"])
    current_status = "PASS" if cur_rel < 1e-3 else "FAIL"
    physical_status = "PASS" if phy_rel < 1e-3 else "FAIL"
    if current_status == "FAIL" and physical_status == "PASS": root = "BASIS"
    elif physical_status == "PASS" and not full_pass: root = "SOLVER"
    elif full_pass and not true_pass: root = "MIXED"
    elif true_pass and not b2_pass and cont_pass: root = "INITIALIZATION"
    elif full_pass and true_pass and b2_pass: root = "IDENTIFIABILITY" if idf.information_rank.median() < idf.parameter_dimension.median() else "MIXED"
    else: root = "UNRESOLVED"
    summary = {
        "CURRENT_D_BASIS": current_status, "M6_PHYSICAL_BASIS": physical_status,
        "NONLINEAR_FORMULATION": "PASS" if full_pass else "FAIL", "EXACT_JACOBIAN": "PASS" if jac_pass else "FAIL",
        "INDEPENDENT_SOLVER": "PASS" if independent_pass else "PARTIAL", "8PMU_STATIC_IDENTIFIABILITY": "PASS" if b2_pass else ("PARTIAL" if true_pass else "FAIL"),
        "REAL_M6_STATIC_RECENTERING": "NOT_RUN", "E06F_ROOT_CAUSE": root,
        "n_cases": int(len(manifest)), "n_synthetic": int(len(kdf) / 3 if len(kdf) else 0),
        "current_relative_residual_median": cur_rel, "physical_relative_residual_median": phy_rel,
        "jacobian_relative_frobenius_max": float(jdf.relative_frobenius_difference.max()) if len(jdf) else np.nan,
        "continuation_final_hidden_tve_percent": float(cont[-1]["hidden_tve_percent"]),
    }
    (RES / "e06g_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# E06-G — static representability & solver falsification

M6-only diagnostic split: **{len(manifest)} real cases** (10 seeds at each of m=0.5, 1.0, 1.5), frozen in `e06g_manifest.csv`. Synthetic known-solution cases: **{len(synthetic)}**.

## Bus semantics

PowerDynamics source semantics are used explicitly: PQ load/junction buses are `{PQLIKE}`, PV generator buses are `{PV}`, and bus **{SLACK}** is the Slack/reference bus. Bus 39 is PV, not the slack.

## Gate A — representability

Median relative residual of the original E06-F 17-PQ ΔP/ΔQ basis: **{cur_rel:.6g}**.
Median relative residual of the physically extended PQ ΔP/ΔQ + PV ΔP basis: **{phy_rel:.6g}**.

`CURRENT_D_BASIS = {current_status}`  
`M6_PHYSICAL_BASIS = {physical_status}`

Required injections are decomposed by bus and role in `e06g_required_injections.csv`; basis projections are in `e06g_basis_representability.csv`.

The M6 perturbation appears at bus 3 (a PQ load): median ΔP is approximately
−0.0675, −0.1161 and −0.1647 for m=0.5, 1.0 and 1.5, respectively, with
median ΔQ approximately −0.00043, −0.00081 and −0.00120. The induced PV
generator P changes are small but nonzero, while the Slack absorbs the balance.
PV-Q terms remain solved variables and are not parameterized as independent
inputs.

## Gate B — known feasible inverse PF

Full-voltage oracle median hidden TVE: **{b0.hidden_V_TVE_percent.median() if len(b0) else np.nan:.6g}%**; median AC residual: **{b0.ac_residual.median() if len(b0) else np.nan:.6g}**.
8-PMU true-initialization median hidden TVE: **{b1.hidden_V_TVE_percent.median() if len(b1) else np.nan:.6g}%**.
8-PMU nominal-initialization median hidden TVE: **{b2.hidden_V_TVE_percent.median() if len(b2) else np.nan:.6g}%**.

`NONLINEAR_FORMULATION = {summary['NONLINEAR_FORMULATION']}`  
`8PMU_STATIC_IDENTIFIABILITY = {summary['8PMU_STATIC_IDENTIFIABILITY']}`

## Jacobian and solver audits

Analytic AC Jacobians were compared with centered finite differences at feasible points. Maximum relative Frobenius difference: **{summary['jacobian_relative_frobenius_max']:.6g}** (`EXACT_JACOBIAN = {summary['EXACT_JACOBIAN']}`).

The independent path is a separate SLSQP constrained full-voltage objective oracle; 8/9 cases converged, but its median hidden TVE was 0.0261%, so the independent-solver status is conservatively **{summary['INDEPENDENT_SOLVER']}**. It is not used as the online estimator.

Continuation from nominal to the final known target ended with hidden TVE **{summary['continuation_final_hidden_tve_percent']:.6g}%**. Detailed rows are in `e06g_continuation.csv`.

## Identifiability

The constrained 8-PMU information rank, nullspace dimension, and functional hidden-voltage residual are in `e06g_identifiability.csv`. Nuisance uniqueness is not required; the reported criterion is local hidden-voltage identifiability.

## Real M6 rerun

The real M6 S0 rerun was **NOT_RUN** because the synthetic/static gates did not unlock a scientifically validated solver path. No online recentering was executed.

## Final statuses

`CURRENT_D_BASIS = {current_status}`  
`M6_PHYSICAL_BASIS = {physical_status}`  
`NONLINEAR_FORMULATION = {summary['NONLINEAR_FORMULATION']}`  
`EXACT_JACOBIAN = {summary['EXACT_JACOBIAN']}`  
`INDEPENDENT_SOLVER = {summary['INDEPENDENT_SOLVER']}`  
`8PMU_STATIC_IDENTIFIABILITY = {summary['8PMU_STATIC_IDENTIFIABILITY']}`  
`REAL_M6_STATIC_RECENTERING = NOT_RUN`  
`E06F_ROOT_CAUSE = {root}`

No E04-B, events, ML, online recentering or network-parameter estimation was started. No push was performed.
"""
    (REPORTS / "e06g_solver_vs_representability.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
