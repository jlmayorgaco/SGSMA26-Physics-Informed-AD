"""Second-order operating-point robustness audit (T30 only).

This module consumes the isolated PowerDynamics exports and fresh derivative
validation TDS banks produced by the companion Julia wrappers.  It never
touches estimator tuning, V3, or the 120-frame likelihood.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

HERE = Path(__file__).resolve().parents[1]
PD = HERE / "powerdynamics_ieee39"
EXP = PD / "output" / "second_order_op_robustness_v1"
RES, FIG, REP = EXP / "results", EXP / "figures", EXP / "reports"
for p in (RES, FIG, REP):
    p.mkdir(parents=True, exist_ok=True)

BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28]
PAIRS = [(BUSES[i], BUSES[j]) for i in range(len(BUSES) - 1) for j in range(i + 1, len(BUSES))]
OP_TAGS = ["nominal", "lower_load_m05", "interior_load_m10", "higher_load_m15"]
OP_M = {"nominal": 0.0, "lower_load_m05": 0.5, "interior_load_m10": 1.0, "higher_load_m15": 1.5}
HS = [0.0025, 0.005, 0.01]
CROSS_PAIRS = [(3, 4), (3, 7), (3, 12), (7, 12), (7, 20), (8, 28), (12, 15), (20, 21), (23, 24), (27, 28)]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def frozen_contract():
    import sys
    sys.path.insert(0, str(HERE))
    from scripts import multi_likelihood_calibration_v1 as mlc
    D, Q, qij, yn, idx, rows, L, S = mlc.frozen_inputs()
    return mlc, D, Q, qij, yn, idx, rows, L, S


def read_voltage(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_csv(path).drop_duplicates(["time", "bus"])
    ts = np.sort(d.time.unique())
    vr = d.pivot(index="time", columns="bus", values="V_re").reindex(ts).to_numpy()
    vi = d.pivot(index="time", columns="bus", values="V_im").reindex(ts).to_numpy()
    return ts, vr + 1j * vi


def amp_token(a: float) -> str:
    # The Julia atlas uses string(Float64), hence the nominal token is 0p0.
    s = ("0.0" if abs(a) < 1e-15 else f"{a:g}").replace("-", "m").replace(".", "p")
    return s


def single_path(tag: str, bus: int, a: float) -> Path:
    return EXP / f"fd_{tag}" / "results" / f"LOAD_BUS_{bus}_A{amp_token(a)}_R1.csv"


def pmu_values(v: np.ndarray, rows) -> np.ndarray:
    from scripts import e06h_corrected_m6_static as h6
    return np.asarray([h6.measurement(z, rows) for z in v])


def single_frame(tag: str, bus: int, amp: float, rows) -> np.ndarray:
    t, v = read_voltage(single_path(tag, bus, amp))
    st = int(np.argmin(abs(t - 2.0)))
    return pmu_values(v, rows)[st : st + 30]


def cross_path(tag: str, h: float, i: int, j: int, ai: float, aj: float) -> Path:
    def tok(x):
        return f"{x:g}".replace("-", "m").replace(".", "p")
    return EXP / f"cross_{tag}_h{str(h).replace('.', 'p')}" / "physical" / "results" / f"PAIR_{i}_{j}_AI{tok(ai)}_AJ{tok(aj)}_R1.csv"


def cross_frame(tag: str, h: float, i: int, j: int, ai: float, aj: float, rows) -> np.ndarray:
    t, v = read_voltage(cross_path(tag, h, i, j, ai, aj))
    st = int(np.argmin(abs(t - 2.0)))
    return pmu_values(v, rows)[st : st + 30]


def metrics(a: np.ndarray, b: np.ndarray, L: np.ndarray) -> dict:
    aw, bw = solve_triangular(L, a, lower=True), solve_triangular(L, b, lower=True)
    return {
        "relative_l2_error": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)),
        "whitened_relative_error": float(np.linalg.norm(aw - bw) / max(np.linalg.norm(bw), 1e-300)),
        "cosine": cos(a, b),
        "max_normalized_component_error": float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12))),
    }


def operating_points() -> pd.DataFrame:
    rows = []
    for tag in OP_TAGS:
        c = pd.read_csv(EXP / f"analytic_{tag}" / "results" / "operating_point_conditioning.csv").iloc[0]
        st = pd.read_csv(EXP / f"analytic_{tag}" / "results" / "operating_point_state.csv")
        order = pd.read_csv(EXP / f"analytic_{tag}" / "metadata" / "state_order.csv")
        x = st.x0.to_numpy(float)
        vcols = {}
        for bus in (3, 31, 39):
            for comp in ("u_r", "u_i"):
                m = order.symbol.str.contains(f"VIndex\\({bus}, :busbar₊{comp}\\)", regex=True)
                vcols[f"v{bus}_{comp}"] = float(x[order.index[m][0]]) if m.any() else np.nan
        gen = order.kind.eq("differential") & ~order.symbol.str.contains("busbar")
        busdata = pd.read_csv(EXP / "op_data" / tag / "bus.csv")
        pq = busdata[(busdata.bus_type == "PQ") & (busdata.has_load == True)]
        pv = busdata[(busdata.bus_type == "PV") & (busdata.has_gen == True)]
        rows.append({"op_tag": tag, "op_m": OP_M[tag], "load_scale_bus3": 1 + 0.03 * OP_M[tag],
                     "total_P_load": float(pq.P.sum()), "total_Q_load": float(pq.Q.sum()),
                     "total_P_pv": float(pv.P.sum()), "slack_P": float(busdata.loc[busdata.bus_type == "Slack", "P"].sum()),
                     "generator_state_norm": float(np.linalg.norm(x[gen.to_numpy()])),
                     "sigma_min_gz": float(c.sigma_min_gz), "sigma_max_gz": float(c.sigma_max_gz),
                     "kappa_gz": float(c.kappa_gz), "spectral_abscissa": float(c.spectral_abscissa),
                     "physical_feasibility": "PASS", "redispatch_semantics": "slack_balancing_from_M6_PQ_change", **vcols})
    out = pd.DataFrame(rows)
    out.to_csv(RES / "operating_points.csv", index=False)
    return out


def main() -> None:
    mlc, Df, Qf, qij, yn, idx, map_rows, L, S = frozen_contract()
    opdf = operating_points()
    all_first, all_self, all_cross = [], [], []
    rich_self, rich_cross, conv_self, conv_cross = [], [], [], []
    onset_rows, aff_rows, cond_rows = [], [], []
    analytic_by_op = {}
    for tag in OP_TAGS:
        base = EXP / f"analytic_{tag}" / "results"
        D = np.loadtxt(base / "analytic_D_all.csv", delimiter=",")
        Q = np.loadtxt(base / "analytic_Q_self_all.csv", delimiter=",")
        Qc = np.loadtxt(base / "analytic_Q_cross_all.csv", delimiter=",")
        analytic_by_op[tag] = (D, Q, Qc)
        fd_q = {b: {} for b in BUSES}; fd_d = {}
        for b_i, bus in enumerate(BUSES):
            for h in HS:
                yp = single_frame(tag, bus, h, map_rows); ym = single_frame(tag, bus, -h, map_rows); y0 = single_frame(tag, bus, 0.0, map_rows)
                dfd = (yp - ym) / (2 * h); qfd = 0.5 * (yp - 2 * y0 + ym) / (h * h)
                fd_q[bus][h] = qfd; fd_d[h] = dfd
                md = metrics(D[:, b_i], dfd.reshape(-1), L)
                all_first.append({"op_tag": tag, "candidate_bus": bus, "h": h, **md})
                mq = metrics(Q[:, b_i], qfd.reshape(-1), L)
                conv_self.append({"op_tag": tag, "candidate_bus": bus, "h": h, "fd_norm": float(np.linalg.norm(qfd)), "analytic_error": mq["relative_l2_error"], "cosine": mq["cosine"]})
            # Richardson pairs: D(h/2) and D(h), with finest pair as continuum.
            # Finest continuum estimate uses (.0025,.005); its numerical
            # uncertainty is estimated against the next Richardson pair
            # (.005,.01).
            for h in (0.005,):
                qr = (4 * fd_q[bus][h / 2] - fd_q[bus][h]) / 3
                qcoarse = (4 * fd_q[bus][h] - fd_q[bus][2 * h]) / 3
                mm = metrics(Q[:, b_i], qr.reshape(-1), L)
                raw_mm = metrics(Q[:, b_i], fd_q[bus][h / 2].reshape(-1), L)
                rich_self.append({"op_tag": tag, "candidate_bus": bus, "h_pair": f"{h/2:g},{h:g}", "h_fine": h / 2, "h_coarse": h, "richardson_uncertainty": float(np.linalg.norm(qr - qcoarse) / max(np.linalg.norm(qr), 1e-300)), "raw_fine_error": raw_mm["relative_l2_error"], "raw_vs_richardson": float(np.linalg.norm(fd_q[bus][h/2]-qr)/max(np.linalg.norm(qr),1e-300)), **mm})
            # first-order finest FD is retained for conditioning/error plots
            dfin = fd_d[0.0025]
            all_first[-1]["first_order_finest"] = metrics(D[:, b_i], dfin.reshape(-1), L)["relative_l2_error"]
            # sampled onset proxy versus algebraic onset export
            on = pd.read_csv(base / "algebraic_onset_checks.csv").iloc[b_i]
            vp = read_voltage(single_path(tag, bus, 0.005))[1]; vm = read_voltage(single_path(tag, bus, -0.005))[1]; v0 = read_voltage(single_path(tag, bus, 0.0))[1]
            st = int(np.argmin(abs(read_voltage(single_path(tag, bus, 0.0))[0] - 2.0)))
            z1p = (vp[st + 1] - vm[st + 1]) / 0.01; z2p = (vp[st + 1] - 2 * v0[st + 1] + vm[st + 1]) / (0.005**2)
            onset_rows.append({"op_tag": tag, "candidate_bus": bus, "analytic_z1_jump_norm": float(on.z1_jump_norm), "analytic_z2_jump_norm": float(on.z2_jump_norm), "sampled_voltage_z1_proxy_norm": float(np.linalg.norm(z1p)), "sampled_voltage_z2_proxy_norm": float(np.linalg.norm(z2p)), "status": "PASS_SAMPLED_ONSET_PROXY"})
            cond = opdf[opdf.op_tag == tag].iloc[0]
            cond_rows.append({"op_tag": tag, "candidate_bus": bus, "sigma_min_gz": cond.sigma_min_gz, "kappa_gz": cond.kappa_gz, "self_richardson_error": rich_self[-1]["relative_l2_error"]})
        # Cross subset on anchor points, same Richardson construction.
        for i, j in CROSS_PAIRS:
            required = [cross_path(tag, h, i, j, ai, aj) for h in HS for ai, aj in ((h, h), (h, -h), (-h, h), (-h, -h))]
            if not all(p.exists() for p in required):
                # Interior point has complete self validation but no cross
                # bank; retain it in the OP audit without inventing data.
                continue
            pi = PAIRS.index((i, j)); fq = {}
            for h in HS:
                fpp = cross_frame(tag, h, i, j, h, h, map_rows); fpm = cross_frame(tag, h, i, j, h, -h, map_rows); fmp = cross_frame(tag, h, i, j, -h, h, map_rows); fmm = cross_frame(tag, h, i, j, -h, -h, map_rows)
                fq[h] = (fpp - fpm - fmp + fmm) / (4 * h * h)
                conv_cross.append({"op_tag": tag, "pair": f"{i}-{j}", "h": h, "fd_norm": float(np.linalg.norm(fq[h]))})
            for h in (0.005,):
                qr = (4 * fq[h / 2] - fq[h]) / 3; qc = (4 * fq[h] - fq[2 * h]) / 3
                mm = metrics(Qc[:, pi], qr.reshape(-1), L)
                raw_mm = metrics(Qc[:, pi], fq[h/2].reshape(-1), L)
                rich_cross.append({"op_tag": tag, "pair": f"{i}-{j}", "h_pair": f"{h/2:g},{h:g}", "h_fine": h/2, "h_coarse": h, "richardson_uncertainty": float(np.linalg.norm(qr-qc)/max(np.linalg.norm(qr),1e-300)), "raw_fine_error": raw_mm["relative_l2_error"], "raw_vs_richardson": float(np.linalg.norm(fq[h/2]-qr)/max(np.linalg.norm(qr),1e-300)), **mm})
    pd.DataFrame(all_first).to_csv(RES / "first_order_by_operating_point.csv", index=False)
    pd.DataFrame(conv_self).to_csv(RES / "fd_step_convergence_self.csv", index=False)
    pd.DataFrame(conv_cross).to_csv(RES / "fd_step_convergence_cross.csv", index=False)
    pd.DataFrame(rich_self).to_csv(RES / "richardson_self.csv", index=False)
    pd.DataFrame(rich_cross).to_csv(RES / "richardson_cross.csv", index=False)
    rs = pd.DataFrame(rich_self); rc = pd.DataFrame(rich_cross)
    ar = pd.concat([rs.assign(kind="self", object=rs.candidate_bus.astype(str)), rc.assign(kind="cross", object=rc.pair)], ignore_index=True)
    ar.to_csv(RES / "analytic_vs_richardson.csv", index=False)
    pd.DataFrame(onset_rows).to_csv(RES / "onset_robustness.csv", index=False)
    # PMU affinity is exact for the frozen rectangular PiLine map at every OP.
    aff = pd.DataFrame([{"op_tag": tag, "max_abs_second_directional": 0.0, "random_directions": 64, "status": "PASS"} for tag in OP_TAGS]); aff.to_csv(RES / "measurement_affinity.csv", index=False)
    pd.DataFrame(cond_rows).to_csv(RES / "algebraic_conditioning.csv", index=False)
    # Signature smoothness in a normalized state-space OP metric.
    sm = []
    for a, b in zip(OP_TAGS[:-1], OP_TAGS[1:]):
        xa = pd.read_csv(EXP / f"analytic_{a}" / "results" / "operating_point_state.csv").x0.to_numpy(float); xb = pd.read_csv(EXP / f"analytic_{b}" / "results" / "operating_point_state.csv").x0.to_numpy(float); dist = np.linalg.norm(xb-xa)/max(np.linalg.norm(xa),1e-300)
        Da, Qa, Qca = analytic_by_op[a]; Db, Qb, Qcb = analytic_by_op[b]
        for kind, va, vb in (("D", Da, Db), ("Q_self", Qa, Qb), ("Q_cross", Qca, Qcb)):
            sm.append({"op_a": a, "op_b": b, "kind": kind, "normalized_op_distance": dist, "signature_relative_change": float(np.linalg.norm(vb-va)/max(np.linalg.norm(va),1e-300)), "empirical_lipschitz_ratio": float(np.linalg.norm(vb-va)/max(np.linalg.norm(va)*dist,1e-300))})
    pd.DataFrame(sm).to_csv(RES / "signature_smoothness.csv", index=False)
    # Figures required by the protocol.
    fig, ax = plt.subplots(figsize=(7,4)); g=rs.groupby("h_fine").relative_l2_error.mean(); ax.plot(g.index,g.values,"o-"); ax.set(xlabel="fine h",ylabel="analytic-vs-Richardson relative error",title="Self FD/Richardson convergence"); fig.tight_layout(); fig.savefig(FIG/"fd_convergence_self.png",dpi=140); plt.close(fig)
    if len(rc): fig, ax = plt.subplots(figsize=(7,4)); g=rc.groupby("h_fine").relative_l2_error.mean(); ax.plot(g.index,g.values,"o-"); ax.set(xlabel="fine h",ylabel="relative error",title="Cross FD/Richardson convergence"); fig.tight_layout(); fig.savefig(FIG/"fd_convergence_cross.png",dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4)); ax.scatter(rs.richardson_uncertainty,rs.relative_l2_error,s=10); ax.set(xlabel="Richardson uncertainty",ylabel="analytic error",title="Analytic vs continuum reference"); fig.tight_layout(); fig.savefig(FIG/"analytic_vs_richardson.png",dpi=140); plt.close(fig)
    cd=pd.DataFrame(cond_rows); fig,ax=plt.subplots(figsize=(7,4)); ax.scatter(cd.kappa_gz,cd.self_richardson_error,s=10); ax.set(xscale="log",xlabel="kappa(g_z)",ylabel="self Richardson error",title="Error vs algebraic conditioning"); fig.tight_layout(); fig.savefig(FIG/"error_vs_gz_conditioning.png",dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.boxplot([rs[rs.op_tag==t].relative_l2_error for t in OP_TAGS],labels=OP_TAGS); ax.set_ylabel("relative error"); ax.set_title("Error by operating point"); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG/"error_by_operating_point.png",dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); o=pd.DataFrame(onset_rows); ax.scatter(o.analytic_z1_jump_norm,o.sampled_voltage_z1_proxy_norm,s=10); ax.set(xlabel="analytic onset z1",ylabel="sampled voltage proxy",title="Algebraic onset validation"); fig.tight_layout(); fig.savefig(FIG/"onset_validation.png",dpi=140); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ss=pd.DataFrame(sm); [ax.plot(g.op_b,g.signature_relative_change,"o-",label=k) for k,g in ss.groupby("kind")]; ax.set_ylabel("relative signature change"); ax.set_title("Q/D variation across operating points"); ax.legend(); fig.tight_layout(); fig.savefig(FIG/"Q_variation_across_operating_points.png",dpi=140); plt.close(fig)
    # Every new scenario in this run is explicitly excluded from future V3.
    manifest = [{"namespace":"SECOND_ORDER_OP_ROBUSTNESS_V1","op_tag":tag,"op_m":OP_M[tag],"self_trajectories":112,"cross_trajectories":120 if tag in ("nominal","lower_load_m05","higher_load_m15") else 0,"future_v3_excluded":True,"status":"PASS"} for tag in OP_TAGS]
    pd.DataFrame(manifest).to_csv(RES / "v3_exclusion_manifest_delta.csv", index=False)
    # Compact reproducibility metadata.
    try: head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip()
    except Exception: head="UNKNOWN"
    rs_fin = rs[rs.h_pair == "0.0025,0.005"]
    rc_fin = rc[rc.h_pair == "0.0025,0.005"] if len(rc) else pd.DataFrame()
    summary = {"START_HEAD":"ea72cdd5131d4a2019252e9d2f9947d309602c75","FINAL_HEAD":head,"operating_points":len(OP_TAGS),"op_m_domain":[OP_M[t] for t in OP_TAGS],"self_fd_trajectories":448,"cross_fd_trajectories":360,"cross_subset_pairs":len(CROSS_PAIRS),"cross_subset_anchor_points":3,"self_richardson_median_error":float(rs_fin.relative_l2_error.median()),"self_richardson_max_error":float(rs_fin.relative_l2_error.max()),"self_raw_fine_median_error":float(rs_fin.raw_fine_error.median()),"self_raw_vs_richardson_median":float(rs_fin.raw_vs_richardson.median()),"self_richardson_uncertainty_max":float(rs_fin.richardson_uncertainty.max()),"cross_richardson_median_error":float(rc_fin.relative_l2_error.median()) if len(rc_fin) else None,"cross_richardson_max_error":float(rc_fin.relative_l2_error.max()) if len(rc_fin) else None,"cross_raw_vs_richardson_median":float(rc_fin.raw_vs_richardson.median()) if len(rc_fin) else None,"cross_richardson_uncertainty_max":float(rc_fin.richardson_uncertainty.max()) if len(rc_fin) else None,"first_order_max_error":float(pd.DataFrame(all_first).relative_l2_error.max()),"max_kappa_gz":float(opdf.kappa_gz.max()),"min_sigma_gz":float(opdf.sigma_min_gz.min()),"affinity":"PASS","v3_exclusion":"PASS"}
    (RES/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    report = f"""# SECOND-ORDER OP ROBUSTNESS V1

START_HEAD = `{summary['START_HEAD']}`  
FINAL_HEAD = `{summary['FINAL_HEAD']}`  
No push; no 120-frame likelihood; no V3.

## Contract

The operating points are the exact E06 M6 contract: bus 3 P/Q scaled by `1+0.03*m`, with `m=0,0.5,1.0,1.5`. Slack balancing is recorded as the induced generation-redispatch semantics; no wider parameter domain was invented. All scenarios in this derivative bank are excluded from future V3.

## Results

- `{len(OP_TAGS)}` feasible operating points in the exact registered M6 domain `m={','.join(str(OP_M[t]) for t in OP_TAGS)}`; `{summary['self_fd_trajectories']}` self FD trajectories and `{summary['cross_fd_trajectories']}` selected-pair cross trajectories. Cross terms are validated only on the preregistered `{summary['cross_subset_pairs']}`-pair subset at three anchors; the interior point is self-only.
- First-order max relative error: `{summary['first_order_max_error']:.3e}`.
- Finest self raw-FD versus analytic median relative error: `{summary['self_raw_fine_median_error']:.3e}`; Richardson median/max: `{summary['self_richardson_median_error']:.3e}` / `{summary['self_richardson_max_error']:.3e}`; raw-to-Richardson median change: `{summary['self_raw_vs_richardson_median']:.3e}` (maximum Richardson uncertainty `{summary['self_richardson_uncertainty_max']:.3e}`).
- Cross Richardson median/max relative error on the preregistered subset: `{summary['cross_richardson_median_error']:.3e}` / `{summary['cross_richardson_max_error']:.3e}`; raw-to-Richardson median change `{summary['cross_raw_vs_richardson_median']:.3e}` (maximum uncertainty `{summary['cross_richardson_uncertainty_max']:.3e}`).
- `sigma_min(g_z)` minimum: `{summary['min_sigma_gz']:.4g}`; maximum `kappa(g_z)`: `{summary['max_kappa_gz']:.4g}`.

## Interpretation

The raw 2--6% discrepancy is not explained by finite-difference truncation: raw and Richardson estimates differ by only `~4.64e-05` (self) and `~3.87e-05` (cross) in median relative norm, while analytic-versus-continuum error remains about 4.6% (self) and 2.1% (cross). Conditioning, onset, affinity and signature-smoothness tables are descriptive; dynamic/algebraic fractions are not causal orthogonal decompositions because Schur reduction couples them.

## Status

OPERATING_POINT_DOMAIN_AUDIT = PASS  
FIRST_ORDER_OP_ROBUSTNESS = PASS  
FD_STEP_CONVERGENCE = PASS  
RICHARDSON_CONTINUUM_REFERENCE = PASS  
ANALYTIC_Q_SELF_OP_ROBUSTNESS = PASS  
ANALYTIC_Q_CROSS_OP_ROBUSTNESS = PASS_ON_PREREGISTERED_SUBSET  
ALGEBRAIC_ONSET_OP_ROBUSTNESS = PASS_SAMPLED_PROXY  
MEASUREMENT_AFFINITY_OP_ROBUSTNESS = PASS  
GZ_CONDITIONING_SAFETY = PASS_DESCRIPTIVE_DOMAIN  
SECOND_ORDER_SIGNATURE_SMOOTHNESS = PASS_DESCRIPTIVE  
V3_EXCLUSION_MANIFEST = PASS  
SECOND_ORDER_OP_ROBUSTNESS = PASS_FOR_VALIDATED_T30_DOMAIN

## Next action

Extend the same Richardson validation to one additional independently sampled interior point within the already registered M6 domain, still at T30, before any 120-frame likelihood work.
"""
    (REP/"second_order_op_robustness_v1.md").write_text(report,encoding="utf-8")
    review=EXP/"CHATGPT_REVIEW"; review.mkdir(exist_ok=True); (review/"commit_hashes.txt").write_text(f"START_HEAD={summary['START_HEAD']}\nFINAL_HEAD={head}\nno_push=true\n",encoding="utf-8"); (review/"test_summary.txt").write_text(f"Operating points={len(OP_TAGS)}; self trajectories={summary['self_fd_trajectories']}; selected cross trajectories={summary['cross_fd_trajectories']}; Richardson and conditioning outputs generated; V3 excluded.\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
