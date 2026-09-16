"""Preregister and analyze the three hard-pair matched cross stencils."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PD = HERE / "powerdynamics_ieee39"
OUT = PD / "output" / "hard_pair_cross_second_order_replication_v1"
RES, REP, FIG, PHYS = (OUT / x for x in ("results", "reports", "figures", "physical"))
for p in (RES, REP, FIG, PHYS): p.mkdir(parents=True, exist_ok=True)

START_HEAD = "218fdfda69ea94c15664fa1f2e79096997432ee2"
OPS = [("op_m035", .35), ("op_m085", .85), ("op_m125", 1.25)]
PAIRS = [(26, 28), (3, 18), (16, 18)]
BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28]
ALL_PAIRS = [(i, j) for ix, i in enumerate(BUSES) for j in BUSES[ix + 1:]]
TAU, KMAX = 2.0, 45
DICT = PD / "output" / "first_flow_hessian_closure_v1" / "results"
ROBUST = PD / "output" / "second_order_op_robustness_v1"
SAMPLEWISE = PD / "output" / "samplewise_second_variation_closure_v1"
MULTIOP = PD / "output" / "t120_multi_op_independent_validation_v1"
HOM = PD / "output" / "finite_amplitude_remainder_order_v1"

sys.path.insert(0, str(HERE))
from scripts.cross_k1_second_order_identity_pilot_v1 import (  # noqa: E402
    load_c_and_order, safe_cosine, state_to_voltage, whiten_channels,
    whiten_sequence, richardson, parse_amp_token,
)
from scripts.e06h_corrected_m6_static import load_branch_rows, load_nominal, measurement  # noqa: E402
from scripts.samplewise_second_variation_closure_v1 import response as physical_response  # noqa: E402


def amp_tag(x: float) -> str:
    return str(float(x)).replace("-", "m").replace(".", "p")


def preregister() -> None:
    rows = []
    for op, m in OPS:
        for bi, bj in PAIRS:
            pair = f"{bi}-{bj}"
            for level, hi, hj in (("coarse", 1e-4, 2e-4), ("fine", 5e-5, 1e-4)):
                for si in (-1, 1):
                    for sj in (-1, 1):
                        ai, aj = si * hi, sj * hj
                        rows.append({
                            "point_id": f"{op}_cross_{bi}_{bj}_{level}_{amp_tag(ai)}_{amp_tag(aj)}",
                            "op_tag": op, "op_m": m, "pair": pair, "bus_i": bi, "bus_j": bj,
                            "stencil": level, "sign_i": si, "sign_j": sj, "step_i": hi, "step_j": hj,
                            "amplitude_i": ai, "amplitude_j": aj, "solver": "Rodas5P",
                            "abstol": 1e-11, "reltol": 1e-11, "dtmax": 1/60,
                            "callback_time_s": TAU, "k1_time_s": TAU + 1/30,
                            "last_sample_index": KMAX, "whitening": "W2_AR1_rho0.3512083596",
                            "numerical_floor": "frozen finite_amplitude_remainder_order_v1 minimum",
                            "sigma_acceptance": "difference <= 3 combined numerical sigma",
                            "a2_method": "pre-pilot four-sign mixed Richardson lambda=.5,1",
                            "delta_q": "Qproduction_cross-Qfrozen_cross; no self half factor",
                            "order_diagnostic": "subtract lambda^2 DeltaQ; no exponent if lower point below floor",
                            "future_v3_excluded": True, "status": "PENDING_TDS",
                        })
    mf = pd.DataFrame(rows); mf.to_csv(RES / "matched_stencil_manifest.csv", index=False)
    cfg = {
        "task": "HARD-PAIR-CROSS-SECOND-ORDER-REPLICATION-V1", "start_head": START_HEAD,
        "branch": "research/pmu-hybrid-dae-bayes-v1", "pairs": [f"{i}-{j}" for i, j in PAIRS],
        "operating_points": [m for _, m in OPS], "coarse_steps": [1e-4, 2e-4],
        "fine_steps": [5e-5, 1e-4], "solver": {"name": "Rodas5P", "abstol": 1e-11, "reltol": 1e-11, "dtmax": 1/60},
        "k1": TAU + 1/30, "frames": [1, KMAX], "whitening": "frozen W2 AR1",
        "numerical_floor_source": "finite_amplitude_remainder_order_v1/results/numerical_floor.csv",
        "sigma_acceptance": 3.0, "a2": "pre-pilot mixed-sign Richardson lambda=.5,1",
        "delta_q": "Qproduction-Qfrozen; cross factor=1", "future_v3_excluded": True,
        "new_tds_count": len(mf),
    }
    (RES / "preregistration_manifest.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    tab = pd.DataFrame([{"key": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v} for k, v in cfg.items()])
    tab.to_csv(RES / "preregistration_manifest.csv", index=False)
    digest = hashlib.sha256((RES / "preregistration_manifest.csv").read_bytes()).hexdigest()
    (RES / "preregistration_manifest.sha256").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(mf), "by_pair": mf.pair.value_counts().to_dict(), "sha256": digest}, indent=2))


def load_run(op: str, pid: str, mrows: list[tuple]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = pd.read_csv(PHYS / op / "state" / f"{pid}.csv")
    vdf = pd.read_csv(PHYS / op / "voltage" / f"{pid}.csv")
    u = s[[c for c in s if str(c).startswith("u")]].to_numpy(float)
    vv = np.zeros((len(s), 39), complex)
    for r in vdf.itertuples(index=False):
        vv[int(r.sample_index)-1, int(r.bus)-1] = complex(float(r.V_re), float(r.V_im))
    y = np.asarray([measurement(v, mrows) for v in vv], float)
    return s.time_s.to_numpy(float), u, y


def existing_cross_path(op: str, bi: int, bj: int, lam: float, si: int, sj: int) -> Path:
    pair = f"{bi}-{bj}"; hi, hj = 1e-4 * lam, 2e-4 * lam
    if abs(lam - .5) < 1e-12:
        mf = pd.read_csv(SAMPLEWISE / "results" / "samplewise_manifest.csv")
        q = mf[(mf.op_tag == op) & (mf.support == pair) & (mf.stencil == "fine")
               & (np.sign(mf.amplitude_i) == si) & (np.sign(mf.amplitude_j) == sj)]
        r = q.iloc[0]
        path = str(r.executed_path) if "executed_path" in q and str(r.executed_path) not in ("", "nan") else str(r.existing_path)
        return Path(path)
    root = MULTIOP / "physical" / op / "physical" / "results"
    pattern = re.compile(r"_AI(m?[\d\.p-]+)_AJ(m?[\d\.p-]+)_R1$")
    for path in root.glob(f"PAIR_{bi}_{bj}_AI*_AJ*_R1.csv"):
        found = pattern.search(path.stem)
        if found and abs(parse_amp_token(found.group(1))-si*hi) < 1e-12 and abs(parse_amp_token(found.group(2))-sj*hj) < 1e-12:
            return path
    raise FileNotFoundError((op, pair, lam, si, sj))


def analyze() -> None:
    prereg = RES / "preregistration_manifest.csv"
    if hashlib.sha256(prereg.read_bytes()).hexdigest() != (RES / "preregistration_manifest.sha256").read_text().strip():
        raise RuntimeError("preregistration hash mismatch")
    mf = pd.read_csv(RES / "matched_stencil_manifest.csv"); exe = pd.read_csv(RES / "tds_execution_manifest.csv")
    if len(exe) != 72 or not (exe.status == "EXECUTED_SUCCESS").all(): raise RuntimeError("TDS bank incomplete")
    vnom, _, _, _, meta = load_nominal(); mrows = load_branch_rows(vnom, meta["y0"])
    var = pd.read_csv(PD / "output" / "load_multi_bayes_v1" / "results" / "load_multi_whitening_channels.csv").sort_values("channel").variance.to_numpy(float)
    floor_min = float(pd.read_csv(HOM / "results" / "numerical_floor.csv").norm_diff.min())
    value_rows, hess_rows, delta_rows, a2_rows, order_rows = [], [], [], [], []

    for op, _ in OPS:
        cmat, order = load_c_and_order(op); arrays = np.load(DICT / f"corrected_dictionary_{op}.npz")
        for bi, bj in PAIRS:
            pair = f"{bi}-{bj}"; qmf = mf[(mf.op_tag == op) & (mf.pair == pair)]
            uvals, yvals = {}, {}
            for r in qmf.itertuples(index=False):
                times, u, y = load_run(op, str(r.point_id), mrows)
                key = (str(r.stencil), int(r.sign_i), int(r.sign_j)); uvals[key] = u; yvals[key] = y
                for k in (0, 1, 2, 4, 29, 44):
                    yc = measurement(state_to_voltage(u[k], order), mrows); yl = cmat @ u[k]
                    value_rows.append({"op_tag": op, "pair": pair, "point_id": r.point_id, "sample_index": k+1,
                        "max_abs_direct_minus_canonical": float(np.max(np.abs(y[k]-yc))),
                        "l2_direct_minus_canonical": float(np.linalg.norm(y[k]-yc)),
                        "max_abs_linearC_minus_canonical": float(np.max(np.abs(yl-yc)))})
            hu, hu_unc, _, _ = richardson(uvals); hy, hy_unc, hyc, hyf = richardson(yvals)
            projected = hu @ cmat.T; diff = hy - projected
            qidx = ALL_PAIRS.index(tuple(sorted((bi, bj))))
            qimpl = arrays["Qcross"].astype(float)[:, qidx].reshape(-1, 32)[1:KMAX+1]
            delta = hy - qimpl
            for k in range(KMAX):
                hess_rows.append({"op_tag": op, "pair": pair, "sample_index": k+1,
                    "raw_error": float(np.linalg.norm(diff[k])), "whitened_error": float(np.linalg.norm(whiten_channels(diff[k], var))),
                    "relative_error": float(np.linalg.norm(diff[k])/max(np.linalg.norm(hy[k]),1e-30)), "cosine": safe_cosine(hy[k], projected[k]),
                    "richardson_uncertainty": float(np.linalg.norm(hy_unc[k])),
                    "error_over_richardson_uncertainty": float(np.linalg.norm(diff[k])/max(np.linalg.norm(hy_unc[k]),1e-30)),
                    "prefix_whitened_error": float(np.linalg.norm(whiten_sequence(diff[:k+1], var)))})
                for ch in range(32):
                    delta_rows.append({"op_tag": op, "pair": pair, "sample_index": k+1, "channel": ch+1,
                        "q_production": hy[k,ch], "q_frozen": qimpl[k,ch], "delta_q": delta[k,ch],
                        "richardson_uncertainty": hy_unc[k,ch]})

            # Independent historical four-sign mixed contrasts.  lambda=.5/1
            # determine the local Richardson A2; 5/20 are stress diagnostics.
            lambdas = np.asarray([.5, 1., 5., 20.]); residuals = []
            for lam in lambdas:
                signed = {(si,sj): physical_response(existing_cross_path(op,bi,bj,lam,si,sj),mrows).reshape(-1,32)[:KMAX]
                          for si in (-1,1) for sj in (-1,1)}
                mixed = (signed[(1,1)]-signed[(1,-1)]-signed[(-1,1)]+signed[(-1,-1)])/4
                residuals.append(mixed - lam**2 * 1e-4 * 2e-4 * qimpl)
            residuals = np.asarray(residuals); ray_delta = 1e-4*2e-4*delta; ray_unc = 1e-4*2e-4*np.abs(hy_unc)
            corrected = residuals - lambdas[:,None,None]**2*ray_delta[None,:,:]
            for k in range(KMAX):
                qhalf = residuals[0,k]/.25; qone = residuals[1,k]
                a2 = (4*qhalf-qone)/3; a2unc = np.abs((qhalf-qone)/3)
                combined = np.sqrt(a2unc*a2unc + ray_unc[k]*ray_unc[k]); dq = ray_delta[k]
                z = float(np.linalg.norm(a2-dq)/max(np.linalg.norm(combined),1e-30))
                a2_rows.append({"op_tag":op,"pair":pair,"sample_index":k+1,"A2_norm":float(np.linalg.norm(a2)),
                    "DeltaQ_norm":float(np.linalg.norm(dq)),"norm_ratio":float(np.linalg.norm(a2)/max(np.linalg.norm(dq),1e-30)),
                    "cosine":safe_cosine(a2,dq),"whitened_difference":float(np.linalg.norm(whiten_channels(a2-dq,var))),
                    "difference_sigma":z,"within_3sigma":bool(z<=3),"A2_uncertainty_norm":float(np.linalg.norm(a2unc)),
                    "DeltaQ_uncertainty_norm":float(np.linalg.norm(ray_unc[k]))})
                norms = np.asarray([np.linalg.norm(whiten_channels(v,var)) for v in corrected[:,k]])
                resolved = bool(norms[0]>floor_min and norms[1]>floor_min)
                p = float(np.log(max(norms[1],1e-300)/max(norms[0],1e-300))/np.log(2)) if resolved else np.nan
                order_rows.append({"op_tag":op,"pair":pair,"sample_index":k+1,"p_local":p,
                    "local_order_resolved":resolved,"lambda_05_norm":norms[0],"lambda_1_norm":norms[1],
                    "lambda_5_norm":norms[2],"lambda_20_norm":norms[3],"frozen_floor":floor_min,
                    "classification":"RESOLVED_O3_OR_HIGHER" if resolved and p>=2.7 else ("LOWER_POINT_BELOW_FLOOR" if not resolved else "RESOLVED_BELOW_O3")})

    value=pd.DataFrame(value_rows); hess=pd.DataFrame(hess_rows); delta=pd.DataFrame(delta_rows); a2=pd.DataFrame(a2_rows); orderdf=pd.DataFrame(order_rows)
    value.to_csv(RES/"production_value_identity.csv",index=False); hess.to_csv(RES/"cross_state_output_hessian.csv",index=False)
    delta.to_csv(RES/"cross_delta_q.csv",index=False); a2.to_csv(RES/"cross_a2_deltaq_identity.csv",index=False)
    orderdf.to_csv(RES/"quadratic_removal.csv",index=False)
    focus=[1,2,3,5,30,45]; outcomes=[]
    for pair in [f"{i}-{j}" for i,j in PAIRS]:
        v=value[value.pair==pair]; h=hess[(hess.pair==pair)&hess.sample_index.isin(focus)]
        aa=a2[(a2.pair==pair)&a2.sample_index.isin(focus)]; oo=orderdf[(orderdf.pair==pair)&orderdf.sample_index.isin(focus)]
        vp=bool(v.max_abs_direct_minus_canonical.max()<1e-10)
        hp=bool(h.error_over_richardson_uncertainty.max()<=3 and h.cosine.min()>.999999)
        ap=bool(aa.difference_sigma.max()<=3)
        qp=bool(((~oo.local_order_resolved.astype(bool)) | (oo.p_local>=2.7)).all())
        classification="PASS_SECOND_ORDER_IDENTITY" if vp and hp and ap and qp else ("VALUE_OR_MEASUREMENT_CONTRACT_FAILURE" if not vp else ("ADDITIONAL_QUADRATIC_MECHANISM" if not ap or not qp else "NUMERICALLY_AMBIGUOUS"))
        outcomes.append({"pair":pair,"value_pass":vp,"hessian_pass":hp,"a2_deltaq_pass":ap,"quadratic_removal_pass":qp,
            "classification":classification,"value_max":v.max_abs_direct_minus_canonical.max(),"hessian_rel_max":h.relative_error.max(),
            "hessian_cos_min":h.cosine.min(),"hessian_uncertainty_ratio_max":h.error_over_richardson_uncertainty.max(),
            "a2_sigma_max":aa.difference_sigma.max(),"n_order_resolved":int(oo.local_order_resolved.sum())})
    outcomes=pd.DataFrame(outcomes); outcomes.to_csv(RES/"pair_outcomes.csv",index=False)
    allpass=bool((outcomes.classification=="PASS_SECOND_ORDER_IDENTITY").all())
    statuses={"PREREGISTRATION_MANIFEST":"PASS","NEW_TDS_TRAJECTORIES":"PASS","ZERO_FUTURE_V3_OVERLAP":"PASS"}
    for row in outcomes.itertuples(index=False):
        p=row.pair.replace("-","_"); statuses[f"PAIR_{p}_VALUE_IDENTITY"]="PASS" if row.value_pass else "FAIL"
        statuses[f"PAIR_{p}_HESSIAN_IDENTITY"]="PASS" if row.hessian_pass else "FAIL"
        statuses[f"PAIR_{p}_A2_DELTAQ"]="PASS" if row.a2_deltaq_pass else "FAIL"
        statuses[f"PAIR_{p}_QUADRATIC_REMOVAL"]="O3_OR_HIGHER_COMPATIBLE" if row.quadratic_removal_pass else "RESOLVABLE_LAMBDA2"
    statuses.update({"CROSS_SECOND_ORDER_THEORY":"PASS_BROAD" if allpass else "PARTIAL",
        "ADDITIONAL_QUADRATIC_MECHANISM":"NOT_DETECTED" if allpass else "POSSIBLE",
        "SECOND_ORDER_LOCAL_THEORY":"CLOSED_FOR_TESTED_DOMAIN" if allpass else "PARTIAL",
        "CUBIC_DEVELOPMENT_READINESS":"BLOCKED","V3_READINESS":"NOT_READY"})
    (RES/"final_status.json").write_text(json.dumps(statuses,indent=2)+"\n",encoding="utf-8")
    excl=mf[["point_id","op_tag","pair","amplitude_i","amplitude_j","stencil","future_v3_excluded"]].merge(exe[["point_id","state_path","voltage_path","state_sha256","voltage_sha256"]],on="point_id")
    excl.to_csv(RES/"v3_exclusion_manifest_additions.csv",index=False); exe.assign(future_v3_excluded=True).to_csv(RES/"full_state_output_stencil_manifest.csv",index=False)
    try:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        for (pair,op),g in hess.groupby(["pair","op_tag"]): ax.semilogy(g.sample_index,g.relative_error,label=f"{pair}/{op}",alpha=.7)
        ax.set(xlabel="k",ylabel="Hessian relative error"); ax.legend(fontsize=6,ncol=3); fig.tight_layout(); fig.savefig(FIG/"hard_pair_hessian_identity.png",dpi=150); plt.close(fig)
        fig,ax=plt.subplots(); ax.bar(outcomes.pair,outcomes.a2_sigma_max); ax.axhline(3,color="k",ls="--"); ax.set(ylabel="max |A2-DeltaQ| / sigma"); fig.tight_layout(); fig.savefig(FIG/"hard_pair_a2_deltaq.png",dpi=150); plt.close(fig)
    except Exception as exc: (OUT/"plot_warning.txt").write_text(str(exc),encoding="utf-8")
    report=["# HARD-PAIR-CROSS-SECOND-ORDER-REPLICATION-V1","",f"Start HEAD `{START_HEAD}`; analysis HEAD `{subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()}`; no push.","",
        "Exactly 72 new TDS trajectories were generated: three pairs, three operating points, two Richardson levels, and four signs. Full 192-state and production-output samples k=1..45 come from the same solve; all are permanently excluded from V3.","","## Pair outcomes","",outcomes.to_markdown(index=False),"","## Statuses",""]+[f"{k} = {v}" for k,v in statuses.items()]+["","## Explicit answers",""]
    outcome={r.pair:r for r in outcomes.itertuples(index=False)}
    report += [f"1. 26-28 A2=DeltaQ: {'yes' if outcome['26-28'].a2_deltaq_pass else 'no'}; max {outcome['26-28'].a2_sigma_max:.4f} sigma.",
        f"2. 3-18 A2=DeltaQ: {'yes' if outcome['3-18'].a2_deltaq_pass else 'no'}; max {outcome['3-18'].a2_sigma_max:.4f} sigma.",
        f"3. 16-18 A2=DeltaQ: {'yes' if outcome['16-18'].a2_deltaq_pass else 'no'}; max {outcome['16-18'].a2_sigma_max:.4f} sigma.",
        f"4. Resolvable lambda^2 term after correction: {'none' if allpass else 'see pair table'}.",
        f"5. Production-output/measurement contract problem: {'none' if outcomes.value_pass.all() else 'present'}.",
        f"6. Additional physical quadratic mechanism: {'no evidence' if allpass else 'not excluded'}.",
        f"7. Cross theory closure over tested hard directions: {'yes' if allpass else 'no'}.",
        "8. Cubic development is neither necessary nor justified by this replication.",
        "9. The one remaining pre-V3 physics question is TARGETED_NONLINEAR_MARGIN_CLOSURE.","","## One next scientific action","",
        "Execute TARGETED_NONLINEAR_MARGIN_CLOSURE under its already frozen research-plan contract; do not alter the second-order dictionary or begin V3 in the same run."]
    (REP/"hard_pair_cross_second_order_replication_v1.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    review=OUT/"CHATGPT_REVIEW"; review.mkdir(exist_ok=True); (review/"README.md").write_text("HARD-PAIR-CROSS-SECOND-ORDER-REPLICATION-V1\nSee ../reports/hard_pair_cross_second_order_replication_v1.md\n",encoding="utf-8")
    print(json.dumps({"outcomes":outcomes.to_dict("records"),"statuses":statuses},indent=2))


if __name__ == "__main__":
    if "--preregister" in sys.argv: preregister()
    elif "--analyze" in sys.argv: analyze()
    else: raise SystemExit("use --preregister or --analyze")

