"""LOAD-TANGENT-V2 physical gate.

This stage freezes the central finite-difference load operators and audits the
native PowerDynamics descriptor inventory.  A trajectory tangent is promoted
only when native descriptor matrices, load directions, and PMU Jacobians are
all exported; otherwise every tangent row is explicitly marked NOT_RUN.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(HERE))
from scripts import e06h_corrected_m6_static as h6

ROOT = HERE / "powerdynamics_ieee39"
V1 = ROOT / "output" / "load_response_atlas_v1"
OUT = ROOT / "output" / "load_tangent_v2"
RES = OUT / "results"
MAN = OUT / "manifests"
REP = OUT / "reports"
for p in (RES, MAN, REP):
    p.mkdir(parents=True, exist_ok=True)

OBS = [2, 5, 6, 10, 19, 22, 29, 39]
EPS = 0.005


def _case(bus: int, amp: float) -> Path:
    tag = str(abs(amp)).replace(".", "p")
    if amp < 0:
        tag = "m" + tag
    return V1 / "results" / f"LOAD_BUS_{bus}_A{tag}_R1.csv"


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_csv(path).drop_duplicates(["time", "bus"])
    t = np.sort(d.time.unique())
    re = d.pivot(index="time", columns="bus", values="V_re").reindex(t).to_numpy()
    im = d.pivot(index="time", columns="bus", values="V_im").reindex(t).to_numpy()
    return t, re + 1j * im


def _rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main() -> dict:
    mf = pd.read_csv(V1 / "simulation_manifest_native.csv")
    reg = pd.read_csv(V1 / "candidate_registry_native.csv")
    buses = sorted(reg.bus.astype(int).tolist())

    # Existing rep=2/3 requests are deterministic because the native harness
    # copies the same IEEE39 data and has no seed/OP perturbation.  Do not run
    # them as if they were independent information.
    reject = []
    for b in buses:
        for amp in (-0.005, 0.005, 0.1):
            reject.append({"candidate_bus": b, "amplitude": amp,
                           "realization": 2,
                           "status": "REJECTED_DETERMINISTIC_DUPLICATE",
                           "reason": "the 48 remaining campaign slots have no preregistered seed/operating-point variation"})
    v2mf = mf.copy()
    v2mf["status"] = v2mf.status.replace({"EXECUTED_FAIL": "EXECUTED_FAIL_PHYSICS"})
    v2mf = pd.concat([v2mf, pd.DataFrame(reject)], ignore_index=True, sort=False)
    v2mf.to_csv(RES / "load_atlas_manifest_v2.csv", index=False)

    # Central and one-sided physical operators for all 16 candidates.
    central, plus, minus, times = [], [], [], None
    consistency, amp_rows = [], []
    nominal_v = {}
    vnom, _, _, _, hmeta = h6.load_nominal()
    _, y0rows = h6.build_pd_ybus()
    pmu_rows = h6.load_branch_rows(vnom, hmeta["y0"])
    for bus in buses:
        t0, v0 = _load(_case(bus, 0.0)); tp, vp = _load(_case(bus, EPS)); tm, vm = _load(_case(bus, -EPS))
        if not (np.allclose(t0, tp) and np.allclose(t0, tm)):
            raise RuntimeError(f"time grid mismatch for bus {bus}")
        times = t0[np.arange(np.argmin(abs(t0 - 2.0)), min(len(t0), np.argmin(abs(t0 - 2.0)) + 30))]
        idx = np.arange(np.argmin(abs(t0 - 2.0)), min(len(t0), np.argmin(abs(t0 - 2.0)) + 30))
        y0 = np.asarray([h6.measurement(v, pmu_rows) for v in v0])
        yp = np.asarray([h6.measurement(v, pmu_rows) for v in vp])
        ym = np.asarray([h6.measurement(v, pmu_rows) for v in vm])
        g = (yp[idx] - ym[idx]) / (2 * EPS)
        gp = (yp[idx] - y0[idx]) / EPS
        gm = (ym[idx] - y0[idx]) / (-EPS)
        central.append(g); plus.append(gp); minus.append(gm); nominal_v[bus] = (t0, y0, yp, ym)
        consistency.append({"candidate_bus": bus, "epsilon": EPS,
                            "plus_minus_norm": float(np.linalg.norm(gp - gm)),
                            "plus_minus_cosine": float(np.dot(gp.ravel(), gm.ravel()) / max(np.linalg.norm(gp) * np.linalg.norm(gm), 1e-12)),
                            "central_vs_plus_relerr": _rel(g, gp),
                            "central_vs_minus_relerr": _rel(g, gm),
                            "central_norm": float(np.linalg.norm(g))})
        for amp, yy in [(-EPS, ym), (EPS, yp), (0.1, np.asarray([h6.measurement(v, pmu_rows) for v in _load(_case(bus, 0.1))[1]]) )]:
            response = yy[idx] - y0[idx]
            ah = float(np.dot(g.ravel(), response.ravel()) / max(np.dot(g.ravel(), g.ravel()), 1e-12) / amp)
            amp_rows.append({"candidate_bus": bus, "amplitude": amp,
                             "estimated_amplitude": ah * amp,
                             "bias": ah * amp - amp,
                             "relative_bias": abs(ah - 1.0)})
    central = np.asarray(central); plus = np.asarray(plus); minus = np.asarray(minus)
    np.savez_compressed(RES / "load_fd_central_operator.npz", candidate_buses=np.asarray(buses),
                        times=np.asarray(times), epsilon=EPS, central=central,
                        dplus=plus, dminus=minus)
    pd.DataFrame(consistency).to_csv(RES / "load_fd_central_consistency.csv", index=False)

    # Native descriptor metadata already exported by PowerDynamics.  Preserve
    # its exact 192-variable ordering and record the frozen hash.
    inv = ROOT / "output" / "results" / "pd_descriptor_inventory.csv"
    if inv.exists():
        meta = pd.read_csv(inv)
        meta.to_csv(RES / "load_descriptor_metadata.csv", index=False)
        descriptor = "PASS" if len(meta) == 192 and (meta.kind == "differential").sum() == 114 and (meta.kind == "algebraic_zero_mass").sum() == 78 else "FAIL"
        descriptor_hash = hashlib.sha256((RES / "load_descriptor_metadata.csv").read_bytes()).hexdigest()
    else:
        meta = pd.DataFrame(); descriptor = "FAIL"; descriptor_hash = ""

    # No native E/A, load B_g, or PMU C export exists in the current harness.
    # Materialize per-candidate blocked rows rather than fabricating a tangent.
    tan = pd.DataFrame([{"candidate_bus": b, "status": "NOT_RUN",
                         "reason": "native descriptor E/A and load B_g exports unavailable",
                         "relative_frobenius_error": np.nan,
                         "cosine_similarity": np.nan} for b in buses])
    tan.to_csv(RES / "load_tangent_metrics.csv", index=False)
    pd.DataFrame([{"status": "FAIL", "reason": "native PowerDynamics load parameter residual derivative B_g not exported"}]).to_csv(RES / "load_parameter_derivatives.csv", index=False)
    pd.DataFrame([{"status": "FAIL", "reason": "full PMU state/algebraic Jacobian C not exported by native harness"}]).to_csv(RES / "pmu_jacobian_audit.csv", index=False)
    pd.DataFrame(columns=["bus_i", "bus_j", "cosine_coherence", "principal_angle_deg", "sigma_min_concat"]).to_csv(RES / "load_pair_geometry.csv", index=False)
    pd.DataFrame([{"bus_i": 7, "bus_j": 12, "status": "NOT_RUN", "reason": "tangent gate not passed"}]).to_csv(RES / "load_bus7_bus12_geometry.csv", index=False)
    pd.DataFrame(amp_rows).to_csv(RES / "load_amplitude_validation.csv", index=False)

    max_cons = float(max(r["central_vs_plus_relerr"] for r in consistency))
    summary = {
        "LOAD_ATLAS_COMPLETE": "PARTIAL",
        "DESCRIPTOR_EXPORT": descriptor,
        "LOAD_PARAMETER_DERIVATIVES": "FAIL",
        "PMU_JACOBIAN": "FAIL",
        "TRAJECTORY_TANGENT_VALIDATION": "FAIL",
        "FINITE_AMPLITUDE_DIRECTION": "NOT_RUN",
        "AMPLITUDE_INVERSION": "PARTIAL",
        "candidate_count": len(buses), "executed_native_cases": int((mf.status == "EXECUTED_SUCCESS").sum()),
        "rejected_duplicate_cases": len(reject), "selected_epsilon": EPS,
        "max_central_vs_one_sided_relerr": max_cons,
        "descriptor_hash": descriptor_hash,
        "source_bayes": "NOT_RUN", "evi": "NOT_EVALUATED", "gsp": "NOT_EVALUATED",
    }
    pd.DataFrame([summary]).to_csv(RES / "load_tangent_summary.csv", index=False)
    (REP / "load_tangent_v2.md").write_text(
        "# LOAD-TANGENT-V2\n\n"
        f"The frozen registry has {len(buses)} hidden ZIP candidates. The native atlas contains {summary['executed_native_cases']} successful first-realization cases; {len(reject)} additional requests were rejected as deterministic duplicates because the harness has no preregistered seed or operating-point variation.\n\n"
        f"Central FD operators use epsilon={EPS}; maximum central-versus-one-sided relative difference is {max_cons:.6g}. The native descriptor inventory is {descriptor} (192 variables, 114 differential, 78 algebraic; hash `{descriptor_hash}`).\n\n"
        "Full native E/A matrices, physical load residual directions B_g, and the PMU state Jacobian C are not exported by the current PowerDynamics harness. Therefore the DAE tangent is **not promoted** and all downstream Bayes/EVI/GSP/graph stages remain blocked.\n\n"
        + json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
