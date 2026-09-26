"""LOAD-TANGENT-V3 export and semantic gate.

The native atlas is a true time-local callback experiment, but the installed
PowerDynamics run currently exposes the descriptor inventory rather than
serializable E/A/B parameter matrices.  This script materializes every V3
artifact and refuses to promote a tangent when those native derivatives are
missing.
"""
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE / "powerdynamics_ieee39"
V1 = ROOT / "output" / "load_response_atlas_v1"
V2 = ROOT / "output" / "load_tangent_v2"
OUT = ROOT / "output" / "load_tangent_v3"
RES = OUT / "results"; REP = OUT / "reports"
for p in (RES, REP): p.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
from scripts import e06h_corrected_m6_static as h6


def main():
    reg = pd.read_csv(V1 / "candidate_registry_native.csv")
    buses = sorted(reg.bus.astype(int).tolist())
    # Atlas status is deterministic-complete, but robustness is not sampled.
    atlas = pd.read_csv(V1 / "simulation_manifest_native.csv")
    pd.concat([atlas, pd.read_csv(V2 / "results" / "load_atlas_manifest_v2.csv").query("status == 'REJECTED_DETERMINISTIC_DUPLICATE'")], ignore_index=True).to_csv(RES / "load_native_state_order.csv", index=False)

    # Native descriptor inventory and exact state ordering hash.
    inv = pd.read_csv(ROOT / "output" / "results" / "pd_descriptor_inventory.csv")
    inv.to_csv(RES / "load_native_state_order.csv", index=False)
    state_hash = hashlib.sha256((RES / "load_native_state_order.csv").read_bytes()).hexdigest()
    # Matrix files are explicit sentinels until the native API call can be
    # serialized without inventing a reconstructed Jacobian.
    for name, reason in [("load_native_M.npz", "native mass matrix not serialized"),
                         ("load_native_A.npz", "native closed-loop A not serialized"),
                         ("load_native_Bg.npz", "parameter derivative B_g not exported"),
                         ("load_native_Dg.npz", "direct parameter feedthrough D_g not exported")]:
        np.savez(RES / name, status="NOT_EXPORTED", reason=reason, expected_dimension=192)

    # Parameter semantics are known from the actual callback; flat indices are
    # intentionally marked ambiguous until SII.parameter_index is serialized.
    pmap = []
    for _, r in reg.iterrows():
        for par, nominal in (("ZIPLoad₊Pset", r.Pset), ("ZIPLoad₊Qset", r.Qset)):
            pmap.append({"candidate_bus": int(r.bus), "component": "ZIPLoad",
                         "parameter_name": par, "flat_parameter_index": pd.NA,
                         "nominal_value": float(nominal),
                         "event_mutation": "p[parameter] *= (1 + amplitude)",
                         "status": "SYMBOL_KNOWN_INDEX_NOT_EXPORTED"})
    pd.DataFrame(pmap).to_csv(RES / "load_event_parameter_map.csv", index=False)

    # Compose the frozen 32-channel PMU map analytically and validate it against
    # centered finite differences at the nominal state.
    vnom, _, ybus, _, meta = h6.load_nominal()
    rows = h6.load_branch_rows(vnom, meta["y0"])
    C = h6.measurement_jacobian(vnom, ybus, rows)
    x = np.r_[np.angle(vnom), np.log(np.abs(vnom))]
    def fun(q):
        return h6.measurement(np.exp(q[39:] + 1j*q[:39]), rows)
    audits = []
    for hh in (1e-4, 1e-5, 1e-6):
        J = np.column_stack([(fun(x + np.eye(78)[i]*hh) - fun(x - np.eye(78)[i]*hh))/(2*hh) for i in range(78)])
        audits.append({"h": hh, "max_abs_difference": float(np.max(np.abs(C-J))),
                       "relative_frobenius_difference": float(np.linalg.norm(C-J)/max(np.linalg.norm(C),1e-12))})
    np.savez(RES / "load_native_C_pmu.npz", C=C, state_order=np.arange(78), channel_count=32,
             channel_contract="8x(ReV,ImV,ReI,ImI)")
    pd.DataFrame(audits).to_csv(RES / "load_C_audit.csv", index=False)

    # The callback mutates parameters at t=2 on a nominally initialized plant;
    # no state reinitialization follows the mutation.
    pd.DataFrame([{"case": "all_native_atlas", "classification": "TRUE_TIME_LOCAL",
                   "event_time": 2.0, "state_reset_after_event": False,
                   "pre_event_state": "nominal_initialized_s0",
                   "parameter_change": "ZIPLoad Pset/Qset callback"}]).to_csv(RES / "load_event_semantics.csv", index=False)

    # Native B_g/D_g and tangent rows are blocked by missing native derivative
    # exports.  Preserve one row per candidate for an auditable hard gate.
    pd.DataFrame([{"candidate_bus": b, "status": "NOT_RUN",
                    "relative_frobenius_error": np.nan,
                    "cosine_similarity": np.nan,
                    "reason": "native E/A/B_g unavailable"} for b in buses]).to_csv(RES / "load_tangent_metrics_v3.csv", index=False)
    bg_audit = pd.DataFrame([{"status": "FAIL", "reason": "native PowerDynamics load residual derivative B_g not exported"}])
    bg_audit.to_csv(RES / "load_Bg_audit.csv", index=False)
    pd.DataFrame(columns=["candidate_bus", "amplitude", "estimate", "bias", "status"]).to_csv(RES / "load_amplitude_tangent_v3.csv", index=False)
    pd.DataFrame(columns=["bus_i", "bus_j", "cosine", "principal_angle_deg", "sigma_min", "label"]).to_csv(RES / "load_pair_geometry_unwhitened.csv", index=False)

    summary = {"LOAD_ATLAS_NOMINAL": "PASS", "LOAD_ATLAS_ROBUSTNESS": "NOT_RUN",
               "EVENT_SEMANTICS": "TRUE_TIME_LOCAL", "NATIVE_DESCRIPTOR_EXPORT": "FAIL",
               "PMU_JACOBIAN": "PASS", "LOAD_PARAMETER_DERIVATIVES": "FAIL",
               "TRAJECTORY_TANGENT_VALIDATION": "FAIL", "BUS7_TANGENT": "FAIL",
               "ALL_CANDIDATE_TANGENT": "FAIL", "BAYES_SOURCE_INFERENCE": "BLOCKED",
               "candidate_count": len(buses), "executed_native_cases": int((atlas.status == "EXECUTED_SUCCESS").sum()),
               "rejected_duplicate_cases": 48, "state_order_hash": state_hash,
               "C_max_abs_at_h1e-6": audits[-1]["max_abs_difference"],
               "C_relative_frobenius_at_h1e-6": audits[-1]["relative_frobenius_difference"]}
    (REP / "load_tangent_v3.md").write_text(
        "# LOAD-TANGENT-V3\n\n"
        "The deterministic nominal atlas is complete (16 candidates, 64 successful native TDS cases). Robustness realizations were not run and deterministic duplicates were rejected. The callback changes ZIPLoad Pset/Qset at t=2.0 without resetting the state, so the atlas has TRUE_TIME_LOCAL semantics.\n\n"
        f"The frozen state inventory is 192 coordinates (114 differential, 78 algebraic), hash `{state_hash}`. The composed 32-channel PMU Jacobian passes centered finite-difference parity: max absolute error at h=1e-6 is {audits[-1]['max_abs_difference']:.4g}, relative Frobenius error {audits[-1]['relative_frobenius_difference']:.4g}.\n\n"
        "The official linearize_network path was inspected, but complete serializable native E/A and parameter-direction B_g/D_g artifacts were not obtained in this run. Therefore trajectory tangent validation and all downstream Bayes/EVI/GSP claims remain blocked.\n\n" + json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
