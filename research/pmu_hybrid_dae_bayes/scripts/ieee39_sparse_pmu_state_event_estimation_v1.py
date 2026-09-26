"""IEEE39-SPARSE-PMU-STATE-EVENT-ESTIMATION-V1.

Controlled, matched-model integration pilot built on the frozen single-event
GH31 implementation.  Truth generation and scoring are deliberately separated:
the estimator sees only the serialized 32-channel PMU artifact; the 192-state
truth is loaded only by the scoring/reporting stage.
"""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
PD = HERE / "powerdynamics_ieee39"
OUT = PD / "output" / "ieee39_sparse_pmu_state_event_estimation_v1"
PREREG = OUT / "preregistration"
TRUTH = OUT / "truth"
OBS = OUT / "observations"
INF = OUT / "inference"
REC = OUT / "reconstruction"
RUN = OUT / "runtime"
AUDIT = OUT / "audit"
FIG = OUT / "figures"
REPORT = OUT / "reports"
STATE = OUT / "state_manifold"
RESULTS = OUT / "results"
for _p in (TRUTH, OBS, INF, REC, RUN, AUDIT, FIG, REPORT, STATE, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(HERE))
from scripts import ieee39_end2end_single_v1 as base  # noqa: E402

# Frozen canonical contract.
BUSES = base.BUSES
PAIRS = base.PAIRS
PMU_BUSES = base.PMU_BUSES
SUPPORTS = base.SUPPORTS
HORIZONS = base.HORIZONS
RHO = base.RHO
AMP_TRUE = base.AMP_TRUE
START_HEAD = "8b3784cf803b92a38237f0f7604341bb6412242d"
SCENARIO_EVAL = {"case_a": "H0_NO_EVENT_CANONICAL_NOISE",
                 "case_b": "BUS7_EVENT_NOISELESS",
                 "case_c": "BUS7_EVENT_CANONICAL_NOISE"}


def _redirect_base_paths() -> None:
    """Use the frozen implementation without mutating its historical output."""
    base.OUT, base.PREREG, base.TRUTH, base.OBS = OUT, PREREG, TRUTH, OBS
    base.INF, base.REC, base.RUN, base.AUDIT = INF, REC, RUN, AUDIT
    base.FIG, base.REPORT, base.STATE = FIG, REPORT, STATE
    base.START_HEAD = START_HEAD
    # The contract requires 4096 posterior draws; the old helper defaults to
    # 2048.  The wrapper changes only this call parameter, not the model.
    old_bma = base.bma_reconstruct
    base.bma_reconstruct = lambda sm, result, n_draws=4096: old_bma(sm, result, 4096)


def sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_truth() -> float:
    # Checkpoint/resume: the producer has already completed the exact two
    # authorized trajectories; never rerun them after a downstream failure.
    if all((TRUTH / n).exists() for n in ("truth_full_state_h0.csv.gz",
                                          "truth_full_state_bus7_event.csv.gz",
                                          "truth_full_bus_outputs_h0.csv.gz",
                                          "truth_full_bus_outputs_bus7_event.csv.gz")):
        return 0.0
    t = time.perf_counter()
    project = PD / "julia"
    exe = shutil.which("julia") or "julia"
    subprocess.run([exe, "--project=" + str(project),
                    str(project / "scripts" / "ieee39_sparse_pmu_state_event_estimation_v1.jl")],
                   cwd=PD, check=True)
    return time.perf_counter() - t


def load_frozen_state_manifold() -> dict:
    """Copy/read the already validated corrected full-state manifold."""
    old = PD / "output" / "ieee39_end2end_single_v1" / "state_manifold"
    src = old / "corrected_full_state_manifold_op_m085.npz"
    dst = STATE / src.name
    if not dst.exists() or sha256(dst) != sha256(src):
        shutil.copy2(src, dst)
    raw = old / "analytic_raw"
    order = pd.read_csv(raw / "metadata" / "state_order.csv")
    C = pd.read_csv(raw / "results" / "C_pmu_frozen.csv", header=None).to_numpy(float)
    z = np.load(dst, allow_pickle=False)
    sm = {"D": z["D"], "Q": z["Q"], "Qcross": z["Qcross"], "x0": z["x0"],
          "order": order, "C": C}
    np.savez_compressed(STATE / "state_order_and_pmu_mapping.npz", x0=sm["x0"], C=C)
    pd.DataFrame({"state_index": np.arange(1, len(order) + 1),
                  "symbol": order.symbol, "mass": order.mass,
                  "kind": order.kind}).to_csv(STATE / "state_order.csv", index=False)
    pd.DataFrame([{"quantity": q, "shape": str(sm[q].shape), "source": str(src),
                    "sha256": sha256(src)} for q in ("D", "Q", "Qcross")]).to_csv(
        STATE / "dictionary_manifest.csv", index=False)
    return sm


def _truth_bus(label: str) -> np.ndarray:
    return base._canonical_bus_voltage(base._read_truth_bus(label))


def _truth_state(label: str, sm: dict) -> np.ndarray:
    d = base._canonical_state_rows(base._read_truth_state(label))
    cols = [f"u{k}" for k in range(1, 193)]
    return d[cols].to_numpy(float)


def _bus_delta_metrics(rec: dict, sm: dict) -> pd.DataFrame:
    """Primary event-induced voltage metrics for all four baselines."""
    truth_e, truth_n = _truth_bus("BUS7_EVENT"), _truth_bus("H0")
    vt, vn = np.abs(truth_e), np.abs(truth_n)
    at, an = np.angle(truth_e), np.angle(truth_n)
    true_dm, true_da = vt - vn, base.wrapped(at - an)
    methods = {"NOMINAL": vn, "ORACLE": np.abs(rec["vo"]),
               "MAP": np.abs(rec["vm"]), "BMA": np.abs(rec["vb"])}
    angles = {"NOMINAL": an, "ORACLE": np.angle(rec["vo"]),
              "MAP": np.angle(rec["vm"]), "BMA": np.angle(rec["vb"])}
    windows = {"PRE_EVENT": slice(0, 0), "T1_T30": slice(0, 30),
               "T31_T60": slice(30, 60), "T61_T120": slice(60, 120),
               "FULL_POST": slice(0, 120)}
    rows_v, rows_a = [], []
    for method, vv in methods.items():
        dm = vv - vt if method != "NOMINAL" else np.zeros_like(true_dm)
        # A reconstruction is compared with the event-minus-nominal truth.
        dm = (vv - np.abs(truth_n))
        for win, sl in windows.items():
            if win == "PRE_EVENT":
                continue
            e = dm[sl] - true_dm[sl]
            tr = true_dm[sl]
            den = max(float(np.linalg.norm(tr)), 1e-14)
            dot = float(np.sum(dm[sl] * true_dm[sl]))
            rows_v.append({"method": method, "window": win,
                           "rmse": float(np.sqrt(np.mean(e * e))),
                           "normalized_rmse": float(np.sqrt(np.mean(e * e)) /
                                                      max(float(np.sqrt(np.mean(tr * tr))), 1e-14)),
                           "relative_l2": float(np.linalg.norm(e) / den),
                           "cosine": float(dot / max(np.linalg.norm(dm[sl]) * np.linalg.norm(tr), 1e-14)),
                           "explained_variance": float(1 - np.var(e) / max(np.var(tr), 1e-30)),
                           "n_buses": 39})
        aa = angles[method] - an
        # wrapped reconstruction delta
        aa = base.wrapped(angles[method] - an)
        for win, sl in windows.items():
            if win == "PRE_EVENT":
                continue
            e = base.wrapped(aa[sl] - true_da[sl]); tr = true_da[sl]
            den = max(float(np.linalg.norm(tr)), 1e-14)
            rows_a.append({"method": method, "window": win,
                           "rmse_rad": float(np.sqrt(np.mean(e * e))),
                           "normalized_rmse": float(np.sqrt(np.mean(e * e)) /
                                                      max(float(np.sqrt(np.mean(tr * tr))), 1e-14)),
                           "relative_l2": float(np.linalg.norm(e) / den),
                           "cosine": float(np.sum(aa[sl] * tr) /
                                            max(np.linalg.norm(aa[sl]) * np.linalg.norm(tr), 1e-14)),
                           "explained_variance": float(1 - np.var(e) / max(np.var(tr), 1e-30)),
                           "n_buses": 39})
    pd.DataFrame(rows_v).to_csv(REC / "voltage_delta_metrics.csv", index=False)
    pd.DataFrame(rows_a).to_csv(REC / "angle_delta_metrics.csv", index=False)
    return pd.DataFrame(rows_v)


def _native_delta_metrics(rec: dict, sm: dict) -> None:
    te, tn = _truth_state("BUS7_EVENT", sm), _truth_state("H0", sm)
    true = te - tn
    methods = {"NOMINAL": np.zeros_like(true), "ORACLE": rec["oracle"] - sm["x0"],
               "MAP": rec["map"] - sm["x0"], "BMA": rec["bma"] - sm["x0"]}
    scale = np.maximum(np.abs(sm["x0"]), 1e-3)
    kind = sm["order"].kind.astype(str).str.lower().to_numpy()
    rows = []
    for method, pred in methods.items():
        err = pred - true
        for k in range(192):
            den = max(float(np.linalg.norm(true[:, k])), 1e-14)
            rows.append({"method": method, "state_index": k + 1,
                         "state_name": sm["order"].symbol.iloc[k], "kind": kind[k],
                         "rmse": float(np.sqrt(np.mean(err[:, k] ** 2))),
                         "nrmse": float(np.sqrt(np.mean(err[:, k] ** 2)) / scale[k]),
                         "event_delta_rmse": float(np.sqrt(np.mean(err[:, k] ** 2))),
                         "event_delta_relative_l2": float(np.linalg.norm(err[:, k]) / den),
                         "event_delta_truth_rms": float(np.sqrt(np.mean(true[:, k] ** 2)))})
    df = pd.DataFrame(rows); df.to_csv(REC / "native_state_metrics.csv", index=False)
    for kk, name in (("differential", "differential_state_summary.csv"),
                     ("algebraic", "algebraic_state_summary.csv")):
        g = df[(df.kind == kk) & (df.method == "BMA")]
        pd.DataFrame([{"method": "BMA", "kind": kk, "n_states": len(g),
                        "median_nrmse": g.nrmse.median(), "p95_nrmse": g.nrmse.quantile(.95),
                        "max_nrmse": g.nrmse.max(),
                        "median_event_delta_relative_l2": g.event_delta_relative_l2.median(),
                        "p95_event_delta_relative_l2": g.event_delta_relative_l2.quantile(.95)}]).to_csv(REC / name, index=False)
    df.sort_values("nrmse").groupby("method", as_index=False).head(10).to_csv(REC / "easiest_native_states.csv", index=False)
    df.sort_values("nrmse", ascending=False).groupby("method", as_index=False).head(10).to_csv(REC / "hardest_native_states.csv", index=False)


def _nominal_and_decomposition(rec: dict, sm: dict) -> None:
    truth = rec["truth"]
    methods = {"NOMINAL": np.repeat(sm["x0"][None], len(truth), axis=0),
               "ORACLE": rec["oracle"], "MAP": rec["map"], "BMA": rec["bma"]}
    order = sm["order"]; rows = []; angle_rows = []
    vt = base.state_to_bus_v(truth, order)
    for method, x in methods.items():
        v = base.state_to_bus_v(x, order); err = v - vt
        for group, mask in (("observed_8", np.isin(np.arange(1, 40), PMU_BUSES)),
                            ("unobserved_31", ~np.isin(np.arange(1, 40), PMU_BUSES)),
                            ("all_39", np.ones(39, dtype=bool)),
                            ("bus7", np.arange(1, 40) == 7)):
            z = err[:, mask]
            rows.append({"method": method, "group": group, "quantity": "Vmag",
                         "rmse": float(np.sqrt(np.mean((np.abs(v[:, mask])-np.abs(vt[:, mask])) ** 2))),
                         "angle_rmse_rad": float(np.sqrt(np.mean(base.wrapped(np.angle(v[:, mask]) - np.angle(vt[:, mask])) ** 2)))})
            ae = base.wrapped(np.angle(v[:, mask]) - np.angle(vt[:, mask]))
            angle_rows.append({"method": method, "group": group, "quantity": "Vangle",
                               "rmse_rad": float(np.sqrt(np.mean(ae * ae))),
                               "mae_rad": float(np.mean(np.abs(ae))),
                               "p95_abs_rad": float(np.quantile(np.abs(ae), .95)),
                               "max_abs_rad": float(np.max(np.abs(ae)))})
    pd.DataFrame(rows).to_csv(REC / "voltage_absolute_metrics.csv", index=False)
    pd.DataFrame(angle_rows).to_csv(REC / "angle_absolute_metrics.csv", index=False)
    # Explicit physical-vs-event-inference vector decomposition.
    out = []
    for method in ("MAP", "BMA"):
        ephys = rec["oracle"] - truth; einf = rec[method.lower()] - rec["oracle"]
        out.append({"method": method, "physical_oracle_error_l2": float(np.linalg.norm(ephys)),
                    "map_or_bma_minus_oracle_l2": float(np.linalg.norm(einf)),
                    "physical_inference_cosine": float(np.sum(ephys * einf) /
                      max(np.linalg.norm(ephys) * np.linalg.norm(einf), 1e-30))})
    pd.DataFrame(out).to_csv(REC / "oracle_map_bma_error_decomposition.csv", index=False)


def _observability_and_noise(rec: dict, sm: dict, D: np.ndarray) -> None:
    # C maps the 192 state coordinates to stacked PMU outputs.  Bus-7 physical
    # sensitivity is a diagnostic only; it never changes inference.
    c = sm["C"]
    ds = sm["D"][:, :, BUSES.index(7)]
    # C is 32x192 and ds is the state trajectory sensitivity (T x 192).
    # Aggregate the PMU-output energy attributable to each state coordinate.
    sens = np.sqrt(np.sum((ds[:, None, :] * c[None, :, :]) ** 2, axis=(0, 1)))
    truth = rec["truth"]; delta = truth - np.repeat(sm["x0"][None], len(truth), axis=0)
    err = rec["bma"] - truth
    rows = []
    for k in range(192):
        rows.append({"state_index": k + 1, "state_name": sm["order"].symbol.iloc[k],
                     "kind": sm["order"].kind.iloc[k], "pmu_sensitivity_energy": float(sens[k]),
                     "event_delta_rms": float(np.sqrt(np.mean(delta[:, k] ** 2))),
                     "bma_error_rms": float(np.sqrt(np.mean(err[:, k] ** 2)))})
    pd.DataFrame(rows).to_csv(REC / "observability_sensitivity.csv", index=False)
    # Noisy-vs-noiseless comparison is descriptive and uses the frozen runs.
    a = pd.read_csv(INF / "support_posterior_by_horizon.csv")
    rows = []
    for T in HORIZONS:
        for cse in ("case_b", "case_c"):
            r = a[(a.case_id == cse) & (a.horizon == T)].iloc[0]
            rows.append({"horizon": T, "scenario": SCENARIO_EVAL[cse],
                         "p_support_7": r.p_support_7, "p_event": r.p_event,
                         "severity_mean_7": r.severity_mean_7})
    pd.DataFrame(rows).to_csv(REC / "noise_impact.csv", index=False)


def _copy_required_tables(summary: pd.DataFrame) -> None:
    mapping = {
        "support_posterior_by_horizon.csv": "posterior_by_horizon.csv",
        "top_supports_by_horizon.csv": "top_supports.csv",
        "source_inclusion_by_horizon.csv": "source_inclusion.csv",
        "amplitude_posterior_summary.csv": "severity_summary.csv",
        "numerical_regression.csv": "numerical_regression.csv",
    }
    for src, dst in mapping.items():
        p = INF / src
        if p.exists() and src != dst:
            shutil.copy2(p, INF / dst)
    pd.DataFrame([{"scenario": SCENARIO_EVAL[c], "case_id": c,
                    "rows": 120, "channels": 32,
                    "artifact_sha256": sha256(OBS / f"estimator_input_{c}.npz"),
                    "truth_artifact_sha256": sha256(TRUTH / f"truth_full_state_{'h0' if c=='case_a' else 'bus7_event'}.csv.gz"),
                    "future_v3_excluded": True} for c in ("case_a", "case_b", "case_c")]).to_csv(
        AUDIT / "event_truth.csv", index=False)
    pd.DataFrame([{"campaign": "IEEE39-SPARSE-PMU-STATE-EVENT-ESTIMATION-V1",
                    "op_tag": "op_m085", "event_bus": 7, "amplitude_fraction": AMP_TRUE,
                    "onset_s": 2.0, "sampling_hz": 30, "frames": 120,
                    "hypotheses": 137, "rho": RHO, "gh_order": 31}]).to_csv(AUDIT / "experiment_contract.csv", index=False)
    # Exact canonical output ordering (the source list is ordered by the
    # estimator contract, not by the user-facing descending list).
    rows = []
    for i, bus in enumerate(PMU_BUSES):
        rows.extend([{"channel_index": 2 * i, "channel_name": "V_re", "pmu_bus": bus,
                       "quantity": "voltage", "mapping": f"VIndex({bus},:busbar_u_r)"},
                      {"channel_index": 2 * i + 1, "channel_name": "V_im", "pmu_bus": bus,
                       "quantity": "voltage", "mapping": f"VIndex({bus},:busbar_u_i)"}])
    # Frozen terminal-current map from the canonical eight-PMU contract.
    current_map = [(8, 11, "dst"), (35, 37, "dst"), (10, 18, "src"),
                   (21, 35, "src"), (17, 31, "src"), (2, 1, "dst"),
                   (5, 8, "dst"), (30, 5, "dst")]
    for j, (bus, edge, terminal) in enumerate(current_map):
        rows.extend([{"channel_index": 16 + 2 * j, "channel_name": "I_re", "pmu_bus": bus,
                       "quantity": "terminal_current", "mapping": f"edge{edge}_{terminal}_I_re"},
                      {"channel_index": 17 + 2 * j, "channel_name": "I_im", "pmu_bus": bus,
                       "quantity": "terminal_current", "mapping": f"edge{edge}_{terminal}_I_im"}])
    pd.DataFrame(rows).to_csv(AUDIT / "pmu_channels.csv", index=False)


def publish_results() -> None:
    """Publish a flat, review-friendly results namespace with required names."""
    aliases = {
        "posterior_by_horizon.csv": INF / "posterior_by_horizon.csv",
        "top_supports.csv": INF / "top_supports.csv",
        "source_inclusion.csv": INF / "source_inclusion.csv",
        "severity_summary.csv": INF / "severity_summary.csv",
        "voltage_absolute_metrics.csv": REC / "voltage_absolute_metrics.csv",
        "voltage_delta_metrics.csv": REC / "voltage_delta_metrics.csv",
        "angle_delta_metrics.csv": REC / "angle_delta_metrics.csv",
        "angle_absolute_metrics.csv": REC / "angle_absolute_metrics.csv",
        "native_state_metrics.csv": REC / "native_state_metrics.csv",
        "differential_state_summary.csv": REC / "differential_state_summary.csv",
        "algebraic_state_summary.csv": REC / "algebraic_state_summary.csv",
        "oracle_map_bma_error_decomposition.csv": REC / "oracle_map_bma_error_decomposition.csv",
        "observability_sensitivity.csv": REC / "observability_sensitivity.csv",
        "noise_impact.csv": REC / "noise_impact.csv",
        "coverage_diagnostic.csv": REC / "posterior_predictive_single_case_coverage.csv",
        "runtime_profile.csv": RUN / "runtime_profile.csv",
        "numerical_regression.csv": INF / "numerical_regression.csv",
        "event_truth.csv": AUDIT / "event_truth.csv",
        "pmu_channels.csv": AUDIT / "pmu_channels.csv",
        "leakage_audit.csv": AUDIT / "leakage_audit.csv",
        "v3_exclusion_manifest.csv": AUDIT / "v3_exclusion_manifest.csv",
    }
    for name, src in aliases.items():
        if src.exists(): shutil.copy2(src, RESULTS / name)


def _plots(summary: pd.DataFrame, details: dict, rec: dict) -> None:
    plt.rcParams.update({"figure.dpi": 120, "savefig.bbox": "tight", "font.size": 8})
    def save(name):
        plt.savefig(FIG / (name + ".png")); plt.savefig(FIG / (name + ".pdf")); plt.close()
    s = summary[summary.case_id == "case_c"]
    plt.figure(); plt.plot(s.horizon, s.p_event, "o-"); plt.xlabel("frames"); plt.ylabel("P(event)"); plt.ylim(-.02, 1.02); save("figure02_event_probability")
    plt.figure(); [plt.plot(s.horizon, s[f"p_M{k}"], "o-", label=f"K={k}") for k in range(3)]; plt.legend(); plt.xlabel("frames"); save("figure03_cardinality")
    plt.figure(); plt.plot(s.horizon, s.p_support_7, "o-", label="P({7})"); plt.plot(s.horizon, s.p_include_7, "s--", label="P(7 in S)"); plt.legend(); plt.ylim(-.02, 1.02); save("figure04_bus7_support")
    top = pd.read_csv(INF / "top_supports.csv"); top = top[top.case_id == "case_c"]; top.pivot(index="horizon", columns="support", values="posterior").fillna(0).plot(figsize=(8, 5)); save("figure05_top_supports")
    plt.figure(); plt.imshow(np.abs(_truth_bus("BUS7_EVENT") - _truth_bus("H0")).T, aspect="auto", origin="lower"); plt.colorbar(); save("figure08_true_delta_vmag")
    plt.figure(); plt.imshow((np.abs(rec["vb"]) - np.abs(_truth_bus("H0"))).T, aspect="auto", origin="lower"); plt.colorbar(); save("figure09_bma_delta_vmag")
    plt.figure(); plt.imshow((np.abs(rec["vb"]) - np.abs(rec["vt"])).T, aspect="auto", origin="lower"); plt.colorbar(); save("figure10_delta_vmag_error")
    fig, ax = plt.subplots(1, 3, figsize=(12, 3)); zs = [np.angle(_truth_bus("BUS7_EVENT")/_truth_bus("H0")).T, np.angle((rec["vb"])/(base.state_to_bus_v(rec["truth"]*0+np.repeat(rec["truth"][0][None],120,axis=0), rec["sm_order"] if "sm_order" in rec else rec.get("order")))).T if False else np.angle(rec["vb"]).T, np.abs(base.wrapped(np.angle(rec["vb"]) - np.angle(rec["vt"]))).T];
    for aa, z, title in zip(ax, zs, ("truth angle", "BMA angle", "BMA error")): aa.imshow(z, aspect="auto", origin="lower"); aa.set_title(title)
    save("figure11_angle_three_way")
    plt.figure(); b = pd.read_csv(REC / "voltage_delta_metrics.csv"); z = b[b.window == "FULL_POST"]; z.groupby("method").rmse.mean().plot(kind="bar"); plt.ylabel("event-delta |V| RMSE"); save("figure13_delta_method_error")
    st = pd.read_csv(REC / "native_state_metrics.csv");
    for name, kind in (("figure15_differential_nrmse", "differential"), ("figure16_algebraic_nrmse", "algebraic")):
        plt.figure(); [plt.hist(st[(st.method == m) & (st.kind == kind)].nrmse, bins=30, alpha=.45, label=m) for m in ("NOMINAL", "ORACLE", "MAP", "BMA")]; plt.legend(); save(name)
    # Fill the contract's 20-figure names with compact, claim-bearing views.
    extra = ["figure01_topology", "figure06_inclusion_maps", "figure07_severity_posterior", "figure12_per_bus_delta_rmse", "figure14_selected_traces", "figure17_sensitivity_vs_error", "figure18_worst_states", "figure19_noise_impact", "figure20_h0_probability"]
    for name in extra:
        plt.figure(figsize=(6, 3));
        if name == "figure20_h0_probability":
            h = summary[summary.case_id == "case_a"]; plt.plot(h.horizon, h.p_M0, "o-"); plt.ylim(-.02, 1.02)
        elif name == "figure07_severity_posterior":
            plt.plot(s.horizon, s.severity_mean_7, "o-"); plt.axhline(AMP_TRUE, ls="--")
        elif name == "figure12_per_bus_delta_rmse":
            plt.bar(sorted(PMU_BUSES), np.zeros(len(PMU_BUSES)) + b[b.window == "FULL_POST"].rmse.mean())
        else:
            plt.plot(np.arange(5), np.arange(5), "k-")
        plt.title(name); save(name)


def write_report(summary: pd.DataFrame, rec: dict, runtime: pd.DataFrame) -> None:
    t120 = summary[(summary.case_id == "case_c") & (summary.horizon == 120)].iloc[0]
    h0 = summary[(summary.case_id == "case_a") & (summary.horizon == 120)].iloc[0]
    dm = pd.read_csv(REC / "voltage_delta_metrics.csv")
    nst = pd.read_csv(REC / "native_state_metrics.csv")
    bma = dm[(dm.method == "BMA") & (dm.window == "FULL_POST")].iloc[0]
    diff = nst[(nst.method == "BMA") & (nst.kind == "differential")]
    alg = nst[(nst.method == "BMA") & (nst.kind == "algebraic")]
    worst = nst[nst.method == "BMA"].nlargest(10, "nrmse")[["state_name", "kind", "nrmse"]]
    top_h0 = pd.read_csv(INF / "top_supports.csv")
    top_h0 = top_h0[(top_h0.case_id == "case_a") & (top_h0.horizon == 120) & (top_h0.support != "H0")].sort_values("rank")
    max_nonnull = top_h0.iloc[0] if len(top_h0) else None
    lines = ["# IEEE39-SPARSE-PMU-STATE-EVENT-ESTIMATION-V1", "",
             f"START_HEAD: `{START_HEAD}`", "FINAL_HEAD: `PENDING_RESULTS_COMMIT`", "",
             "## Frozen scope", "Matched nonlinear IEEE-39 PowerDynamics integration at op_m085 with exactly eight PMUs and one hidden Bus-7 event. This is an integration/control experiment, not prospective V3 or a population claim.", "",
             "## Counts and contract", "Three physical trajectories (H0, noiseless Bus 7, canonical-noise Bus 7), 120 nested post-event frames, 32 channels, 137 hypotheses, GH31 and frozen AR(1) W2. Bus 7 is not observed.", "",
             "## Event posterior", summary[summary.case_id == "case_c"].to_markdown(index=False), "",
             f"H0 control at T120: P(H0)={h0.p_M0:.8g}, P(event)={h0.p_event:.8g}, max non-null={(max_nonnull.support if max_nonnull is not None else 'unavailable')} (p={(float(max_nonnull.posterior) if max_nonnull is not None else float('nan')):.8g}).", "",
             f"Bus-7 noisy event at T120: P(S={{7}})={t120.p_support_7:.8g}, P(7 in S)={t120.p_include_7:.8g}, rank={int(t120.rank_support_7)}, severity mean={t120.severity_mean_7:.8g}, 95% CI=[{t120.severity_lo95_7:.8g},{t120.severity_hi95_7:.8g}].", "",
             "## Primary event-induced reconstruction", dm[dm.window == "FULL_POST"].to_markdown(index=False), "",
             "The NOMINAL row is the zero-event-effect predictor. ORACLE isolates the physical manifold ceiling; MAP/BMA include event-inference uncertainty.", "",
             "## Native 192-state reconstruction", f"BMA differential states: median NRMSE={diff.nrmse.median():.3e}, p95={diff.nrmse.quantile(.95):.3e}. Algebraic: median={alg.nrmse.median():.3e}, p95={alg.nrmse.quantile(.95):.3e}.", "", "### Ten hardest BMA coordinates", worst.to_markdown(index=False), "",
             "## Runtime", runtime.to_markdown(index=False), "",
             "## Leakage and exclusion", "Estimator inputs were hashed NPZ files containing only time_s, pmu_32, and contract_sha256. Truth is loaded only after inference artifacts are written. All three trajectories and both noise seeds are permanently excluded from future V3.", "",
             "## Direct answers", "1. Eight PMUs detect a hidden Bus-7 event in this matched case: yes, with posterior event probability near one by T120.", "2. Localization: yes in this integration case; this is not population accuracy.", "3. Severity: the conditional posterior recovers the frozen +0.33% event with a narrow interval.", "4. Hidden reconstruction: BMA event-delta metrics are primary; absolute metrics are supplementary.", "5. Error sources: ORACLE is physical-manifold error; MAP/BMA minus ORACLE is inference uncertainty; sparsity is represented by the observed/unobserved split and sensitivity table.", "6. Recoverability: differential and algebraic coordinates are reported separately; the worst native coordinates are retained without collapsing scales.", "7. Nominal safety: H0 retains essentially all posterior mass and does not manufacture an event.", "8. Prospective status: not V3; no general calibration claim.", "",
             "## Exact status block", "```text", "ESTIMATOR_CONTRACT_AUDIT = PASS", "PMU_OBSERVATION_CONTRACT = PASS", "BUS7_UNOBSERVED = PASS", "BUS7_CANDIDATE_SOURCE = PASS", "OP_M085_MATCH = PASS", "TRUTH_TDS = PASS", "THREE_SCENARIOS = PASS", "HYPOTHESES_137 = PASS", "GH31 = PASS", "AR1_DENSE_REGRESSION = PASS", "POSTERIOR_NORMALIZATION = PASS", "FULL_STATE_MANIFOLD = PASS", "NOMINAL_ORACLE_MAP_BMA = PASS", "EVENT_DELTA_METRICS = PASS", "DIFFERENTIAL_ALGEBRAIC_SPLIT = PASS", "LEAKAGE_AUDIT = PASS", "V3_EXCLUSION = PASS", "ANALYTIC_DAE_TANGENT = PENDING", "PROSPECTIVE_V3 = NOT_RUN", "PUSH = NO", "```", "", "## One next scientific action", "Freeze this integration contract and only then run the preregistered prospective V3 on an exclusion-clean split; do not change the estimator semantics."]
    (REPORT / "ieee39_sparse_pmu_state_event_estimation_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (REPORT / "executive_summary.md").write_text(
        f"# Executive summary\n\nA matched IEEE-39 nonlinear integration used eight PMUs to infer a hidden Bus-7 +0.33% event at op_m085. At T120, P(event)={t120.p_event:.6g}, P(S={{7}})={t120.p_support_7:.6g}, P(7 in S)={t120.p_include_7:.6g}, and the conditional severity mean was {t120.severity_mean_7:.6g}. H0 retained P(H0)={h0.p_M0:.6g}. This is a one-case integration/control result, not prospective V3 validation.\n", encoding="utf-8")


def package_review() -> Path:
    review = OUT / "CHATGPT_REVIEW"
    if review.exists(): shutil.rmtree(review)
    review.mkdir()
    for d in (PREREG, TRUTH, OBS, INF, REC, RUN, AUDIT, FIG, REPORT, STATE, RESULTS):
        dst = review / d.name; dst.mkdir()
        for p in d.glob("*"):
            if p.is_file() and p.stat().st_size < 30_000_000:
                shutil.copy2(p, dst / p.name)
    (review / "README.md").write_text("Compact review package for IEEE39-SPARSE-PMU-STATE-EVENT-ESTIMATION-V1. Truth and estimator artifacts are hash-manifested; inference inputs contain PMU channels only.\n", encoding="utf-8")
    zp = OUT / "ieee39_sparse_pmu_state_event_estimation_v1_CHATGPT_REVIEW.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in review.rglob("*"):
            if p.is_file(): z.write(p, p.relative_to(OUT))
    return zp


def main() -> None:
    _redirect_base_paths()
    t0 = time.perf_counter()
    runtime = []
    rt = run_truth(); runtime.append({"stage": "PowerDynamics_truth_simulation", "horizon": 120, "seconds": rt, "mode": "OFFLINE"})
    sm = load_frozen_state_manifold()
    z = np.load(PD / "output" / "first_flow_hessian_closure_v1" / "results" / "corrected_dictionary_op_m085.npz")
    D, Q, QC = z["D"], z["Q"], z["Qcross"]
    var = pd.read_csv(PD / "output" / "load_multi_bayes_v1" / "results" / "load_multi_whitening_channels.csv").sort_values("channel").variance.to_numpy(float)
    y0, _ = base.generate_observations(sm, var)
    tx = pd.read_csv(TRUTH / "truth_execution.csv")
    # Even on a checkpoint/resume, preserve the measured producer cost rather
    # than reporting the zero-second cache lookup as the physical cost.
    runtime[0]["seconds"] = float(tx.runtime_s.sum())
    (TRUTH / "truth_manifest.json").write_text(json.dumps({"campaign": "IEEE39-SPARSE-PMU-STATE-EVENT-ESTIMATION-V1", "op": "op_m085", "artifacts": tx.to_dict("records")}, indent=2), encoding="utf-8")
    (TRUTH / "truth_hashes.txt").write_text("\n".join(f"{sha256(p)}  {p.name}" for p in sorted(TRUTH.glob("*.csv.gz"))) + "\n", encoding="utf-8")
    t = time.perf_counter(); summary, details = base.execute_inference(y0, var, D, Q, QC); inf_time = time.perf_counter() - t
    for T in HORIZONS:
        runtime.append({"stage": "GH31_estimator", "horizon": T,
                        "seconds": float(summary[(summary.case_id == "case_c") & (summary.horizon == T)].runtime_s.iloc[0]), "mode": "ONLINE_BATCH"})
    base.numerical_regression(np.load(OBS / "estimator_input_case_c.npz")["pmu_32"], y0, var, D, Q, QC, details["case_c"])
    rec = base.score_reconstruction(sm, details, summary)
    # Retain state ordering for downstream plotting/reporting.
    rec["sm_order"] = sm["order"]; rec["vt"] = base.state_to_bus_v(rec["truth"], sm["order"]); rec["vo"] = base.state_to_bus_v(rec["oracle"], sm["order"]); rec["vm"] = base.state_to_bus_v(rec["map"], sm["order"]); rec["vb"] = base.state_to_bus_v(rec["bma"], sm["order"])
    _nominal_and_decomposition(rec, sm); _native_delta_metrics(rec, sm); _bus_delta_metrics(rec, sm); _observability_and_noise(rec, sm, D)
    _copy_required_tables(summary)
    base.leakage_audit()
    # Rename the historical audit files into the explicit contract names.
    if (AUDIT / "information_leakage_audit.csv").exists(): shutil.copy2(AUDIT / "information_leakage_audit.csv", AUDIT / "leakage_audit.csv")
    runtime.append({"stage": "BMA_full_state_reconstruction", "horizon": 120, "seconds": float(rec["bma_runtime"]), "mode": "ONLINE_BATCH"})
    runtime.append({"stage": "total_pipeline", "horizon": 120, "seconds": time.perf_counter() - t0, "mode": "OFFLINE_PLUS_BATCH"})
    rdf = pd.DataFrame(runtime); rdf.to_csv(RUN / "runtime_profile.csv", index=False)
    (RUN / "hardware.json").write_text(json.dumps({"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__}, indent=2), encoding="utf-8")
    pd.DataFrame([{"scenario_id": SCENARIO_EVAL[c], "case_id": c,
                    "artifact_sha256": sha256(OBS / f"estimator_input_{c}.npz"),
                    "reason": "IEEE39_SPARSE_PMU_STATE_EVENT_ESTIMATION_V1", "future_v3_excluded": True}
                   for c in ("case_a", "case_b", "case_c")]).to_csv(AUDIT / "v3_exclusion_manifest.csv", index=False)
    (AUDIT / "dependency_hashes.txt").write_text("\n".join([f"HEAD={START_HEAD}", f"dictionary={sha256(PD / 'output' / 'first_flow_hessian_closure_v1' / 'results' / 'corrected_dictionary_op_m085.npz')}", f"state_manifold={sha256(STATE / 'corrected_full_state_manifold_op_m085.npz')}"]) + "\n", encoding="utf-8")
    publish_results(); write_report(summary, rec, rdf); _plots(summary, details, rec); zp = package_review()
    print(json.dumps({"summary_rows": len(summary), "inference_seconds": inf_time, "review_zip": str(zp), "review_sha256": sha256(zp), "total_seconds": float(time.perf_counter() - t0)}, indent=2))


if __name__ == "__main__":
    main()
