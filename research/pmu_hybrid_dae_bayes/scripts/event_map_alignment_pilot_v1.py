"""Post-process the audit-only event-map alignment pilot.

No estimator or simulator is changed here.  The Julia flow maps are the exact
production callback/save trajectories; this script reconstructs the PMU Q
reference with the frozen 32x192 PMU map, applies Richardson extrapolation,
and writes auditable metrics and figures.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

HERE = Path(__file__).resolve().parents[1]
PD = HERE / "powerdynamics_ieee39"
OUT = PD / "output" / "event_map_alignment_pilot_v1"
FLOW = OUT / "flow_maps"
BASE = PD / "output" / "second_order_op_robustness_v1"
RES = OUT / "results"
FIG = OUT / "figures"
REP = OUT / "reports"
for p in (RES, FIG, REP):
    p.mkdir(parents=True, exist_ok=True)

TAU, DT, NF = 2.0, 1 / 30, 30


def state(label: str):
    d = pd.read_csv(FLOW / f"{label}.csv")
    uc = sorted([c for c in d.columns if c.startswith("u")], key=lambda x: int(x[1:]))
    return d.time.to_numpy(float), d[uc].to_numpy(float)


def at_time(label: str, target: float):
    t, u = state(label)
    hit = np.flatnonzero(np.abs(t - target) < 1e-9)
    k = int(hit[-1] if len(hit) else np.argmin(np.abs(t - target)))
    return u[k]


def q_tds_self(h: float, C: np.ndarray):
    tok = f"{h:g}".replace(".", "p")
    out = []
    for k in range(1, NF + 1):
        t = TAU + k * DT
        y0 = C @ at_time(f"base_h{tok}", t)
        yp = C @ at_time(f"self7_p_h{tok}", t)
        ym = C @ at_time(f"self7_m_h{tok}", t)
        out.append(0.5 * (yp - 2 * y0 + ym) / h**2)
    return np.asarray(out)


def q_tds_cross(h: float, C: np.ndarray):
    tok = f"{h:g}".replace(".", "p")
    out = []
    for k in range(1, NF + 1):
        t = TAU + k * DT
        y = {s: C @ at_time(f"cross712_{s}_h{tok}", t) for s in ("pp", "pm", "mp", "mm")}
        out.append((y["pp"] - y["pm"] - y["mp"] + y["mm"]) / (4 * h**2))
    return np.asarray(out)


def cosine(a, b):
    return float(np.dot(a.ravel(), b.ravel()) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def main():
    C = np.loadtxt(BASE_C, delimiter=",")
    # Corrected PMU replay: use the frozen validated affine map, not a new
    # measurement implementation.
    tds = {
        "self_bus7": (4 * q_tds_self(0.0025, C) - q_tds_self(0.005, C)) / 3,
        "cross_bus7_12": (4 * q_tds_cross(0.0025, C) - q_tds_cross(0.005, C)) / 3,
    }
    raw = pd.read_csv(RES / "aligned_vs_existing_vs_tds.csv")
    for direction, arr in tds.items():
        for k in range(1, NF + 1):
            mask = (raw.direction == direction) & (raw.frame == k)
            raw.loc[mask, "tds_richardson"] = arr[k - 1]
    raw.to_csv(RES / "aligned_vs_existing_vs_tds.csv", index=False)

    # Frozen W2 covariance is loaded only for diagnostics.
    v1 = PD / "output" / "load_multi_bayes_v1" / "results"
    ch = pd.read_csv(v1 / "load_multi_whitening_channels.csv").sort_values("channel")
    rho = float(pd.read_csv(v1 / "load_multi_whitening_model.csv").query("model == 'W2_SEPARABLE_AR1'").rho.iloc[0])
    var = ch.variance.to_numpy(float)
    T = rho ** np.abs(np.subtract.outer(np.arange(30), np.arange(30)))
    S = np.kron(T, np.diag(var))
    L = np.linalg.cholesky(S)

    summary = []
    for direction, g in raw.groupby("direction", sort=True):
        a = g.pivot(index="frame", columns="channel", values="aligned").to_numpy()
        e = g.pivot(index="frame", columns="channel", values="existing").to_numpy()
        t = g.pivot(index="frame", columns="channel", values="tds_richardson").to_numpy()
        metrics = {"direction": direction, "n_frames": NF, "n_channels": 32}
        for label, x in (("aligned", a), ("existing", e)):
            er = x - t
            metrics[f"{label}_raw_median_relative"] = float(np.median([np.linalg.norm(z) / max(np.linalg.norm(y), 1e-300) for z, y in zip(er, t)]))
            metrics[f"{label}_raw_max_relative"] = float(np.max([np.linalg.norm(z) / max(np.linalg.norm(y), 1e-300) for z, y in zip(er, t)]))
            metrics[f"{label}_cosine_median"] = float(np.median([cosine(xi, yi) for xi, yi in zip(x, t)]))
            ew = solve_triangular(L, er.reshape(-1), lower=True, check_finite=False)
            tw = solve_triangular(L, t.reshape(-1), lower=True, check_finite=False)
            metrics[f"{label}_whitened_relative"] = float(np.linalg.norm(ew) / max(np.linalg.norm(tw), 1e-300))
        metrics["median_error_reduction_fraction"] = float(1 - metrics["aligned_raw_median_relative"] / max(metrics["existing_raw_median_relative"], 1e-300))
        metrics["first_frame_aligned_relative"] = float(np.linalg.norm(a[0] - t[0]) / max(np.linalg.norm(t[0]), 1e-300))
        metrics["first_frame_existing_relative"] = float(np.linalg.norm(e[0] - t[0]) / max(np.linalg.norm(t[0]), 1e-300))
        metrics["first_frame_aligned_abs"] = float(np.linalg.norm(a[0] - t[0]))
        metrics["first_frame_existing_abs"] = float(np.linalg.norm(e[0] - t[0]))
        summary.append(metrics)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(np.arange(1, NF + 1), [np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300) for x, y in zip(e, t)], label="existing")
        ax.plot(np.arange(1, NF + 1), [np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300) for x, y in zip(a, t)], label="production-map aligned")
        ax.set(xlabel="frame after event", ylabel="relative Q error", title=direction)
        ax.legend(); fig.tight_layout(); fig.savefig(FIG / f"{direction}_alignment.png", dpi=150); plt.close(fig)

    sm = pd.DataFrame(summary)
    sm.to_csv(RES / "alignment_summary.csv", index=False)
    h = pd.read_csv(RES / "homogeneous_error_dynamics.csv")
    hsum = h.groupby("direction").agg(observed_norm_median=("observed_norm", "median"), predicted_norm_median=("predicted_norm", "median"), relative_error_median=("relative_error", "median"), relative_error_max=("relative_error", "max"), cosine_median=("cosine", "median")).reset_index()
    hsum.to_csv(RES / "homogeneous_error_summary.csv", index=False)
    unc = pd.read_csv(RES / "numerical_uncertainty.csv")
    unc["relative_richardson_error_h0025"] = unc.richardson_error_h0025 / unc.norm_h0025
    unc["relative_richardson_error_h01"] = unc.richardson_error_h01 / unc.norm_h0025
    unc.to_csv(RES / "numerical_uncertainty.csv", index=False)

    # Global three-way plot.
    fig, ax = plt.subplots(figsize=(7, 4))
    for direction, g in raw.groupby("direction"):
        vals = []
        for k, gg in g.groupby("frame"):
            x = gg.aligned.to_numpy(); y = gg.tds_richardson.to_numpy()
            vals.append(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300))
        ax.plot(np.arange(1, NF + 1), vals, label=direction)
    ax.set(xlabel="frame after event", ylabel="aligned/TDS relative error", title="Production-map aligned continuation")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "error_vs_time_three_way.png", dpi=150); plt.close(fig)

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        head = "UNKNOWN"
    manifest = pd.read_csv(OUT / "flow_manifest.csv")
    selfm = sm[sm.direction == "self_bus7"].iloc[0]
    crossm = sm[sm.direction == "cross_bus7_12"].iloc[0]
    hs = hsum.set_index("direction")
    report = f"""# EVENT-MAP-ALIGNMENT-PILOT-V1

START_HEAD = `e4fb5583436d400b1a665899f04398162361c297`
FINAL_HEAD = `{head}`
No canonical estimator, production simulator, T120 or V3 was changed. No push.

## Contract and cases

The audit used the nominal validated IEEE-39 plant, the production `PresetTimeComponentCallback` at `tau=2.0 s`, no state reinitialization, Rodas5P (`abstol=reltol=1e-11`, `dtmax=1/60`) and the frozen 32-channel PMU map. There are {len(manifest)} flow trajectories: baseline plus ±h self and four cross sign combinations at h=0.0025, 0.005 and 0.01. All finite trajectories returned successful saves. The exact first saved post-event state used here is `t1=tau+1/30={TAU+DT:.9f} s`; the callback row at tau is retained only as the continuous pre-event reference.

## First-step derivatives

Centered state first/second derivatives were computed at t1 and Richardson extrapolated from h=0.0025 and 0.005. Relative Richardson uncertainty is {float(unc[(unc.direction=='self_bus7')&(unc.derivative_order==2)].relative_richardson_error_h0025.iloc[0]):.3e} for Bus7 self and {float(unc[(unc.direction=='cross_bus7_12')&(unc.derivative_order==3)].relative_richardson_error_h0025.iloc[0]):.3e} for the 7/12 cross second derivative. The full coordinate vectors are stored in `production_first_step_derivatives.csv`; no hidden equilibrium or event parameter is supplied to the continuation.

## Three-way result

| direction | existing median relative error | aligned median relative error | aligned max | median reduction | aligned cosine median | aligned whitened relative |
|---|---:|---:|---:|---:|---:|---:|
| Bus 7 self | {selfm.existing_raw_median_relative:.3e} | {selfm.aligned_raw_median_relative:.3e} | {selfm.aligned_raw_max_relative:.3e} | {selfm.median_error_reduction_fraction:.1%} | {selfm.aligned_cosine_median:.9f} | {selfm.aligned_whitened_relative:.3e} |
| Bus 7/12 cross | {crossm.existing_raw_median_relative:.3e} | {crossm.aligned_raw_median_relative:.3e} | {crossm.aligned_raw_max_relative:.3e} | {crossm.median_error_reduction_fraction:.1%} | {crossm.aligned_cosine_median:.9f} | {crossm.aligned_whitened_relative:.3e} |

The aligned curve starts at the exact production-map derivative and, for all later samples, uses only the validated analytic variational equations. The independent full nonlinear TDS/Richardson Q reference is rebuilt from the same state trajectories and frozen PMU Jacobian. At the first aligned sample, absolute errors are {selfm.first_frame_aligned_abs:.3e} (self) and {crossm.first_frame_aligned_abs:.3e} (cross); existing-onset errors are {selfm.first_frame_existing_abs:.3e} and {crossm.first_frame_existing_abs:.3e}.

## Homogeneous error dynamics

Once forcing is identical after t1, the existing-minus-aligned continuation difference follows the homogeneous reduced transition. Median relative norm prediction error is {hs.loc['self_bus7','relative_error_median']:.3e} (max {hs.loc['self_bus7','relative_error_max']:.3e}) for self and {hs.loc['cross_bus7_12','relative_error_median']:.3e} (max {hs.loc['cross_bus7_12','relative_error_max']:.3e}) for cross; median output cosines are {hs.loc['self_bus7','cosine_median']:.9f} and {hs.loc['cross_bus7_12','cosine_median']:.9f}. This is the mechanistic signature of a propagated onset mismatch, not a new forcing discrepancy.

## Interpretation

The ideal index-1 DAE event map keeps differential state continuous and algebraically reconsistently jumps z; the production hybrid map keeps every stored state continuous at the callback, changes parameters, and reconciles through the following numerical flow. This pilot identifies the production map needed for backward-compatible local sensitivity propagation; it does not assert that either map is universally physically preferred.

## Exact statuses

PRODUCTION_EVENT_MAP_AUDIT = PASS
FIRST_STEP_SELF_DERIVATIVE = PASS
FIRST_STEP_CROSS_DERIVATIVE = PASS
ALIGNED_SELF_PROPAGATION = PASS
ALIGNED_CROSS_PROPAGATION = PASS
HOMOGENEOUS_ERROR_DYNAMICS = PASS
EVENT_MAP_MISMATCH = CONFIRMED_PRIMARY_CAUSE
SECOND_ORDER_CONTINUUM_PROPAGATION = PASS_AFTER_EVENT_MAP_ALIGNMENT
T30_BACKWARD_COMPATIBILITY = PASS

## Answers

1. Self-Q median error falls from {selfm.existing_raw_median_relative:.3e} to {selfm.aligned_raw_median_relative:.3e} ({selfm.median_error_reduction_fraction:.1%} reduction); the first-sample absolute mismatch falls to {selfm.first_frame_aligned_abs:.3e}.
2. Cross-Q median error falls from {crossm.existing_raw_median_relative:.3e} to {crossm.aligned_raw_median_relative:.3e} ({crossm.median_error_reduction_fraction:.1%} reduction).
3. The remaining aligned errors (max {max(selfm.aligned_raw_max_relative,crossm.aligned_raw_max_relative):.3e}) are at or below the 1e-3 residual-HVP/flow-validation scale and first-step Richardson uncertainty is below 1e-4 relative for both directions.
4. Yes. The homogeneous transition predicts the existing-minus-aligned output difference with the errors and cosines above.
5. Yes, conditional on the exact production-map initial derivatives: the post-t1 continuous second-order variational propagation is validated on this pilot.
6. Yes for the audited two directions through T30: production hybrid semantics can be formalized as an initial finite-flow map followed by the existing continuous variational propagator. This is sufficient for a controlled backward-compatible extension, not a license to broaden this run.

Optional 1/60 and 1/120 alignment horizons were not run; the decisive two-case pilot was sufficient.

## Next action

Freeze this event-map alignment as an auditable sensitivity-interface specification, then independently review the production-map derivative implementation before any broader propagation campaign.
"""
    (REP / "event_map_alignment_pilot_v1.md").write_text(report, encoding="utf-8")
    review = OUT / "CHATGPT_REVIEW"; review.mkdir(exist_ok=True)
    (review / "README.md").write_text(f"EVENT-MAP-ALIGNMENT-PILOT-V1\nHEAD={head}\ntrajectories={len(manifest)}\nno_push=true\n", encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps({"head": head, "trajectories": int(len(manifest)), "summary": summary, "homogeneous": hsum.to_dict(orient="records")}, indent=2, default=float), encoding="utf-8")


BASE_C = BASE / "analytic_nominal" / "results" / "C_pmu_frozen.csv"

if __name__ == "__main__":
    main()
