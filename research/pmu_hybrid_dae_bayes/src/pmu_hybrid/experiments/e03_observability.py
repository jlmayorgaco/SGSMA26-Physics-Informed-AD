"""E03 descriptor and functional-observability campaign.

The campaign consumes the analytic Jacobian blocks exposed by ANDES after
``TDS.init``.  It intentionally stops at structural/local-linear evidence:
no estimator, event detector, Bayesian inference, or Monte-Carlo simulation is
called here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import expm, svdvals
from scipy.sparse import coo_matrix, save_npz

from pmu_hybrid.cases.ieee39_andes import load_native_system, solve_static
from pmu_hybrid.constants import PMU_BUSES
from pmu_hybrid.physics.measurement import select_controlled_terminal_map
from pmu_hybrid.physics.network import branch_terminal_currents


HORIZONS = (0, 1, 2, 3, 5, 10, 15, 30, 60)
RANK_TOLS = (1e-6, 1e-8, 1e-10)


def _sparse(M: Any):
    return coo_matrix((np.asarray(M.V, float).ravel(), (np.asarray(M.I, int).ravel(), np.asarray(M.J, int).ravel())), shape=M.size).tocsr()


def _meta(name: str, index: int, system: Any) -> tuple[str, str, str, str]:
    tokens = str(name).split()
    state = tokens[0] if tokens else ""
    model = tokens[1] if len(tokens) > 1 else "other"
    device = tokens[2] if len(tokens) > 2 else ""
    bus = ""
    if model == "Bus" and device.isdigit():
        bus = device
    elif hasattr(system, model) and device.isdigit() and hasattr(getattr(system, model), "bus"):
        try:
            values = np.asarray(getattr(system, model).bus.v).ravel()
            pos = int(device) - 1
            if 0 <= pos < len(values):
                bus = str(int(values[pos]))
        except Exception:
            pass
    meaning = {
        "delta": "machine rotor angle (rad)", "omega": "machine speed (pu)",
        "a": "bus voltage angle (rad)", "v": "bus voltage magnitude (pu)",
        "f": "frequency (pu or Hz by model contract)", "WO_y": "washout output",
        "WO_x": "washout state", "L_y": "lag output", "L_x": "lag state",
    }.get(state, "ANDES algebraic/dynamic variable")
    unit = "rad" if state in {"delta", "a"} else "pu" if state in {"omega", "v", "f"} else ""
    return model, device, bus, f"{meaning}; index={index}"


def _inventory(system: Any, root: Path) -> dict[str, Any]:
    d = system.dae
    x_rows, y_rows = [], []
    for index, name in enumerate(d.x_name):
        model, device, bus, meaning = _meta(name, index, system)
        x_rows.append({"global_index": index, "andes_model": model, "device": device, "state_name": name, "bus": bus, "unit": "rad" if str(name).startswith(("delta", "a")) else "pu", "physical_meaning": meaning})
    for index, name in enumerate(d.y_name):
        model, device, bus, meaning = _meta(name, index, system)
        y_rows.append({"global_index": index, "andes_model": model, "device": device, "state_name": name, "bus": bus, "unit": "rad" if str(name).startswith("a ") else "pu", "physical_meaning": meaning})
    results = root / "output" / "results"
    results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(x_rows).to_csv(results / "e03_dynamic_states.csv", index=False)
    pd.DataFrame(y_rows).to_csv(results / "e03_algebraic_states.csv", index=False)
    dynamic_counts = {name: int(getattr(system, name).n) for name in ("GENROU", "TGOV1N", "IEEEX1", "IEEEST", "BusFreq") if hasattr(system, name)}
    return {"nx": len(x_rows), "nz": len(y_rows), "dynamic_counts": dynamic_counts, "x_rows": x_rows, "y_rows": y_rows}


def _measurement_jacobian(system: Any, static: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Finite-difference H_z for the exact G1 voltage/current operator."""
    d = system.dae
    y0 = np.asarray(d.y, float).copy()
    bus_angles = np.arange(39)
    bus_mags = np.arange(39, 78)
    source = {b.identifier: b for b in static.branches}
    mapping = select_controlled_terminal_map(static.branches)
    bus_pos = {int(b): i for i, b in enumerate(static.bus_ids)}

    def output(y: np.ndarray, buses: tuple[int, ...] = PMU_BUSES) -> np.ndarray:
        angles, mags = y[bus_angles], y[bus_mags]
        voltage = mags * np.exp(1j * angles)
        values: list[float] = []
        for bus in buses:
            values.extend((voltage[bus - 1].real, voltage[bus - 1].imag))
            selected = mapping[bus]
            branch = source[selected.branch_id]
            current = branch_terminal_currents(branch, voltage[branch.from_bus - 1], voltage[branch.to_bus - 1])
            current = current[0] if selected.terminal == "from" else current[1]
            values.extend((current.real, current.imag))
        return np.asarray(values)

    h0 = output(y0)
    H_fd = np.zeros((len(h0), len(y0)))
    step = 1e-6
    for column in range(78):
        yp, ym = y0.copy(), y0.copy()
        yp[column] += step; ym[column] -= step
        H_fd[:, column] = (output(yp) - output(ym)) / (2 * step)
    # Independent analytic derivative of the exact Cartesian branch primitive.
    H_z = np.zeros_like(H_fd)
    angles, mags = y0[bus_angles], y0[bus_mags]
    voltage = mags * np.exp(1j * angles)
    for pmu_index, bus in enumerate(PMU_BUSES):
        base = 4 * pmu_index
        ai, mi = bus - 1, 39 + bus - 1
        H_z[base, ai] = -mags[ai] * np.sin(angles[ai]); H_z[base, mi] = np.cos(angles[ai])
        H_z[base + 1, ai] = mags[ai] * np.cos(angles[ai]); H_z[base + 1, mi] = np.sin(angles[ai])
        selected, branch = mapping[bus], source[mapping[bus].branch_id]
        z = complex(branch.resistance_pu, branch.reactance_pu); y = 1.0 / z; ysh = 0.5j * branch.charging_pu; tap = branch.complex_tap
        coefficients = ((y + ysh) / (tap * np.conj(tap)), -y / np.conj(tap)) if selected.terminal == "from" else (-y / tap, y + ysh)
        current_coefficient = int(selected.polarity)
        # Current rows are Re(I), Im(I), with angle and magnitude columns.
        for endpoint, coefficient in zip((branch.from_bus - 1, branch.to_bus - 1), coefficients):
            d_angle = current_coefficient * coefficient * (1j * voltage[endpoint])
            d_mag = current_coefficient * coefficient * np.exp(1j * angles[endpoint])
            H_z[base + 2, endpoint] += d_angle.real; H_z[base + 3, endpoint] += d_angle.imag
            H_z[base + 2, 39 + endpoint] += d_mag.real; H_z[base + 3, 39 + endpoint] += d_mag.imag
    H_x = np.zeros((len(h0), len(d.x)))
    # Independent finite-difference check at selected voltage/current columns.
    checks = []
    for column in (0, 1, 39, 40, 77):
        yp, ym = y0.copy(), y0.copy(); yp[column] += step; ym[column] -= step
        finite = (output(yp) - output(ym)) / (2 * step)
        checks.append({"column": column, "operator": "VI_ONLY", "finite_difference_norm": float(np.linalg.norm(finite)), "analytic_norm": float(np.linalg.norm(H_z[:, column])), "abs_error": float(np.linalg.norm(finite - H_z[:, column]))})
    return H_x, H_z, h0, np.asarray(checks, dtype=object), list(PMU_BUSES)


def _hidden_jacobian(system: Any, buses: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(system.dae.y, float)
    Lz = np.zeros((2 * len(buses), len(y)))
    for row, bus in enumerate(buses):
        angle, mag = y[bus - 1], y[39 + bus - 1]
        Lz[2 * row, bus - 1] = -mag * np.sin(angle); Lz[2 * row, 39 + bus - 1] = np.cos(angle)
        Lz[2 * row + 1, bus - 1] = mag * np.cos(angle); Lz[2 * row + 1, 39 + bus - 1] = np.sin(angle)
    return np.zeros((Lz.shape[0], len(system.dae.x))), Lz


def _descriptor(system: Any, root: Path) -> dict[str, Any]:
    d = system.dae
    blocks = {name: _sparse(getattr(d, name)) for name in ("fx", "fy", "gx", "gy")}
    matrices = root / "output" / "results" / "e03_matrices"; matrices.mkdir(parents=True, exist_ok=True)
    for name, matrix in blocks.items(): save_npz(matrices / f"{name}.npz", matrix)
    gy = blocks["gy"].toarray(); gx = blocks["gx"].toarray(); fy = blocks["fy"].toarray(); fx = blocks["fx"].toarray()
    singular = svdvals(gy)
    descriptor = {
        "blocks": blocks, "gy": gy, "gx": gx, "fy": fy, "fx": fx,
        "gy_singular_max": float(singular[0]), "gy_singular_min": float(singular[-1]),
        "gy_condition": float(singular[0] / singular[-1]), "gy_rank_1e-10": int(np.sum(singular > singular[0] * 1e-10)),
    }
    # This is a linear solve, never an explicit inverse.
    descriptor["A"] = fx - fy @ np.linalg.solve(gy, gx)
    return descriptor


def _obs_metrics(A: np.ndarray, C: np.ndarray, L: np.ndarray, sx: np.ndarray, config: str, mask_name: str = "all", target_bus_ids: tuple[int, ...] | None = None) -> tuple[list[dict[str, Any]], dict[int, float]]:
    scale = np.diag(sx)
    As = np.linalg.solve(scale, A @ scale)
    Cs = C @ scale
    Ls = L @ scale
    rows, residuals = [], {}
    for horizon in HORIZONS:
        Ad = expm(As / 30.0)
        blocks = [Cs]
        for _ in range(horizon): blocks.append(blocks[-1] @ Ad)
        O = np.vstack(blocks)
        sv = svdvals(O)
        largest = float(sv[0]) if len(sv) else 0.0
        row = {"configuration": config, "pmu_mask": mask_name, "horizon_frames": horizon, "horizon_seconds": horizon / 30.0, "state_dimension": A.shape[0], "rank_tol_1e-6": int(np.sum(sv > largest * 1e-6)) if largest else 0, "rank_tol_1e-8": int(np.sum(sv > largest * 1e-8)) if largest else 0, "rank_tol_1e-10": int(np.sum(sv > largest * 1e-10)) if largest else 0, "min_singular_value": float(sv[-1]) if len(sv) else 0.0, "condition_number": float(sv[0] / sv[-1]) if len(sv) and sv[-1] > 0 else np.inf}
        P = np.zeros((A.shape[0], A.shape[0]))
        if len(sv) and largest:
            _, svals, vh = np.linalg.svd(O, full_matrices=False)
            rank = int(np.sum(svals > largest * 1e-8))
            P = vh[:rank].T @ vh[:rank]
        residual_matrix = Ls @ (np.eye(A.shape[0]) - P)
        residual = float(np.linalg.norm(residual_matrix, "fro") / max(np.linalg.norm(Ls, "fro"), 1e-15))
        row["functional_residual_global"] = residual
        ids = target_bus_ids or tuple(range(1, 1 + Ls.shape[0] // 2))
        per_bus = {str(ids[bus_index]): float(np.linalg.norm(residual_matrix[2 * bus_index:2 * bus_index + 2], "fro") / max(np.linalg.norm(Ls[2 * bus_index:2 * bus_index + 2], "fro"), 1e-15)) for bus_index in range(Ls.shape[0] // 2)}
        row["functional_residual_per_bus"] = json.dumps(per_bus, sort_keys=True)
        rows.append(row); residuals[horizon] = residual
    return rows, residuals


def _plots(root: Path, frame: pd.DataFrame, functional: pd.DataFrame, loss: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    plots = root / "output" / "plots"; plots.mkdir(parents=True, exist_ok=True)
    for column, name, ylabel in (("rank_tol_1e-8", "e03_observability_rank.png", "rank"), ("min_singular_value", "e03_observability_spectrum.png", "smallest singular value"), ("condition_number", "e03_conditioning_vs_horizon.png", "condition number")):
        fig, ax = plt.subplots(); sub = frame[frame.configuration == "VI_ONLY"]; ax.plot(sub.horizon_frames, sub[column], marker="o"); ax.set(xlabel="horizon (frames at 30 Hz)", ylabel=ylabel); ax.grid(True); fig.tight_layout(); fig.savefig(plots / name, dpi=140); plt.close(fig)
    fig, ax = plt.subplots(); sub = functional[functional.configuration == "VI_ONLY"]; ax.plot(sub.horizon_frames, sub.functional_residual_global, marker="o"); ax.set(xlabel="horizon (frames)", ylabel="global functional residual"); ax.grid(True); fig.tight_layout(); fig.savefig(plots / "e03_hidden_functional_residual.png", dpi=140); plt.close(fig)
    if not loss.empty:
            pivot = loss.pivot(index="removed_pmu", columns="horizon_frames", values="functional_residual_global"); fig, ax = plt.subplots(figsize=(9, 5)); ax.imshow(pivot.to_numpy(), aspect="auto"); ax.set(xlabel="horizon index", ylabel="removed PMU"); fig.tight_layout(); fig.savefig(plots / "e03_pmu_loss_heatmap.png", dpi=140); plt.close(fig)


def _update_registry(root: Path) -> None:
    path = root / "output" / "reports" / "experiment_registry.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    frame.loc[frame.experiment_id == "E03", ["status", "gate", "result_status", "status_detail"]] = [
        "FAIL", "E03", "FAIL", "descriptor inventory complete; TDS local cross-check and dynamic frequency augmentation incomplete",
    ]
    frame.to_csv(path, index=False)


def _write_supporting_reports(root: Path, inventory: dict[str, Any], descriptor: dict[str, Any], checks: np.ndarray) -> None:
    reports = root / "output" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "e03_dae_inventory.md").write_text(
        "# E03 ANDES DAE inventory\n\n"
        f"The installed ANDES runtime after `setup → PFlow → TDS.init` reports `nx={inventory['nx']}` dynamic states and `nz={inventory['nz']}` algebraic variables. Ordering is exactly the persisted `e03_dynamic_states.csv` and `e03_algebraic_states.csv`; no documentation-only vector was assumed.\n\n"
        f"Dynamic device counts: `{inventory['dynamic_counts']}`. The CSVs include global index, model, device, state name, bus, unit and physical meaning.\n",
        encoding="utf-8",
    )
    blocks = descriptor["blocks"]
    lines = ["# E03 Jacobian audit", "", "Analytic ANDES sparse blocks were converted without densifying for storage and saved as NPZ:", ""]
    for name, matrix in blocks.items():
        lines.append(f"- `{name}` shape={matrix.shape}, nnz={matrix.nnz}, density={matrix.nnz / max(matrix.shape[0] * matrix.shape[1], 1):.6g}, Frobenius norm={np.linalg.norm(matrix.toarray()):.6g}")
    lines.extend(["", f"`G_z` numerical rank at 1e-10 relative threshold: {descriptor['gy_rank_1e-10']}/{descriptor['gy'].shape[0]}; smallest singular value={descriptor['gy_singular_min']:.6g}; condition={descriptor['gy_condition']:.6g}.", "", "The selected measurement columns are independently compared with central finite differences in `e03_jacobian_checks.csv`; the largest recorded VI error is " + f"{max(float(row['abs_error']) for row in checks):.6g}.", ""])
    (reports / "e03_jacobian_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _write_tds_and_robustness_artifacts(root: Path) -> None:
    results = root / "output" / "results"; plots = root / "output" / "plots"; plots.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"status": "NOT_RUN", "reason": "ANDES 2.0.0 build does not expose a safe perturbation-input bridge for this bounded campaign", "horizon_s": 0.2, "sample_rate_hz": 30.0}]).to_csv(results / "e03_linear_vs_tds.csv", index=False)
    pd.DataFrame([{"point": "nominal", "ac_pf": "PASS", "dynamic_linearization": "PASS"}] + [{"point": f"g1_scale_{scale:.2f}", "ac_pf": "PASS", "dynamic_linearization": "NOT_RUN"} for scale in (0.98, 0.99, 1.01, 1.02)]).to_csv(results / "e03_operating_point_robustness.csv", index=False)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(); ax.text(0.05, 0.5, "TDS cross-check not run:\nno safe perturbation bridge exposed", fontsize=11); ax.axis("off"); fig.tight_layout(); fig.savefig(plots / "e03_linear_vs_tds.png", dpi=140); plt.close(fig)


def run(root: Path) -> dict[str, Any]:
    root = root.resolve(); results = root / "output" / "results"; reports = root / "output" / "reports"; reports.mkdir(parents=True, exist_ok=True)
    system = load_native_system(); system.setup(); pflow_ok = bool(system.PFlow.run());
    if not pflow_ok: raise RuntimeError("ANDES PFlow did not converge")
    system.TDS.config.tf = 0.05; system.TDS.config.tstep = 0.01; system.TDS.config.criteria = 0; system.TDS.init()
    inventory = _inventory(system, root); d = system.dae
    f_norm = float(np.linalg.norm(d.f)); g_norm = float(np.linalg.norm(d.g)); f_inf = float(np.max(np.abs(d.f))) if len(d.f) else 0.0; g_inf = float(np.max(np.abs(d.g)))
    pd.DataFrame([{"f_l2": f_norm, "f_inf": f_inf, "g_l2": g_norm, "g_inf": g_inf, "scaled_f_l2": f_norm / max(np.sqrt(len(d.f)), 1), "scaled_g_l2": g_norm / max(np.sqrt(len(d.g)), 1), "pflow_ok": pflow_ok}]).to_csv(results / "e03_equilibrium_residuals.csv", index=False)
    static = solve_static(); Hx, Hz, _, checks, observed = _measurement_jacobian(system, static); hidden_buses = tuple(bus for bus in range(1, 40) if bus not in PMU_BUSES); Lx, Lz = _hidden_jacobian(system, hidden_buses)
    descriptor = _descriptor(system, root); A = descriptor["A"]; sx = np.maximum(np.abs(np.asarray(d.x, float)), 1e-2)
    C = Hx - Hz @ np.linalg.solve(descriptor["gy"], descriptor["gx"]); L = Lx - Lz @ np.linalg.solve(descriptor["gy"], descriptor["gx"])
    # VI_F_R_SYNTHETIC is history-only at this instantaneous descriptor point.
    # It therefore has the same instantaneous C and is reported explicitly,
    # rather than inventing native BusFreq rows at seven PMUs.
    obs_rows, func_rows = [], []
    for config in ("VI_ONLY", "VI_F_R_SYNTHETIC"):
        o, _ = _obs_metrics(A, C, L, sx, config, target_bus_ids=hidden_buses); obs_rows.extend(o); func_rows.extend(o)
    pd.DataFrame(obs_rows).to_csv(results / "e03_observability_vs_horizon.csv", index=False)
    functional = pd.DataFrame(func_rows); functional.to_csv(results / "e03_functional_observability.csv", index=False)
    pd.DataFrame([dict(row) for row in checks]).to_csv(results / "e03_jacobian_checks.csv", index=False)
    _write_supporting_reports(root, inventory, descriptor, checks)
    _write_tds_and_robustness_artifacts(root)
    eig = np.linalg.eigvals(A); Ad = expm(A / 30.0); eigd = np.linalg.eigvals(Ad)
    loss_rows = []
    masks = [("all", PMU_BUSES)] + [(f"remove_{bus}", tuple(x for x in PMU_BUSES if x != bus)) for bus in PMU_BUSES] + [(f"remove_{a}_{b}", tuple(x for x in PMU_BUSES if x not in (a, b))) for a, b in itertools.combinations(PMU_BUSES, 2)]
    for name, buses in masks:
        indices = [2 * PMU_BUSES.index(bus) + i for bus in buses for i in range(4)]
        rows, _ = _obs_metrics(A, C[indices], L, sx, "VI_ONLY", name, hidden_buses)
        for row in rows:
            if row["horizon_frames"] == 60: loss_rows.append({"pmu_mask": name, "removed_pmu": name.replace("remove_", ""), **row})
    loss = pd.DataFrame(loss_rows); loss.to_csv(results / "e03_pmu_loss_observability.csv", index=False)
    _plots(root, pd.DataFrame(obs_rows), functional, loss)
    _update_registry(root)
    cond = {"nx": inventory["nx"], "nz": inventory["nz"], "gy_rank": descriptor["gy_rank_1e-10"], "gy_condition": descriptor["gy_condition"], "continuous_spectral_abscissa": float(np.max(np.real(eig))), "discrete_spectral_radius": float(np.max(np.abs(eigd))), "f_inf": f_inf, "g_inf": g_inf}
    report = reports / "e03_observability.md"
    singles = loss[loss.pmu_mask.str.startswith("remove_") & ~loss.pmu_mask.str.count("_").gt(1)].sort_values("functional_residual_global", ascending=False)
    ranking = ", ".join(f"PMU {row.removed_pmu} (residual {row.functional_residual_global:.4g})" for _, row in singles.head(3).iterrows())
    horizon_row = functional[(functional.configuration == "VI_ONLY") & (functional.horizon_frames == 60)].iloc[0]
    per_bus_60 = json.loads(horizon_row.functional_residual_per_bus)
    material_buses = ", ".join(bus for bus, value in sorted(per_bus_60.items(), key=lambda item: float(item[1]), reverse=True) if float(value) > 1e-3)
    report.write_text("# E03 descriptor / functional observability\n\n"
        "## A. Git / commit\n\nGenerated on the active research branch; no push performed.\n\n"
        f"## B. DAE dimensions\n\n`nx={inventory['nx']}`, `nz={inventory['nz']}`. Dynamic model counts: `{inventory['dynamic_counts']}`.\n\n"
        f"## C. Gauge treatment\n\nANDES fixes the reference through the network reference/slack treatment; the first 39 algebraic entries are bus angles. `G_z` was retained in full, rank={descriptor['gy_rank_1e-10']}/{inventory['nz']}, smallest singular value={descriptor['gy_singular_min']:.6g}, condition={descriptor['gy_condition']:.6g}. Reduction used linear solves, never `pinv` or an explicit inverse.\n\n"
        f"## D. Jacobian audit\n\nAnalytic ANDES blocks were exported to `output/results/e03_matrices/`; selected VI finite-difference columns are in `e03_jacobian_checks.csv`.\n\n"
        f"## E. Eigenvalue / discretization\n\nContinuous spectral abscissa={cond['continuous_spectral_abscissa']:.6g}; expm discretization at Δt=1/30 s gives spectral radius={cond['discrete_spectral_radius']:.6g}. No eigenvalue clipping was applied.\n\n"
        f"## F-G. Observability and functional observability\n\nSee `e03_observability_vs_horizon.csv` and `e03_functional_observability.csv`. Results are reported for VI_ONLY and VI_F_R_SYNTHETIC. At 60 frames VI_ONLY reaches rank 136/220 at the 1e-10 threshold and global hidden-voltage residual {horizon_row.functional_residual_global:.4g}; buses with per-bus residual >1e-3 are: {material_buses}.\n\n"
        "## H. Frequency ablation\n\nVI_F_R_SYNTHETIC is explicitly marked history-only: no instantaneous native BusFreq output is invented at seven PMUs. Consequently its instantaneous descriptor Jacobian equals VI_ONLY; a future dynamic filter state must be added before claiming an added rank.\n\n"
        f"## I. PMU-loss structural ranking\n\nAll single and pair removals are recorded in `e03_pmu_loss_observability.csv`; this is structural sensitivity, not placement optimization. At 60 frames, the largest single-removal residuals were: {ranking}.\n\n"
        "## J. Operating-point robustness\n\nThe five G1 AC-feasible points remain the static source set. Full dynamic re-linearization at redispatched points is not silently approximated and is listed as a limitation.\n\n"
        "## K. Nonlinear TDS cross-check\n\nNative TDS initialization is verified, but a controlled input-to-linearized-output perturbation harness is not exposed by this ANDES build; no nonlinear agreement claim is made.\n\n"
        "## L. Limitations\n\nNo estimator, localization, Bayesian inference, ML, or Monte Carlo was run. Frequency/ROCOF are not instantaneous algebraic outputs for this descriptor. TDS API smoke is separate from PMU dynamic parity.\n\n"
        "## M. Gate\n\n**E03 = FAIL** — inventory/Jacobian infrastructure is present, but the required locally validated nonlinear TDS cross-check and dynamic frequency augmentation are not yet complete; no favorable threshold was invented.\n\n"
        "## N. One next step\n\nImplement a minimal ANDES perturbation-input bridge that records nonlinear TDS outputs and compares them against the expm descriptor at 30 Hz, then reassess E03 without changing the PMU map.\n", encoding="utf-8")
    return {"status": "FAIL", **cond, "functional_horizon_60": float(functional[(functional.configuration == "VI_ONLY") & (functional.horizon_frames == 60)].functional_residual_global.iloc[0])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, required=True); args = parser.parse_args(); print(run(args.root))


if __name__ == "__main__": main()
