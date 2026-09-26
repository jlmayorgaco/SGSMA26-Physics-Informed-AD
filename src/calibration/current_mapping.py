"""Current mapping candidate selection for m3 raw calibration."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration.calibration_fit import fit_affine_distribution
from src.calibration.calibration_metrics import compute_basic_metrics
from src.calibration.raw_chunk_loader import raw_col, raw_values_from_chunks


def build_current_candidates_for_bus(bus_id, bus_injection, branch_currents, generator_currents):
    candidates = {}
    if bus_id in bus_injection:
        candidates["bus_injection_current"] = bus_injection[bus_id]
    for name, arr in branch_currents.get(bus_id, []):
        candidates[f"branch_current:{name}"] = arr
    if branch_currents.get(bus_id):
        name, arr = max(branch_currents[bus_id], key=lambda item: float(np.mean(item[1])))
        candidates[f"dominant_incident_branch_current:{name}"] = arr
        p_name, p_arr = max(branch_currents[bus_id], key=lambda item: float(np.mean(np.square(item[1]))))
        candidates[f"max_power_branch_current:{p_name}"] = p_arr
    for name, arr in generator_currents.get(bus_id, []):
        candidates[f"generator_terminal_current:{name}"] = arr
    return candidates


def choose_best_current_mapping_for_bus(bus_id, chunks, traj):
    real_phases = []
    for ph in ["A", "B", "C"]:
        vals = raw_values_from_chunks(chunks, bus_id, f"I{ph}_MAG")
        if len(vals):
            real_phases.append(vals)
    if not real_phases:
        return None, []
    real = np.concatenate(real_phases)

    rows = []
    for name, candidate in traj.get("current_candidates", {}).items():
        if candidate is None or len(candidate) == 0:
            continue
        a, b, fit_mode = fit_affine_distribution(candidate, real, "current_mag")
        prelim = a * np.asarray(candidate, dtype=float) + b
        per_chunk_ks = []
        for chunk in chunks:
            phase_vals = []
            for ph in ["A", "B", "C"]:
                col = raw_col(bus_id, f"I{ph}_MAG")
                if col in chunk["df"].columns:
                    vals = pd.to_numeric(chunk["df"][col], errors="coerce").dropna().to_numpy(float)
                    if len(vals):
                        phase_vals.append(vals)
            if phase_vals:
                from scipy.stats import ks_2samp

                per_chunk_ks.append(float(ks_2samp(np.concatenate(phase_vals), prelim).statistic))
        stability = float(np.std(per_chunk_ks)) if len(per_chunk_ks) > 1 else 0.0
        metrics = compute_basic_metrics(real, prelim, "current_mag", fs=30.0, chunk_penalty=stability)
        rows.append(
            {
                "bus_id": bus_id,
                "candidate": name,
                "fit_mode": fit_mode,
                "fit_a": a,
                "fit_b": b,
                "ks_stat": metrics["ks_stat"],
                "relative_mean_error": metrics["relative_mean_error"],
                "relative_std_error": metrics["relative_std_error"],
                "percentile_error": metrics["percentile_error"],
                "psd_lowfreq_mismatch": metrics["psd_lowfreq_mismatch"],
                "spectral_centroid_error": metrics["spectral_centroid_error"],
                "dominant_freq_error": metrics["dominant_freq_error"],
                "chunk_stability_penalty": stability,
                "composite_score": metrics["composite_score"],
            }
        )
    if not rows:
        return None, []
    rows = sorted(rows, key=lambda r: r["composite_score"])
    return rows[0]["candidate"], rows


def plot_current_mapping_diagnostics(rows, output_dir):
    if not rows:
        return
    out_dir = Path(output_dir)
    df = pd.DataFrame(rows)
    for bus_id, grp in df.groupby("bus_id"):
        grp = grp.sort_values("composite_score").head(12)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        labels = [str(x).replace("branch_current:", "br:").replace("dominant_incident_", "dom_") for x in grp["candidate"]]
        ax.barh(labels[::-1], grp["composite_score"].to_numpy()[::-1])
        ax.set_xlabel("composite score")
        ax.set_title(f"Bus {bus_id} current mapping candidates")
        ax.grid(axis="x", alpha=0.3)
        plot_path = out_dir / "plots" / "current_mapping" / "by_bus" / f"Bus{bus_id}.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)


def select_current_mappings(all_chunks, trajectories, output_dir):
    selection_rows = []
    diagnostics_rows = []
    chosen = {}
    for bus_id in sorted(all_chunks.keys(), key=lambda x: int(x)):
        best, rows = choose_best_current_mapping_for_bus(bus_id, all_chunks.get(bus_id, []), trajectories.get(bus_id, {}))
        if best is not None:
            chosen[bus_id] = best
        diagnostics_rows.extend(rows)
        selection_rows.append(
            {
                "bus_id": bus_id,
                "chosen_mapping": best or "unsupported_mapping",
                "score": rows[0]["composite_score"] if rows else np.nan,
                "candidate_scores_json": json.dumps(rows),
                "notes": "Lowest composite score from KS, std error, quantiles, spectral mismatch, and chunk stability.",
            }
        )

    out_dir = Path(output_dir)
    sel_path = out_dir / "current_mapping_selection.csv"
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(sel_path, index=False)
    diag_path = out_dir / "metrics" / "current_mapping_diagnostics.csv"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics_rows).to_csv(diag_path, index=False)
    plot_current_mapping_diagnostics(diagnostics_rows, out_dir)
    return chosen
