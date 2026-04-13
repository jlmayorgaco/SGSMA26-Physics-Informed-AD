"""Generate one reproducible simulation bundle for each visible event label."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.augmentation.andes_sim import (
    _write_synthetic_csvs,
    extract_normal_baseline,
    generate_bad_data,
    generate_fault,
    generate_gen_change,
    generate_line_outage,
    generate_load_change,
    generate_pmu_dropout,
    generate_post_cyber_physical,
    load_synthetic,
)
from src.io.load_csv import PMU_BUSES, load_all
from src.localizer.cosine_match import locate
from src.localizer.physics_scores import ybus_residual_by_bus


log = logging.getLogger(__name__)

LABEL_NAMES = {
    1: "fault",
    2: "line_outage",
    3: "generation_change",
    4: "load_change",
    5: "missing_data",
    6: "missing_data_plus_physical",
    7: "bad_data",
}

TARGETS = {
    1: {"target_bus": 7, "target_line": None},
    2: {"target_bus": None, "target_line": [24, 23]},
    3: {"target_bus": 2, "target_line": None},
    4: {"target_bus": 7, "target_line": None},
    5: {"target_bus": 29, "target_line": None},
    6: {"target_bus": 29, "physical_bus": 2, "target_line": None},
    7: {"target_bus": 29, "target_line": None},
}


def _fallback_stats() -> dict:
    stats: dict = {}
    for bus in PMU_BUSES:
        for phase, angle in (("A", -5.0), ("B", -125.0), ("C", 115.0)):
            stats[f"BUS{bus}_V{phase}_MAG"] = (200_000.0, 500.0)
            stats[f"BUS{bus}_V{phase}_ANG"] = (angle, 0.1)
            stats[f"BUS{bus}_I{phase}_MAG"] = (585.0, 2.0)
            stats[f"BUS{bus}_I{phase}_ANG"] = (angle - 25.0, 0.5)
        stats[f"BUS{bus}_Freq"] = (60.0, 0.005)
        stats[f"BUS{bus}_ROCOF"] = (0.0, 0.002)
    return stats


def _load_stats(data_dir: Path) -> dict:
    try:
        df = load_all(data_dir)
        return extract_normal_baseline(df)
    except Exception as exc:
        log.warning("Using fallback simulation baseline stats: %s", exc)
        return _fallback_stats()


def _load_grid(raw_path: Path):
    try:
        from src.grid.jacobians import bus_sensitivity_columns, compute_jacobians
        from src.grid.load_case import load_case

        grid = load_case(raw_path)
        J_cols = bus_sensitivity_columns(compute_jacobians(grid), grid)
        return grid, J_cols, grid.branch_list
    except Exception as exc:
        log.warning("Simulation localization will omit grid score table: %s", exc)
        return None, {}, []


def _simulate_label(label: int, stats: dict, rng: np.random.Generator):
    if label == 1:
        return generate_fault(
            event_bus=7,
            stats=stats,
            rng=rng,
            t_event=2.0,
            fault_impedance=0.01,
            clear_cycles=6,
            delta_p_pu=0.22,
        )
    if label == 2:
        return generate_line_outage(
            line_from=24,
            line_to=23,
            stats=stats,
            rng=rng,
            t_event=2.0,
            delta_p_pu=0.12,
        )
    if label == 3:
        return generate_gen_change(
            gen_bus=2,
            stats=stats,
            rng=rng,
            t_event=2.0,
            delta_mw=-45.0,
        )
    if label == 4:
        return generate_load_change(
            load_bus=7,
            stats=stats,
            rng=rng,
            t_event=2.0,
            delta_mw=55.0,
        )
    if label == 5:
        return generate_pmu_dropout(
            dropout_bus=29,
            stats=stats,
            rng=rng,
            t_event=2.0,
            dropout_sec=2.2,
        )
    if label == 6:
        return generate_post_cyber_physical(
            stats=stats,
            rng=rng,
            physical="gen",
            dropout_bus=29,
            t_dropout=1.0,
            t_physical=2.2,
            t_recover=3.6,
        )
    if label == 7:
        return generate_bad_data(
            bad_bus=29,
            stats=stats,
            rng=rng,
            t_event=2.0,
            duration_sec=1.2,
        )
    raise ValueError(f"Unsupported simulation label {label}")


def _plot_pmu_overview(df: pd.DataFrame, label: int, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    t = df["TIMESTAMP"].to_numpy(dtype=float)
    for bus in PMU_BUSES:
        vcol = f"BUS{bus}_VA_MAG"
        icol = f"BUS{bus}_IA_MAG"
        fcol = f"BUS{bus}_Freq"
        if vcol in df:
            axes[0].plot(t, df[vcol].to_numpy(dtype=float), lw=1.0, label=f"Bus {bus}")
        if icol in df:
            axes[1].plot(t, df[icol].to_numpy(dtype=float), lw=1.0)
        if fcol in df:
            axes[2].plot(t, df[fcol].to_numpy(dtype=float), lw=1.0)
    active = df["Event"].to_numpy(dtype=int) == int(label)
    if np.any(active):
        t0 = float(t[np.argmax(active)])
        t1 = float(t[len(active) - 1 - np.argmax(active[::-1])])
        for ax in axes:
            ax.axvspan(t0, t1, color="#d9480f", alpha=0.14)
    axes[0].set_title(f"Label {label}: {LABEL_NAMES[label]}")
    axes[0].set_ylabel("VA magnitude [V]")
    axes[1].set_ylabel("IA magnitude [A]")
    axes[2].set_ylabel("Frequency [Hz]")
    axes[2].set_xlabel("Time [s]")
    axes[0].legend(ncol=4, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_candidate_scores(loc_result: dict, out_path: Path) -> None:
    rows = loc_result.get("score_table", [])[:10]
    if not rows:
        rows = [
            {"bus": b, "score": s, "jacobian": s, "ybus": 0.0, "zbus": 0.0, "pmu_direct": 0.0}
            for b, s in loc_result.get("top3_buses", [])
        ]
    labels = [str(row["bus"]) for row in rows]
    score = [float(row.get("score", 0.0)) for row in rows]
    ybus = [float(row.get("ybus", 0.0)) for row in rows]
    zbus = [float(row.get("zbus", 0.0)) for row in rows]
    swing = [float(row.get("swing", 0.0)) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(labels))
    ax.bar(x - 0.27, score, width=0.18, label="combined")
    ax.bar(x - 0.09, ybus, width=0.18, label="Ybus")
    ax.bar(x + 0.09, zbus, width=0.18, label="Zbus")
    ax.bar(x + 0.27, swing, width=0.18, label="swing")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Candidate bus")
    ax.set_ylabel("normalized score")
    ax.set_title("Localization physics score table")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_ybus_residuals(
    df: pd.DataFrame,
    grid,
    onset: int,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    if grid is None:
        ax.text(0.5, 0.5, "Ybus metadata unavailable", ha="center", va="center")
        ax.set_xticks([])
    else:
        energy = ybus_residual_by_bus(df, grid, onset)
        rows = sorted(energy.items(), key=lambda item: -item[1])[:15]
        labels = [str(bus) for bus, _ in rows]
        values = [float(value) for _, value in rows]
        ax.bar(labels, values, color="#4c78a8")
        ax.set_xlabel("Candidate bus")
        ax.set_ylabel("Ybus residual energy")
        ax.grid(axis="y", alpha=0.25)
    ax.set_title("Reconstructed Ybus/KCL residual evidence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _spectral_layout(grid) -> dict[int, tuple[float, float]]:
    buses = list(grid.ext_bus_order)
    idx = {bus: i for i, bus in enumerate(buses)}
    n = len(buses)
    A = np.zeros((n, n), dtype=float)
    for a, b in grid.branch_list:
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            A[i, j] = A[j, i] = 1.0
    L = np.diag(A.sum(axis=1)) - A
    try:
        vals, vecs = np.linalg.eigh(L)
        order = np.argsort(vals)
        xy = vecs[:, order[1:3]]
        xy = xy / np.maximum(np.max(np.abs(xy), axis=0), 1e-12)
    except Exception:
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xy = np.c_[np.cos(theta), np.sin(theta)]
    return {bus: (float(xy[i, 0]), float(xy[i, 1])) for bus, i in idx.items()}


def _plot_topology_candidates(
    grid,
    loc_result: dict,
    out_path: Path,
    *,
    target_buses: set[int] | None = None,
    target_line: list[int] | tuple[int, int] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    target_buses = target_buses or set()
    target_line_set = set(target_line) if target_line is not None else set()
    if grid is None:
        ax.text(0.5, 0.5, "Topology metadata unavailable", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        coords = _spectral_layout(grid)
        score_by_bus = {
            int(row["bus"]): float(row.get("score", 0.0))
            for row in loc_result.get("score_table", [])
        }
        if not score_by_bus:
            score_by_bus = {int(bus): float(score) for bus, score in loc_result.get("top3_buses", [])}
        max_score = max(score_by_bus.values()) if score_by_bus else 0.0
        top_line = loc_result.get("top1_line")
        for a, b in grid.branch_list:
            if a not in coords or b not in coords:
                continue
            xa, ya = coords[a]
            xb, yb = coords[b]
            is_top_line = top_line is not None and set(top_line) == {a, b}
            is_target_line = bool(target_line_set) and target_line_set == {a, b}
            color = "#b8b8b8"
            if is_top_line:
                color = "#4c78a8"
            if is_target_line:
                color = "#d9480f"
            ax.plot(
                [xa, xb],
                [ya, yb],
                color=color,
                lw=3.2 if is_top_line or is_target_line else 1.0,
                zorder=1,
            )
        for bus, (x, y) in coords.items():
            is_pmu = bus in PMU_BUSES
            is_target_bus = bus in target_buses
            raw_score = score_by_bus.get(bus, 0.0)
            scaled = raw_score / max(max_score, 1e-12)
            color = "#d9480f" if is_target_bus else (
                "#4c78a8" if raw_score > 0 else ("#8aa6c1" if is_pmu else "#d0d0d0")
            )
            size = 70 + 260 * scaled
            if is_target_bus:
                size = max(size, 240)
            ax.scatter(
                [x],
                [y],
                s=size,
                marker="s" if is_pmu else "o",
                color=color,
                edgecolor="black",
                linewidth=0.7,
                zorder=3,
            )
            ax.text(x, y, str(bus), ha="center", va="center", fontsize=8, zorder=4)
        ax.set_axis_off()
    ax.set_title("Topology and candidate-score evidence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_line_candidates(loc_result: dict, out_path: Path) -> None:
    rows = loc_result.get("top3_lines", [])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if rows:
        labels = [f"{a}-{b}" for (a, b), _score in rows]
        scores = [float(score) for _line, score in rows]
        ax.bar(labels, scores, color="#4c78a8")
    else:
        ax.text(0.5, 0.5, "No line candidates for this label", ha="center", va="center")
        ax.set_xticks([])
    ax.set_ylabel("score")
    ax.set_title("Line candidate evidence")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def generate_simulations(
    *,
    data_dir: Path,
    raw_path: Path,
    out_dir: Path,
    labels: list[int],
    seed: int = 42,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _load_stats(data_dir)
    grid, J_cols, branches = _load_grid(raw_path)
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for label in labels:
        name = LABEL_NAMES[int(label)]
        target = TARGETS[int(label)]
        target_buses = {
            int(value)
            for key, value in target.items()
            if key.endswith("bus") and value is not None
        }
        scenario_dir = out_dir / f"label_{label:02d}_{name}"
        csv_dir = scenario_dir / "csv"
        fig_dir = scenario_dir / "figures"
        csv_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)

        bus_data, ts, event_labels = _simulate_label(int(label), stats, rng)
        _write_synthetic_csvs(0, bus_data, ts, event_labels, csv_dir)
        df = load_synthetic(csv_dir, 0)
        frames = np.where(df["Event"].to_numpy(dtype=int) == int(label))[0]
        if len(frames) == 0:
            frames = np.where(df["Event"].to_numpy(dtype=int) != 0)[0]
        onset = int(frames[0]) if len(frames) else 0

        loc_result = locate(
            df,
            onset,
            int(label),
            J_cols,
            branches,
            grid=grid,
        )
        _plot_pmu_overview(df, int(label), fig_dir / "pmu_overview.png")
        _plot_ybus_residuals(df, grid, onset, fig_dir / "ybus_residuals.png")
        _plot_candidate_scores(loc_result, fig_dir / "candidate_scores.png")
        _plot_topology_candidates(
            grid,
            loc_result,
            fig_dir / "topology_candidates.png",
            target_buses=target_buses,
            target_line=target.get("target_line"),
        )
        _plot_line_candidates(loc_result, fig_dir / "line_candidates.png")

        metadata = {
            "label": int(label),
            "label_name": name,
            "target": target,
            "onset_frame": onset,
            "onset_time_sec": float(df["TIMESTAMP"].iloc[onset]),
            "pmu_buses": list(PMU_BUSES),
            "top1_bus": loc_result.get("top1_bus"),
            "top3_buses": [[int(b), float(s)] for b, s in loc_result.get("top3_buses", [])],
            "top1_line": loc_result.get("top1_line"),
            "top3_lines": [
                {"line": [int(a), int(b)], "score": float(score)}
                for (a, b), score in loc_result.get("top3_lines", [])
            ],
            "fault_subtype": loc_result.get("fault_subtype"),
            "score_table": loc_result.get("score_table", [])[:10],
            "figures": [
                "figures/pmu_overview.png",
                "figures/ybus_residuals.png",
                "figures/candidate_scores.png",
                "figures/topology_candidates.png",
                "figures/line_candidates.png",
            ],
        }
        (scenario_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = [
            f"# Label {label}: {name}",
            "",
            f"- Onset: frame {onset}, t={metadata['onset_time_sec']:.3f} s.",
            f"- Top-1 bus: {metadata['top1_bus']}.",
            f"- Top-3 buses: {metadata['top3_buses']}.",
            f"- Top-1 line: {metadata['top1_line']}.",
            "- Input evidence: only the eight PMU streams plus IEEE-39 topology.",
            "",
            "Figures:",
            "- `figures/pmu_overview.png`",
            "- `figures/ybus_residuals.png`",
            "- `figures/candidate_scores.png`",
            "- `figures/topology_candidates.png`",
            "- `figures/line_candidates.png`",
            "",
        ]
        (scenario_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
        records.append({"path": str(scenario_dir), **metadata})

    manifest = {
        "seed": int(seed),
        "labels": labels,
        "n_simulations": len(records),
        "simulations": records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return records


def _parse_labels(text: str) -> list[int]:
    labels = [int(part.strip()) for part in text.split(",") if part.strip()]
    bad = [label for label in labels if label not in LABEL_NAMES]
    if bad:
        raise ValueError(f"Unsupported labels: {bad}")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--raw", type=Path, default=Path("data/metadata/IEEE 39 Bus Power System.raw"))
    parser.add_argument("--out", type=Path, default=Path("experiments/event_simulations"))
    parser.add_argument("--labels", default="1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    records = generate_simulations(
        data_dir=args.data,
        raw_path=args.raw,
        out_dir=args.out,
        labels=_parse_labels(args.labels),
        seed=args.seed,
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
