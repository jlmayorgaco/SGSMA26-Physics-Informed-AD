"""Build the evidence figures used by the revised six-page manuscript.

Every plotted count and score is read from the candidate audit or the completed
non-anticipative benchmark.  The script deliberately avoids illustrative performance
values so that the PDF can be regenerated from repository evidence alone.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "paper" / "evidence" / "revised"
BENCHMARK = EVIDENCE / "causal_benchmark" / "target_complete" / "target_complete"

NAVY = "#184E77"
BLUE = "#2A6F97"
TEAL = "#168AAD"
GREEN = "#2A9D8F"
ORANGE = "#E76F51"
RED = "#B23A48"
PURPLE = "#6D597A"
GRAY = "#6B7280"
LIGHT = "#E9ECEF"
DARK = "#20242A"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 180,
        "savefig.dpi": 320,
        "savefig.bbox": "tight",
    }
)


def _save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", format="pdf", pad_inches=0.025)
    fig.savefig(OUT / f"{stem}.png", format="png", pad_inches=0.025)
    plt.close(fig)


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str = "white",
    edge: str = DARK,
    fontsize: float = 7.5,
    linewidth: float = 0.8,
    pad: float = 0.012,
) -> None:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad={pad},rounding_size=0.015",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GRAY,
    linewidth: float = 0.75,
) -> None:
    """Draw a short directed connector in a lane reserved outside boxes."""

    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=linewidth, color=color, shrinkA=0, shrinkB=0),
        zorder=4,
    )


def build_system_figure() -> None:
    """Sparse-grid setting, evidence construction, and hierarchical diagnosis."""

    fig = plt.figure(figsize=(7.16, 2.58))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.38, 1.20, 1.08], wspace=0.25)

    # Panel (a): the actual IEEE 39-bus drawing used in the conference slides.
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("(a) Sparse IEEE-39 observation", loc="left", fontweight="bold", pad=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    raw_pos = {
        1:(0.0,2.1), 2:(1.1,2.15), 3:(2.1,1.85), 4:(3.0,1.55), 5:(4.0,1.55),
        6:(5.0,1.35), 7:(6.0,1.45), 8:(6.8,1.95), 9:(7.3,2.65), 39:(0.0,3.05),
        10:(5.35,0.25), 11:(4.55,0.55), 12:(3.70,0.20), 13:(3.25,0.80),
        14:(3.00,0.25), 15:(2.30,-0.25), 16:(1.55,-0.55), 17:(0.85,-0.05),
        18:(1.25,0.80), 19:(0.75,-1.05), 20:(0.20,-1.65), 21:(2.15,-1.15),
        22:(2.85,-1.55), 23:(3.65,-1.35), 24:(2.65,-0.85), 25:(-0.15,1.15),
        26:(-0.95,0.35), 27:(0.15,0.00), 28:(-1.55,1.00), 29:(-2.35,0.35),
        30:(1.35,2.85), 31:(5.75,2.05), 32:(6.10,0.00), 33:(0.20,-2.35),
        34:(0.95,-2.15), 35:(3.05,-2.35), 36:(4.35,-2.00), 37:(-0.85,1.65),
        38:(-2.95,1.05),
    }
    edges = [
        (1,2),(1,39),(2,3),(2,25),(2,30),(3,4),(3,18),(4,5),(4,14),(5,6),
        (5,8),(6,7),(6,11),(6,31),(7,8),(8,9),(9,39),(10,11),(10,13),(10,32),
        (11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(16,19),(16,21),(16,24),
        (17,18),(17,27),(19,20),(19,33),(20,34),(21,22),(22,23),(22,35),(23,24),
        (23,36),(25,26),(25,37),(26,27),(26,28),(26,29),(28,29),(29,38),
    ]
    xs = np.array([p[0] for p in raw_pos.values()])
    ys = np.array([p[1] for p in raw_pos.values()])
    pos = {
        bus: (0.05 + 0.90 * (x - xs.min()) / (xs.max() - xs.min()),
              0.27 + 0.67 * (y - ys.min()) / (ys.max() - ys.min()))
        for bus, (x, y) in raw_pos.items()
    }
    pmus = {2, 5, 6, 10, 19, 22, 29, 39}
    generators = set(range(30, 39))
    for u, v in edges:
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#AAB4BF", lw=0.45, zorder=1)
    for bus, (x, y) in pos.items():
        if bus in pmus:
            face, edge, size, weight = "#DCEFF7", BLUE, 62, "bold"
        elif bus in generators:
            face, edge, size, weight = "#FFF0E6", ORANGE, 42, "normal"
        else:
            face, edge, size, weight = "white", "#AAB4BF", 32, "normal"
        ax.scatter([x], [y], s=size, facecolor=face, edgecolor=edge, linewidth=0.75, zorder=2)
        ax.text(x, y, str(bus), ha="center", va="center", fontsize=4.3, fontweight=weight, zorder=3)
    ax.text(0.50, 0.215, "8 PMUs / 39 buses", ha="center", color=BLUE, fontweight="bold", fontsize=6.4)
    stages = [(0.04, "branches"), (0.29, "$Y_{bus}$"), (0.53, "$Z_{bus}$"), (0.79, "$d^{eff}_{ij}$")]
    for x, label in stages:
        _box(ax, (x, 0.035), 0.17, 0.105, label, face="#F5F8FA", edge=NAVY, fontsize=5.5)
    for x0, x1 in ((0.21,0.29),(0.46,0.53),(0.70,0.79)):
        _arrow(ax, (x0, 0.087), (x1, 0.087), linewidth=0.65)

    # Panel (b): the four views repeatedly developed in the slides.
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("(b) Physics-aware evidence", loc="left", fontweight="bold", pad=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.27, 0.84), 0.46, 0.10, "multi-PMU window", face="#EAF2F8", edge=BLUE, fontsize=6.5)
    views = [
        ((0.02,0.54), "Local reference\nmedian/MAD + persistence", "#E8F5F1", GREEN),
        ((0.56,0.54), "Multiscale dynamics\nrolling 5--120", "#FFF5E9", ORANGE),
        ((0.02,0.25), "Adaptive departures\nRLS + innovation", "#EEF3F8", NAVY),
        ((0.56,0.25), "Cross-PMU contrasts\nseverity + timing", "#F3EEF7", PURPLE),
    ]
    for (xy, label, face, edge) in views:
        _box(ax, xy, 0.42, 0.18, label, face=face, edge=edge, fontsize=5.35)

    # A central trunk and two horizontal rails give every view its own connector.
    ax.plot([0.50, 0.50], [0.84, 0.47], color=GRAY, lw=0.60, zorder=1)
    for rail_y, box_top in ((0.78, 0.72), (0.49, 0.43)):
        ax.plot([0.23, 0.77], [rail_y, rail_y], color=GRAY, lw=0.60, zorder=1)
        _arrow(ax, (0.23, rail_y), (0.23, box_top), linewidth=0.60)
        _arrow(ax, (0.77, rail_y), (0.77, box_top), linewidth=0.60)
    _box(ax, (0.08,0.02), 0.84, 0.105, "45,162 archived  |  136 past-only", face="white", edge="#AAB4BF", fontsize=5.35)

    # Panel (c): measurement integrity first, then physical event and typed origin.
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("(c) Hierarchical diagnosis", loc="left", fontweight="bold", pad=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, (0.25,0.85), 0.50, 0.10, "PMU evidence", face="#EAF2F8", edge=BLUE, fontsize=6.3)
    _box(ax, (0.02,0.60), 0.43, 0.14, "integrity gate\nmissing / bad data", face="#E8F5F1", edge=GREEN, fontsize=5.35)
    _box(ax, (0.55,0.60), 0.43, 0.14, "physical event\nExtraTrees", face="#FFF0E6", edge=ORANGE, fontsize=5.35)
    _box(ax, (0.28,0.39), 0.44, 0.09, "event 0--8", face="white", edge=NAVY, fontsize=6.1)
    _box(ax, (0.28,0.23), 0.44, 0.09, "typed route", face="#F3EEF7", edge=PURPLE, fontsize=6.0)
    for x, label, edge in ((0.02,"BUS",BLUE),(0.355,"LINE",ORANGE),(0.69,"PMU",GREEN)):
        _box(ax, (x,0.02), 0.29, 0.09, label, face="white", edge=edge, fontsize=5.8)

    # Orthogonal split and merge rails prevent the hierarchy arrows from crossing.
    ax.plot([0.50, 0.50], [0.85, 0.79], color=GRAY, lw=0.70, zorder=1)
    ax.plot([0.235, 0.765], [0.79, 0.79], color=GRAY, lw=0.70, zorder=1)
    _arrow(ax, (0.235, 0.79), (0.235, 0.74), linewidth=0.70)
    _arrow(ax, (0.765, 0.79), (0.765, 0.74), linewidth=0.70)
    ax.plot([0.235, 0.235], [0.60, 0.55], color=GRAY, lw=0.70, zorder=1)
    ax.plot([0.765, 0.765], [0.60, 0.55], color=GRAY, lw=0.70, zorder=1)
    ax.plot([0.235, 0.765], [0.55, 0.55], color=GRAY, lw=0.70, zorder=1)
    _arrow(ax, (0.50, 0.55), (0.50, 0.48), linewidth=0.70)
    _arrow(ax, (0.50, 0.39), (0.50, 0.32), linewidth=0.70)
    ax.plot([0.50, 0.50], [0.23, 0.17], color=GRAY, lw=0.65, zorder=1)
    ax.plot([0.165, 0.835], [0.17, 0.17], color=GRAY, lw=0.65, zorder=1)
    for x in (0.165,0.50,0.835):
        _arrow(ax, (x, 0.17), (x, 0.11), linewidth=0.65)

    _save(fig, "fig1_system_reconstruction")


def build_protocol_figure() -> None:
    """Non-anticipative benchmark, split, baselines, and task-level outputs."""

    fig, ax = plt.subplots(figsize=(7.16, 2.18))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Five stages with generous outer margins and dedicated connector gaps.
    stage_centers = (0.105, 0.305, 0.520, 0.745, 0.925)
    stage_names = ("Scenario bank", "Grouped split", "Causal view", "Matched models", "Test outputs")
    stage_colors = (BLUE, GREEN, ORANGE, PURPLE, DARK)
    for number, (x, name, color) in enumerate(zip(stage_centers, stage_names, stage_colors), start=1):
        ax.text(x, 0.805, f"{number}. {name}", ha="center", va="center", color=color, fontsize=7.0, fontweight="bold")

    _box(
        ax,
        (0.025, 0.43),
        0.16,
        0.29,
        "138 target keys\n$\\times$ 5 replicas\n690 simulations",
        face="#EAF2F8",
        edge=BLUE,
        fontsize=7.5,
    )
    _box(
        ax,
        (0.225, 0.43),
        0.16,
        0.29,
        "whole scenarios\n414 train\n138 validation\n138 test",
        face="#E8F5F1",
        edge=GREEN,
        fontsize=7.25,
    )
    _box(
        ax,
        (0.425, 0.43),
        0.19,
        0.29,
        "3-s trailing window\n136 past-only variables\none output per row",
        face="#FFF0E6",
        edge=ORANGE,
        fontsize=7.35,
    )

    # The model family is one pipeline stage, not three competing flow lines.
    model_group = patches.FancyBboxPatch(
        (0.655, 0.41),
        0.18,
        0.33,
        boxstyle="round,pad=0.010,rounding_size=0.014",
        facecolor="#FAFAFB",
        edgecolor="#AAB4BF",
        linewidth=0.75,
        zorder=1,
    )
    ax.add_patch(model_group)
    ax.text(0.745, 0.705, "equal 360-tree budget", ha="center", va="center", color=GRAY, fontsize=5.7)
    # Keep each alternative on one line.  At IEEE column scale, two-line labels
    # were taller than their rows and visually collided with adjacent boxes.
    for y, label, edge in (
        (0.585, "flat | 3 heads", GRAY),
        (0.500, "typed | 12 heads", BLUE),
        (0.415, "typed+topology | 4 heads", PURPLE),
    ):
        _box(ax, (0.670, y), 0.15, 0.060, label, face="white", edge=edge, fontsize=5.05, linewidth=0.75, pad=0.004)

    _box(
        ax,
        (0.875, 0.43),
        0.10,
        0.29,
        "event 0--8\nphysical Top-3\nintegrity Top-3\nFAR + delay",
        face="#F4F4F5",
        edge=DARK,
        fontsize=6.0,
    )

    # Arrowheads end in whitespace rather than sitting on a node border.
    for start, end in ((0.195, 0.215), (0.395, 0.415), (0.625, 0.645), (0.845, 0.865)):
        _arrow(ax, (start, 0.575), (end, 0.575), color=DARK, linewidth=0.95)

    ax.text(0.50, 0.965, "Non-anticipative evaluation: complete test scenarios remain held out", ha="center", color=NAVY, fontweight="bold", fontsize=8.4)

    # The lower strip exposes the protocol controls that protect each transition.
    ax.text(0.025, 0.305, "Protocol controls", ha="left", va="center", color=GRAY, fontsize=6.2, fontweight="bold")
    controls = (
        (0.025, 0.16, "target-complete\n138 keys per split", BLUE),
        (0.225, 0.16, "scenario-disjoint\nno trajectory overlap", GREEN),
        (0.425, 0.19, "$X_{\\leq t}$ only\nfuture-change invariant", ORANGE),
        (0.655, 0.18, "same seeds + tree budget\nthresholds from validation", PURPLE),
        (0.875, 0.10, "paired blocks\n5,000 bootstrap", DARK),
    )
    for x, width, label, edge in controls:
        _box(ax, (x, 0.12), width, 0.12, label, face="white", edge=edge, fontsize=5.45, linewidth=0.65)

    _save(fig, "fig2_causal_protocol")


def _pooled_confusion(model: str) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for seed in (11, 29, 47):
        path = BENCHMARK / "primary" / f"confusion_{model}_seed{seed}.csv"
        frame = pd.read_csv(path)
        arrays.append(frame.iloc[:, 1:].to_numpy(dtype=int))
    return np.sum(arrays, axis=0)


def build_results_figure() -> None:
    """Main trade-offs plus the pooled typed confusion matrix."""

    summary = pd.read_csv(BENCHMARK / "all_metrics_summary.csv").set_index("model")
    scenario_summary = pd.read_csv(EVIDENCE / "scenario_localization_summary.csv").set_index(
        ["model", "target_type"]
    )
    models = ["flat", "typed", "typed_topology"]
    labels = ["Flat", "Typed", "Typed + topology"]
    colors = [GRAY, BLUE, PURPLE]

    fig = plt.figure(figsize=(7.16, 2.52))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.10, 1.18, 1.52], wspace=0.40, left=0.065, right=0.99, bottom=0.22, top=0.85)

    ax = fig.add_subplot(gs[0, 0])
    measures = [
        ("Det. precision", "detection_precision_mean", "detection_precision_std"),
        ("Det. F1", "detection_f1_mean", "detection_f1_std"),
        ("Event macro-F1", "event_macro_f1_all_labels_mean", "event_macro_f1_all_labels_std"),
    ]
    x = np.arange(len(measures))
    width = 0.23
    for pos, (model, label, color) in enumerate(zip(models, labels, colors)):
        values = [summary.loc[model, key] for _, key, _ in measures]
        errors = [summary.loc[model, err] for _, _, err in measures]
        ax.bar(x + (pos - 1) * width, values, width, yerr=errors, color=color, label=label, capsize=2, linewidth=0.4, edgecolor=DARK)
    ax.set_ylim(0.55, 1.01)
    ax.set_xticks(x, [item[0] for item in measures], rotation=25, ha="right")
    ax.set_ylabel("score")
    ax.set_title("(a) Detection and event ID", loc="left", fontweight="bold")
    ax.grid(axis="y", color=LIGHT, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    localization = ["Phys. row\nTop-1", "Phys. scen.\nTop-1", "Integ. row\nTop-1"]
    x = np.arange(len(localization))
    for pos, (model, label, color) in enumerate(zip(models, labels, colors)):
        values = [
            summary.loc[model, "physical_top1_mean"],
            scenario_summary.loc[(model, "physical"), "scenario_top1_mean"],
            summary.loc[model, "integrity_top1_mean"],
        ]
        errors = [
            summary.loc[model, "physical_top1_std"],
            scenario_summary.loc[(model, "physical"), "scenario_top1_std"],
            summary.loc[model, "integrity_top1_std"],
        ]
        ax.bar(x + (pos - 1) * width, values, width, yerr=errors, color=color, capsize=2, linewidth=0.4, edgecolor=DARK)
    ax.set_ylim(0.35, 1.01)
    ax.set_xticks(x, localization, rotation=0)
    ax.tick_params(axis="x", labelsize=6.4)
    ax.set_ylabel("exact recovery")
    ax.set_title("(b) Row and scenario localization", loc="left", fontweight="bold")
    ax.grid(axis="y", color=LIGHT, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    confusion = _pooled_confusion("typed")
    normalized = confusion / confusion.sum(axis=1, keepdims=True)
    image = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="equal")
    short = ["N", "F", "L", "G", "Ld", "M", "MP", "B", "Mx"]
    for i in range(9):
        for j in range(9):
            if normalized[i, j] >= 0.08:
                ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=7.0, color="white" if normalized[i, j] > 0.55 else DARK)
    ax.set_xticks(range(9), short)
    ax.set_yticks(range(9), short)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("(c) Typed event confusion", loc="left", fontweight="bold")
    ax.tick_params(length=0)

    handles = [patches.Patch(facecolor=color, edgecolor=DARK, label=label) for color, label in zip(colors, labels)]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.33, 0.99), columnspacing=1.0)
    _save(fig, "fig3_causal_results")


def build_per_event_figure() -> None:
    """Expose event-specific gains and failures hidden by aggregate metrics."""

    frame = pd.read_csv(EVIDENCE / "per_event_metrics_summary.csv")
    lookup = frame.set_index(["model", "event"])
    models = ["flat", "typed", "typed_topology"]
    labels = ["Flat", "Typed", "Typed + topology"]
    colors = [GRAY, BLUE, PURPLE]
    width = 0.23

    fig = plt.figure(figsize=(7.16, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 1.23, 0.90], wspace=0.33, left=0.06, right=0.99, bottom=0.24, top=0.83)

    ax = fig.add_subplot(gs[0, 0])
    events = list(range(9))
    x = np.arange(len(events))
    for pos, (model, label, color) in enumerate(zip(models[:2], labels[:2], colors[:2])):
        values = [lookup.loc[(model, event), "event_recall_mean"] for event in events]
        errors = [lookup.loc[(model, event), "event_recall_std"] for event in events]
        ax.bar(x + (pos - 0.5) * width, values, width, yerr=errors, color=color, edgecolor=DARK, linewidth=0.35, capsize=1.5, label=label)
    ax.set_xticks(x, ["N", "F", "L", "G", "Ld", "M", "MP", "B", "Mx"])
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("recall")
    ax.set_title("(a) Event recall", loc="left", fontweight="bold")
    ax.grid(axis="y", color=LIGHT, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    events = [1, 2, 3, 4, 6, 8]
    names = ["F", "L", "G", "Ld", "MP", "Mx"]
    x = np.arange(len(events))
    for pos, (model, label, color) in enumerate(zip(models, labels, colors)):
        values = [lookup.loc[(model, event), "physical_top1_mean"] for event in events]
        errors = [lookup.loc[(model, event), "physical_top1_std"] for event in events]
        ax.bar(x + (pos - 1) * width, values, width, yerr=errors, color=color, edgecolor=DARK, linewidth=0.35, capsize=1.5, label=label)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Top-1")
    ax.set_title("(b) Physical localization", loc="left", fontweight="bold")
    ax.grid(axis="y", color=LIGHT, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    events = [5, 6, 7, 8]
    names = ["M", "MP", "B", "Mx"]
    x = np.arange(len(events))
    for pos, (model, label, color) in enumerate(zip(models, labels, colors)):
        values = [lookup.loc[(model, event), "integrity_top1_mean"] for event in events]
        errors = [lookup.loc[(model, event), "integrity_top1_std"] for event in events]
        ax.bar(x + (pos - 1) * width, values, width, yerr=errors, color=color, edgecolor=DARK, linewidth=0.35, capsize=1.5, label=label)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Top-1")
    ax.set_title("(c) Integrity localization", loc="left", fontweight="bold")
    ax.grid(axis="y", color=LIGHT, linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [patches.Patch(facecolor=color, edgecolor=DARK, label=label) for color, label in zip(colors, labels)]
    fig.legend(handles=handles, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.53, 0.99), columnspacing=1.0)
    _save(fig, "fig4_per_event_diagnostics")


if __name__ == "__main__":
    build_system_figure()
    build_protocol_figure()
    build_results_figure()
    build_per_event_figure()
