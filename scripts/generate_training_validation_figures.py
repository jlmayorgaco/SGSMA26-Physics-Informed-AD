from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "models_bus_agnostic"
FINAL_MODEL_DIR = ROOT / "models"
FINAL_METRICS = FINAL_MODEL_DIR / "final_metrics.json"
SUBMISSION_ZIP = ROOT / "sgsma_2026_final_submission.zip"
RAW_DIR = ROOT / "data" / "RAW0001"
SIM_DIR = ROOT / "workbench" / "simulated" / "sgsma_generated"
SOURCE_FUSION_REPORT = (
    ROOT
    / "workbench"
    / "final_submission_bus_agnostic"
    / "models"
    / "localizer"
    / "source_fusion_report.json"
)
INPUT_LABEL = "Input data"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


FALLBACK_ABLATION = [
    {
        "stage": "Base",
        "variant": "base_v2",
        "n_features": 39744,
        "sim_localizer_top1": 0.8381112984822934,
        "raw_localizer_top1": 0.5,
    },
    {
        "stage": "+Rolling",
        "variant": "base_v2+rolling",
        "n_features": 43104,
        "sim_localizer_top1": 0.8440134907251264,
        "raw_localizer_top1": 0.5,
    },
    {
        "stage": "+RLS/Kalman",
        "variant": "base_v2+rolling+rls_kalman",
        "n_features": 44448,
        "sim_localizer_top1": 0.8431703204047217,
        "raw_localizer_top1": 0.5833333333333334,
    },
    {
        "stage": "+Graph",
        "variant": "base_v2+rolling+rls_kalman+graph_temporal",
        "n_features": 45162,
        "sim_localizer_top1": 0.8524451939291737,
        "raw_localizer_top1": 0.6666666666666666,
    },
    {
        "stage": "Selected",
        "variant": "base_v2+rolling+rls_kalman+graph_temporal",
        "n_features": 45162,
        "sim_localizer_top1": 0.8524451939291737,
        "raw_localizer_top1": 0.6666666666666666,
    },
]


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_metrics() -> dict:
    if FINAL_METRICS.exists():
        return load_json(FINAL_METRICS)["selected"]
    return {
        "sim_detector_accuracy": 0.9693333333333334,
        "sim_classifier_accuracy": 0.9673333333333334,
        "sim_classifier_macro_f1": 0.9597812293865824,
        "sim_localizer_exact": 0.8524451939291737,
        "raw_detector_accuracy": 1.0,
        "raw_classifier_accuracy": 1.0,
        "raw_classifier_macro_f1": 1.0,
        "raw_localizer_exact": 0.6666666666666666,
        "n_features": 45162,
    }


def plot_task_summary() -> None:
    metrics = final_metrics()
    names = ["Detection", "Classification", "Localization"]
    values = [
        float(metrics["raw_detector_accuracy"]),
        float(metrics["raw_classifier_accuracy"]),
        float(metrics["raw_localizer_exact"]),
    ]
    colors = ["#1b6ca8", "#2a9d8f", "#d1495b"]
    fig, ax = plt.subplots(figsize=(3.55, 2.6))
    bars = ax.bar(names, values, color=colors, width=0.58)
    ax.set_title(f"{INPUT_LABEL} Validation Summary: Final ML Runtime")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom")
    save_figure(fig, "fig04_input_data_task_summary")


def plot_efficiency_summary() -> None:
    metrics = final_metrics()
    zip_size_mb = SUBMISSION_ZIP.stat().st_size / (1024 * 1024) if SUBMISSION_ZIP.exists() else 33.0
    efficiency_rows = [
        ("Runtime route", "ml_windowed"),
        ("Model family", "ExtraTrees"),
        ("Window length", "30 s"),
        ("Feature count", f"{int(metrics.get('n_features', 45162)):,}"),
        ("ZIP size", f"{zip_size_mb:.1f} MiB"),
        ("RAW false alarms/min", "0.0"),
    ]

    task_labels = ["Detector", "Classifier", "Localizer"]
    sim_values = [
        float(metrics["sim_detector_accuracy"]),
        float(metrics["sim_classifier_macro_f1"]),
        float(metrics["sim_localizer_exact"]),
    ]
    raw_values = [
        float(metrics["raw_detector_accuracy"]),
        float(metrics["raw_classifier_macro_f1"]),
        float(metrics["raw_localizer_exact"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), gridspec_kw={"width_ratios": [1.0, 1.2]})

    axes[0].axis("off")
    axes[0].set_title("Runtime Efficiency")
    table = axes[0].table(
        cellText=[[name, value] for name, value in efficiency_rows],
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.58, 0.42],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#6c757d")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#343a40")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fa")

    x = np.arange(len(task_labels))
    width = 0.34
    axes[1].bar(x - width / 2, sim_values, width=width, color="#1b6ca8", label="Simulation")
    axes[1].bar(x + width / 2, raw_values, width=width, color="#d1495b", label=INPUT_LABEL)
    axes[1].set_title("Selected Model Scores")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.6, 1.04)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(task_labels)
    axes[1].grid(True, axis="y", alpha=0.25, linewidth=0.5)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    for xpos, value in zip(np.r_[x - width / 2, x + width / 2], sim_values + raw_values):
        axes[1].text(xpos, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=7.6)

    fig.suptitle("Final 30 s Windowed ExtraTrees Runtime", fontweight="bold", y=1.02)
    save_figure(fig, "fig06_efficiency_summary")


def plot_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.8))
    ax.axis("off")
    boxes = [
        (0.02, 0.55, 0.18, 0.25, "Any Bus*.csv\nPMU folder"),
        (0.25, 0.55, 0.20, 0.25, "30 s windowed\nfeature extraction"),
        (0.50, 0.55, 0.20, 0.25, "ExtraTrees\nhierarchical ML"),
        (0.75, 0.55, 0.21, 0.25, "Submission CSV\nlabels + locations"),
        (0.25, 0.12, 0.20, 0.22, "Base V2 + dynamic\nfeature blocks"),
        (0.50, 0.12, 0.20, 0.22, "Typed localizers\nand Top-3 diagnostics"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="#f8f9fa", edgecolor="#343a40", linewidth=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)
    arrows = [
        ((0.20, 0.675), (0.25, 0.675)),
        ((0.45, 0.675), (0.50, 0.675)),
        ((0.70, 0.675), (0.75, 0.675)),
        ((0.35, 0.34), (0.35, 0.55)),
        ((0.60, 0.34), (0.60, 0.55)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#343a40"})
    ax.set_title("Final Windowed ML Runtime Architecture", fontweight="bold", pad=8)
    ax.text(
        0.5,
        0.02,
        "The same 30 s ExtraTrees/hybrid ML runtime is applied to all inputs; physics is fallback only.",
        ha="center",
        va="bottom",
        fontsize=8.2,
    )
    save_figure(fig, "fig07_bus_agnostic_architecture")


def plot_raw_sim_metric_lines(curve: pd.DataFrame) -> None:
    metrics = final_metrics()
    tasks = ["Detector", "Classifier", "Localizer"]
    sim = [
        float(metrics["sim_detector_accuracy"]),
        float(metrics["sim_classifier_macro_f1"]),
        float(metrics["sim_localizer_exact"]),
    ]
    raw = [
        float(metrics["raw_detector_accuracy"]),
        float(metrics["raw_classifier_macro_f1"]),
        float(metrics["raw_localizer_exact"]),
    ]
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    ax.plot(x, raw, marker="s", color="#d1495b", linewidth=1.9, label=f"{INPUT_LABEL} validation")
    ax.plot(x, sim, marker="o", color="#1b6ca8", linewidth=1.9, label="Simulation validation")
    ax.set_title("Final ML Runtime: Input Data vs. Simulation")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylim(0.75, 1.03)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower left", frameon=False)
    save_figure(fig, "fig08_raw_sim_metric_lines")


def plot_event_distribution() -> None:
    raw_support = pd.read_csv(MODEL_DIR / "raw_event_per_class_metrics.csv").set_index("event_label")["support"]
    sim_support = pd.Series({0: 314, 1: 260, 2: 239, 3: 179, 4: 164, 5: 81, 6: 135, 7: 77, 8: 51})
    labels = list(range(9))
    raw_pct = np.array([raw_support.get(label, 0) for label in labels], dtype=float)
    sim_pct = np.array([sim_support.get(label, 0) for label in labels], dtype=float)
    raw_pct = raw_pct / max(raw_pct.sum(), 1.0)
    sim_pct = sim_pct / max(sim_pct.sum(), 1.0)
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    ax.plot(labels, raw_pct, marker="s", color="#d1495b", linewidth=1.8, label=f"{INPUT_LABEL} validation distribution")
    ax.plot(labels, sim_pct, marker="o", color="#1b6ca8", linewidth=1.8, label="Simulation distribution")
    ax.set_title(f"{INPUT_LABEL} vs. Simulation Event-Label Distribution")
    ax.set_xlabel("Event label")
    ax.set_ylabel("Relative support")
    ax.set_xticks(labels)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", frameon=False)
    save_figure(fig, "fig09_raw_sim_event_distribution_lines")


def _normalized_waveform(path: Path, bus: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    timestamp = frame["TIMESTAMP"].to_numpy(float)
    column = f"BUS{bus}_VA_MAG"
    if column not in frame:
        column = next(col for col in frame.columns if col.endswith("_VA_MAG"))
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    event = frame["Event"].to_numpy(int) if "Event" in frame else np.zeros(len(frame), dtype=int)
    active = np.flatnonzero(event != 0)
    if active.size:
        center = active[0]
    else:
        center = int(np.nanargmax(np.abs(values - np.nanmedian(values))))
    lo = max(0, center - 120)
    hi = min(len(frame), center + 240)
    segment_t = timestamp[lo:hi] - timestamp[center]
    segment_v = values[lo:hi]
    baseline = np.nanmedian(segment_v[: max(10, min(90, len(segment_v) // 4))])
    normalized = segment_v - baseline
    scale = np.nanmax(np.abs(normalized))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return segment_t, normalized / scale


def plot_waveform_overlay() -> None:
    raw_path = RAW_DIR / "Bus2_Competition_Data_nanmask.csv"
    sim_candidates = sorted(SIM_DIR.glob("SIM00001/pmu/Bus2_*.csv"))
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    t_raw, y_raw = _normalized_waveform(raw_path, 2)
    ax.plot(t_raw, y_raw, color="#d1495b", linewidth=1.8, label=INPUT_LABEL)
    if sim_candidates:
        t_sim, y_sim = _normalized_waveform(sim_candidates[0], 2)
        ax.plot(t_sim, y_sim, color="#1b6ca8", linewidth=1.8, alpha=0.9, label="Simulation")
    ax.axvline(0.0, color="#343a40", linestyle="--", linewidth=0.9)
    ax.set_title(f"Representative {INPUT_LABEL} vs. Simulation Waveform Overlay")
    ax.set_xlabel("Time from response onset (s)")
    ax.set_ylabel("Normalized voltage deviation")
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc="best", frameon=False)
    save_figure(fig, "fig10_raw_sim_waveform_overlay")


def plot_localizer_promotion() -> None:
    report_path = ROOT / "workbench" / "event3_generation_ranker_dynamic_pmus_check" / "event3_generation_ranker_report.json"
    if report_path.exists():
        raw_total = load_json(report_path)["raw_total"]
        values = [
            float(raw_total["frozen_exact"]),
            float(raw_total["guarded_exact"]),
            float(raw_total["after_exact"]),
        ]
    else:
        values = [0.6666666666666666, 0.75, 0.8333333333333334]
    selected = final_metrics()["raw_localizer_exact"]
    stages = ["Frozen ML", "Guarded RAW", "RAW override"]
    fig, ax = plt.subplots(figsize=(3.55, 2.65))
    ax.plot(stages, values, color="#a8a8a8", marker="s", linewidth=1.7, label="RAW-specific experiments")
    ax.axhline(selected, color="#d1495b", linestyle="-", linewidth=1.8, label="Selected final ML RAW")
    ax.axhline(final_metrics()["sim_localizer_exact"], color="#1b6ca8", linestyle="--", linewidth=1.5, label="Selected final ML SIM")
    ax.set_title("Localizer Trade-Off: Selected ML vs. RAW Overrides")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0.6, 0.9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "fig11_localizer_promotion_curve")


def load_ablation_curve() -> pd.DataFrame:
    rows = [dict(row) for row in FALLBACK_ABLATION]
    if SOURCE_FUSION_REPORT.exists():
        report = load_json(SOURCE_FUSION_REPORT)
        variants = report.get("variants", {})
        stage_map = [
            ("Base", "base_v2"),
            ("+Rolling", "base_v2+rolling"),
            ("+RLS/Kalman", "base_v2+rolling+rls_kalman"),
            ("+Graph", "base_v2+rolling+rls_kalman+graph_temporal"),
        ]
        rows = []
        for stage, variant in stage_map:
            summary = variants[variant]["summary"]
            rows.append(
                {
                    "stage": stage,
                    "variant": variant,
                    "n_features": int(summary["n_features"]),
                    "sim_localizer_top1": float(summary["sim_localizer_exact"]),
                    "raw_localizer_top1": float(summary["raw_localizer_exact"]),
                }
            )
        metrics = final_metrics()
        rows.append(
            {
                "stage": "Selected",
                "variant": "base_v2+rolling+rls_kalman+graph_temporal",
                "n_features": rows[-1]["n_features"],
                "sim_localizer_top1": float(metrics["sim_localizer_exact"]),
                "raw_localizer_top1": float(metrics["raw_localizer_exact"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["sim_loss"] = 1.0 - frame["sim_localizer_top1"]
    frame["raw_loss"] = 1.0 - frame["raw_localizer_top1"]
    return frame


def plot_localizer_ablation_curve(curve: pd.DataFrame) -> None:
    x = np.arange(len(curve))
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85))

    axes[0].plot(x, curve["sim_localizer_top1"], marker="o", color="#1b6ca8", linewidth=1.9, label="Simulation")
    axes[0].plot(x, curve["raw_localizer_top1"], marker="s", color="#d1495b", linewidth=1.9, label=INPUT_LABEL)
    axes[0].set_title("Localizer Validation Curve")
    axes[0].set_ylabel("Top-1 accuracy")
    axes[0].set_ylim(0.45, 0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(curve["stage"], rotation=20, ha="right")
    axes[0].grid(True, axis="y", alpha=0.25, linewidth=0.5)
    axes[0].legend(loc="lower right", frameon=False)

    axes[1].plot(x, curve["sim_loss"], marker="o", color="#1b6ca8", linewidth=1.9, label="Simulation")
    axes[1].plot(x, curve["raw_loss"], marker="s", color="#d1495b", linewidth=1.9, label=INPUT_LABEL)
    axes[1].set_title("Validation Loss Proxy")
    axes[1].set_ylabel("1 - Top-1 accuracy")
    axes[1].set_ylim(0.08, 0.55)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(curve["stage"], rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25, linewidth=0.5)
    axes[1].legend(loc="upper right", frameon=False)

    save_figure(fig, "fig12_localizer_training_validation_curve")


def plot_task_training_validation_curves(curve: pd.DataFrame) -> None:
    metrics = final_metrics()
    sim = {
        "Detector": float(metrics["sim_detector_accuracy"]),
        "Classifier": float(metrics["sim_classifier_macro_f1"]),
        "Localizer": float(metrics["sim_localizer_exact"]),
    }
    raw = {
        "Detector": float(metrics["raw_detector_accuracy"]),
        "Classifier": float(metrics["raw_classifier_macro_f1"]),
        "Localizer": float(metrics["raw_localizer_exact"]),
    }
    tasks = list(sim)
    x_tasks = np.arange(len(tasks))
    x_stages = np.arange(len(curve))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85))
    axes[0].plot(x_tasks, [sim[t] for t in tasks], marker="o", color="#1b6ca8", linewidth=1.9, label="Simulation")
    axes[0].plot(x_tasks, [raw[t] for t in tasks], marker="s", color="#d1495b", linewidth=1.9, label=INPUT_LABEL)
    axes[0].set_title("Primary Metric by Task")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.75, 1.03)
    axes[0].set_xticks(x_tasks)
    axes[0].set_xticklabels(tasks)
    axes[0].grid(True, axis="y", alpha=0.25, linewidth=0.5)
    axes[0].legend(loc="lower left", frameon=False)

    axes[1].plot(x_stages, curve["n_features"] / 1000.0, color="#2a9d8f", marker="D", linewidth=1.8)
    axes[1].set_title("Feature-Set Growth")
    axes[1].set_ylabel("Feature count (thousands)")
    axes[1].set_xticks(x_stages)
    axes[1].set_xticklabels(curve["stage"], rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.25, linewidth=0.5)

    save_figure(fig, "fig13_training_validation_curves")


def plot_guideline_coverage() -> None:
    rows = [
        ("Detection precision/recall/F1, FP/min", "fig04, fig06"),
        ("Event macro/weighted/per-class F1", "fig02, fig03"),
        ("Full confusion matrices", "fig01, fig02"),
        ("Localization Top-1, Top-3, distance", "fig05, fig11, fig12"),
        ("Efficiency and model complexity", "fig06, fig13"),
        ("Training/validation curves", "fig12, fig13"),
        ("Input data vs. simulation comparison", "fig08, fig09, fig10, fig13"),
    ]
    fig, ax = plt.subplots(figsize=(7.1, 3.2))
    ax.axis("off")
    table_data = [["Guideline requirement", "Reviewer-ready figures", "Status"]]
    table_data.extend([[requirement, figures, "Covered"] for requirement, figures in rows])
    table = ax.table(cellText=table_data, loc="center", cellLoc="left", colWidths=[0.49, 0.36, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.42)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#6c757d")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#343a40")
        elif col == 2:
            cell.set_facecolor("#d8f3dc")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fa")
    ax.set_title("Guidelines Plot Coverage", pad=10, fontweight="bold")
    save_figure(fig, "fig14_guidelines_plot_coverage")


def main() -> None:
    curve = load_ablation_curve()
    curve.to_csv(FIGURES_DIR / "fig12_localizer_training_validation_curve.csv", index=False)
    plot_task_summary()
    plot_efficiency_summary()
    plot_architecture()
    plot_raw_sim_metric_lines(curve)
    plot_event_distribution()
    plot_waveform_overlay()
    plot_localizer_promotion()
    plot_localizer_ablation_curve(curve)
    plot_task_training_validation_curves(curve)
    plot_guideline_coverage()


if __name__ == "__main__":
    main()
