from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_confusion_matrix(confusion: dict[str, int], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mat = np.array(
        [
            [int(confusion.get("tn", 0)), int(confusion.get("fp", 0))],
            [int(confusion.get("fn", 0)), int(confusion.get("tp", 0))],
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    image = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred Normal", "Pred Abnormal"])
    ax.set_yticks([0, 1], labels=["True Normal", "True Abnormal"])
    for (i, j), value in np.ndenumerate(mat):
        ax.text(j, i, f"{int(value)}", ha="center", va="center", color="black")
    ax.set_title("Confusion Matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_curve(frame: pd.DataFrame, x: str, y: str, output_path: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if not frame.empty and x in frame.columns and y in frame.columns:
        ax.plot(frame[x], frame[y], lw=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_calibration_curve(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    valid = frame.dropna(subset=["mean_pred", "empirical"]) if not frame.empty else frame
    if not valid.empty:
        ax.plot(valid["mean_pred"], valid["empirical"], marker="o")
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical abnormal frequency")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_threshold_tradeoff(sweep: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    if not sweep.empty:
        top = sweep.head(min(30, len(sweep)))
        ax.plot(top.index.to_numpy(), top["f1_abnormal"].to_numpy(dtype=float), label="F1 abnormal")
        ax.plot(top.index.to_numpy(), top["recall_abnormal"].to_numpy(dtype=float), label="Recall abnormal")
        ax.plot(top.index.to_numpy(), top["false_positives_per_minute"].to_numpy(dtype=float), label="FP/min")
    ax.set_title("Threshold Sweep Tradeoff")
    ax.set_xlabel("Candidate rank")
    ax.set_ylabel("Metric value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_per_scenario_f1(per_scenario: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if not per_scenario.empty:
        frame = per_scenario.sort_values("f1_abnormal", ascending=False).reset_index(drop=True)
        ax.bar(np.arange(len(frame)), frame["f1_abnormal"].to_numpy(dtype=float))
        ax.set_xticks(np.arange(len(frame)), labels=frame["scenario_id"].astype(str).tolist(), rotation=60, ha="right")
    ax.set_title("Per-scenario F1 abnormal")
    ax.set_ylabel("F1 abnormal")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def plot_familywise_metrics(familywise: dict[str, dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = ["normal", "physical_heavy", "cyber_heavy", "concurrent_heavy"]
    f1_values = [float(familywise.get(name, {}).get("f1_abnormal", 0.0) or 0.0) for name in names]
    recall_values = [float(familywise.get(name, {}).get("recall_abnormal", 0.0) or 0.0) for name in names]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2.0, f1_values, w, label="F1 abnormal")
    ax.bar(x + w / 2.0, recall_values, w, label="Recall abnormal")
    ax.set_xticks(x, labels=[n.replace("_", "-") for n in names], rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Family-wise Metrics")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
