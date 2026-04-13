"""Plotting helpers for the V2 8-PMU ML pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TrainingPlotter:
    """Create compact training diagnostics from metrics and targets."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.figure_dir = self.output_dir / "figures"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    def plot_history(self, history: pd.DataFrame) -> list[str]:
        if history.empty:
            return []
        metric_cols = [
            "event_macro_f1",
            "bus_state_macro_f1",
            "active_bus_jaccard_mean",
            "selection_score",
        ]
        available = [col for col in metric_cols if col in history.columns]
        if not available:
            return []
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for col in available:
            ax.plot(history["run"], history[col], marker="o", linewidth=2, label=col.replace("_", " "))
        ax.set_title("Training Metrics by Run")
        ax.set_xlabel("Run")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        return [self._save(fig, "training_metrics_by_run.png")]

    def plot_label_counts(self, targets: pd.DataFrame) -> list[str]:
        if "event_label" not in targets:
            return []
        counts = targets["event_label"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar([str(int(label)) for label in counts.index], counts.values, color="#2f6f8f")
        ax.set_title("Training Window Label Counts")
        ax.set_xlabel("Event label")
        ax.set_ylabel("Windows")
        ax.grid(axis="y", alpha=0.25)
        return [self._save(fig, "training_label_counts.png")]

    def plot_train_test_sims(self, train_sims: list[str], test_sims: list[str], run: int) -> list[str]:
        fig, ax = plt.subplots(figsize=(9, 2.8))
        labels = ["train", "test"]
        counts = [len(train_sims), len(test_sims)]
        ax.bar(labels, counts, color=["#356f4c", "#9b3d3d"])
        ax.set_title(f"SIM Split for Run {run:02d}")
        ax.set_ylabel("Scenario count")
        ax.grid(axis="y", alpha=0.25)
        for index, count in enumerate(counts):
            ax.text(index, count, str(count), ha="center", va="bottom")
        return [self._save(fig, f"run_{run:02d}_sim_split.png")]

    def _save(self, fig: plt.Figure, name: str) -> str:
        path = self.figure_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return str(path.resolve())


class PredictionPlotter:
    """Create event timeline and 39-bus state plots for inference output."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = Path(output_path)
        self.figure_dir = self.output_path.parent / "figures"
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    def plot(self, prediction: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        events = prediction.get("events", [])
        paths.append(self._plot_timeline(events))
        if events:
            paths.append(self._plot_state_heatmap(events, prediction.get("target_buses", [])))
        return paths

    def _plot_timeline(self, events: list[dict[str, Any]]) -> str:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        if not events:
            ax.text(0.5, 0.5, "No predicted events", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            labels = sorted({int(event["label"]) for event in events})
            label_to_y = {label: index for index, label in enumerate(labels)}
            for event in events:
                label = int(event["label"])
                y = label_to_y[label]
                start = float(event["start_sec"])
                width = max(0.01, float(event["end_sec"]) - start)
                ax.broken_barh([(start, width)], (y - 0.35, 0.7), facecolors="#2f6f8f", alpha=0.85)
                ax.text(start + width / 2.0, y, str(event.get("location", "")), ha="center", va="center", fontsize=8)
            ax.set_yticks(list(label_to_y.values()))
            ax.set_yticklabels([f"label {label}" for label in labels])
            ax.set_xlabel("Seconds")
            ax.set_ylim(-0.8, len(labels) - 0.2)
        ax.set_title("Predicted Event Timeline")
        ax.grid(axis="x", alpha=0.25)
        return self._save(fig, f"{self.output_path.stem}_timeline.png")

    def _plot_state_heatmap(self, events: list[dict[str, Any]], target_buses: list[int]) -> str:
        buses = [int(bus) for bus in (target_buses or range(1, 40))]
        matrix = np.zeros((len(buses), len(events)), dtype=int)
        for col, event in enumerate(events):
            states = event.get("bus_states", {})
            for row, bus in enumerate(buses):
                matrix[row, col] = int(states.get(str(bus), 0))

        fig, ax = plt.subplots(figsize=(max(7, len(events) * 0.9), 9))
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=8)
        ax.set_title("Predicted 39-Bus State by Event Segment")
        ax.set_xlabel("Predicted segment")
        ax.set_ylabel("IEEE-39 bus")
        ax.set_xticks(range(len(events)))
        ax.set_xticklabels([str(index + 1) for index in range(len(events))])
        ax.set_yticks(range(len(buses)))
        ax.set_yticklabels([str(bus) for bus in buses], fontsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("State label")
        return self._save(fig, f"{self.output_path.stem}_bus_state_heatmap.png")

    def _save(self, fig: plt.Figure, name: str) -> str:
        path = self.figure_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return str(path.resolve())
