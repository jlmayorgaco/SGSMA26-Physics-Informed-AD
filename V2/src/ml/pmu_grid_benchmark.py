"""Benchmark compact V2 PMU grid models without regenerating synthetic data."""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.ml.models import SUPPORTED_MODEL_NAMES
from src.ml.pmu_grid_pipeline import PmuGridTrainingPipeline, TrainingConfig


SUPPORTED_MODELS = SUPPORTED_MODEL_NAMES


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for comparing several 8-PMU to 39-bus model families."""

    synthetic_dir: Path
    output_dir: Path
    models: list[str] = field(default_factory=lambda: SUPPORTED_MODELS.copy())
    window_sec: float = 1.0
    samples_per_event: int = 3
    normal_samples_per_scenario: int = 8
    max_scenarios: int | None = None
    train_fraction: float = 0.70
    runs: int = 1
    seed: int = 20260412
    skip_completed: bool = False


class PmuGridBenchmarkPipeline:
    """Train several candidate models and select the strongest production model."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        start = time.perf_counter()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        rows = [self._train_model(model_name) for model_name in self._models()]
        successful = [row for row in rows if row["status"] == "ok"]
        if not successful:
            self._write_outputs(rows, None, time.perf_counter() - start)
            raise RuntimeError("No benchmark model completed successfully.")

        best = max(
            successful,
            key=lambda row: (
                float(row["benchmark_score"]),
                float(row["selection_score"]),
            ),
        )
        production_model = self.config.output_dir / "pmu_grid_model.pkl"
        shutil.copy2(best["best_model_path"], production_model)
        best["production_model_path"] = str(production_model.resolve())
        summary = self._write_outputs(rows, best, time.perf_counter() - start)
        print(json.dumps(summary, indent=2))
        return summary

    def _models(self) -> list[str]:
        models = [str(model).strip().lower() for model in self.config.models if str(model).strip()]
        invalid = sorted(set(models) - set(SUPPORTED_MODELS))
        if invalid:
            raise ValueError(f"Unsupported models: {invalid}. Supported: {SUPPORTED_MODELS}")
        return list(dict.fromkeys(models))

    def _train_model(self, model_name: str) -> dict[str, Any]:
        model_dir = self.config.output_dir / model_name
        summary_path = model_dir / "pmu_grid_training_summary.json"
        try:
            if self.config.skip_completed and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                return self._row_from_summary(model_name, model_dir, summary, cached=True)
            summary = PmuGridTrainingPipeline(
                TrainingConfig(
                    synthetic_dir=self.config.synthetic_dir,
                    output_dir=model_dir,
                    model_name=model_name,
                    window_sec=self.config.window_sec,
                    samples_per_event=self.config.samples_per_event,
                    normal_samples_per_scenario=self.config.normal_samples_per_scenario,
                    max_scenarios=self.config.max_scenarios,
                    train_fraction=self.config.train_fraction,
                    runs=self.config.runs,
                    seed=self.config.seed,
                )
            ).run()
            return self._row_from_summary(model_name, model_dir, summary, cached=False)
        except Exception as exc:
            return {
                "status": "error",
                "requested_model": model_name,
                "resolved_model": "",
                "error": str(exc),
                "summary_path": str((model_dir / "pmu_grid_training_summary.json").resolve()),
            }

    def _row_from_summary(
        self,
        model_name: str,
        model_dir: Path,
        summary: dict[str, Any],
        cached: bool,
    ) -> dict[str, Any]:
        best = summary["best_metrics"]
        serialized_size = int(best.get("serialized_size_bytes") or 0)
        selection_score = float(best.get("selection_score") or 0.0)
        return {
            "status": "ok",
            "requested_model": model_name,
            "resolved_model": best.get("model_name"),
            "best_run": summary.get("best_run"),
            "selection_score": selection_score,
            "benchmark_score": self._benchmark_score(selection_score, serialized_size),
            "event_macro_f1": best.get("event_macro_f1"),
            "event_weighted_f1": best.get("event_weighted_f1"),
            "bus_state_macro_f1": best.get("bus_state_macro_f1"),
            "bus_state_weighted_f1": best.get("bus_state_weighted_f1"),
            "active_bus_jaccard_mean": best.get("active_bus_jaccard_mean"),
            "full_state_exact_match": best.get("full_state_exact_match"),
            "serialized_size_bytes": serialized_size,
            "n_samples": summary.get("n_samples"),
            "n_features": summary.get("n_features"),
            "n_scenarios": summary.get("n_scenarios"),
            "train_fraction": summary.get("train_fraction"),
            "test_fraction": summary.get("test_fraction"),
            "elapsed_sec": summary.get("elapsed_sec"),
            "best_model_path": summary.get("best_model_path"),
            "history_path": summary.get("history_path"),
            "summary_path": str((model_dir / "pmu_grid_training_summary.json").resolve()),
            "cached": cached,
        }

    def _write_outputs(
        self,
        rows: list[dict[str, Any]],
        best: dict[str, Any] | None,
        elapsed_sec: float,
    ) -> dict[str, Any]:
        csv_path = self.config.output_dir / "pmu_model_benchmark.csv"
        json_path = self.config.output_dir / "pmu_model_benchmark.json"
        self._write_csv(csv_path, rows)
        summary = {
            "elapsed_sec": round(elapsed_sec, 3),
            "synthetic_dir": str(self.config.synthetic_dir.resolve()),
            "output_dir": str(self.config.output_dir.resolve()),
            "models": rows,
            "best_model": best,
            "benchmark_csv": str(csv_path.resolve()),
            "benchmark_json": str(json_path.resolve()),
            "selection_rule": "max benchmark_score = selection_score - 0.01*log10(serialized_size_bytes + 1)",
            "input_contract": {
                "feature_buses": [2, 5, 6, 10, 19, 22, 29, 39],
                "uses_non_pmu_csv_as_features": False,
                "hidden_non_pmu_csv_role": "39_bus_state_truth_and_review_only",
                "target_buses": list(range(1, 40)),
            },
        }
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if best is not None:
            source_path = self.config.output_dir / "pmu_grid_model_source.json"
            source_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
            summary["production_model_path"] = best.get("production_model_path")
            summary["production_model_source"] = str(source_path.resolve())
        return summary

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _benchmark_score(self, selection_score: float, serialized_size_bytes: int) -> float:
        return float(selection_score - 0.01 * math.log10(float(serialized_size_bytes) + 1.0))
