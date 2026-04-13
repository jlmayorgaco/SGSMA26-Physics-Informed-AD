"""Backfill missing review plots for an existing V2 synthetic dataset.

This script does not rerun the simulator and does not rewrite PMU CSVs. It
reads each saved ``SIM_####.json`` plus the already generated CSV files, writes
missing engineer-review PNG/markdown artifacts, and updates scenario metadata
so training can continue from the same 5000 scenarios.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "pipelines"
for path in (PROJECT_ROOT, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from make_synth_data import (  # noqa: E402
    PMU_BUSES,
    bus_columns,
    load_config,
    parse_plot_bus_option,
    plot_and_report_scenario,
)


SCENARIO_ID_RE = re.compile(r"SIM_(\d+)")
BUS_FILE_RE = re.compile(r"Bus(\d+)_Competition_Data_nanmask\.csv$")


@dataclass(frozen=True)
class PlotBackfillConfig:
    dataset_dir: Path
    generator_config: dict[str, Any]
    pattern: str
    plot_buses: str | list[int]
    include_current_plots: bool
    include_frequency_plots: bool
    overwrite: bool
    dry_run: bool
    max_scenarios: int | None
    start_index: int | None
    end_index: int | None


@dataclass
class PlotBackfillResult:
    scenario_id: str
    scenario_json: str
    status: str
    artifacts: list[str]
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic_v2",
        help="Existing synthetic dataset directory containing SIM_#### folders.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "pipelines" / "make_synth_data.json",
        help="Generator config used as a fallback when the dataset manifest has no config.",
    )
    parser.add_argument(
        "--pattern",
        default="SIM_*/SIM_*.json",
        help="Glob pattern, relative to --dataset-dir, used to find scenario JSON files.",
    )
    parser.add_argument(
        "--plot-buses",
        default="all",
        help='Review buses to plot: "pmu", "all", or comma-separated bus numbers.',
    )
    parser.add_argument("--no-current-plots", action="store_true", help="Skip phase-current plots.")
    parser.add_argument("--no-frequency-plots", action="store_true", help="Skip frequency/ROCOF plots.")
    parser.add_argument("--overwrite", action="store_true", help="Rewrite plots even if all expected artifacts exist.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be plotted without writing files.")
    parser.add_argument("--max-scenarios", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--start-index", type=int, default=None, help="First SIM index to include, inclusive.")
    parser.add_argument("--end-index", type=int, default=None, help="Last SIM index to include, inclusive.")
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def scenario_number(path: Path) -> int | None:
    match = SCENARIO_ID_RE.search(str(path))
    return int(match.group(1)) if match else None


class ScenarioPlotBackfiller:
    """Regenerate review artifacts for one scenario from saved CSV/JSON data."""

    def __init__(self, config: PlotBackfillConfig) -> None:
        self.config = config

    def run(self, scenario_json: Path) -> PlotBackfillResult:
        scenario = load_json(scenario_json)
        scenario_id = str(scenario.get("scenario_id", scenario_json.parent.name))
        try:
            buses = self._selected_buses(scenario)
            expected = self._expected_artifacts(buses)
            complete = self._artifacts_exist(scenario_json.parent, expected)
            if complete and not self.config.overwrite:
                status = self._update_scenario_artifacts(scenario_json, scenario, expected)
                return PlotBackfillResult(
                    scenario_id=scenario_id,
                    scenario_json=str(scenario_json.resolve()),
                    status=status,
                    artifacts=expected,
                )

            if self.config.dry_run:
                return PlotBackfillResult(
                    scenario_id=scenario_id,
                    scenario_json=str(scenario_json.resolve()),
                    status="would_write",
                    artifacts=expected,
                )

            csv_files = self._csv_files(scenario, scenario_json.parent)
            frames = self._read_frames(scenario_json.parent, csv_files, buses)
            scenario_config = self._scenario_config(scenario, buses)
            artifacts = plot_and_report_scenario(
                scenario,
                frames,
                csv_files,
                scenario_json.parent,
                scenario_config,
            )
            scenario["review_artifacts"] = artifacts
            write_json(scenario_json, scenario)
            return PlotBackfillResult(
                scenario_id=scenario_id,
                scenario_json=str(scenario_json.resolve()),
                status="written",
                artifacts=artifacts,
            )
        except Exception as exc:
            return PlotBackfillResult(
                scenario_id=scenario_id,
                scenario_json=str(scenario_json.resolve()),
                status="error",
                artifacts=[],
                error=str(exc),
            )

    def _selected_buses(self, scenario: dict[str, Any]) -> list[int]:
        csv_files = scenario.get("csv_files", {})
        available = {int(bus) for bus in csv_files if str(bus).isdigit()}
        if not available:
            available = set(int(bus) for bus in scenario.get("output_buses", []))
        pmu_buses = [int(bus) for bus in scenario.get("pmu_buses", PMU_BUSES)]
        option = self.config.plot_buses
        if isinstance(option, str) and option.lower() == "all":
            buses = sorted(available or set(range(1, 40)))
        elif isinstance(option, str) and option.lower() in {"pmu", "pmus"}:
            buses = pmu_buses
        elif isinstance(option, list):
            buses = [int(bus) for bus in option]
        else:
            parsed = parse_plot_bus_option(str(option), pmu_buses)
            buses = sorted(available or set(range(1, 40))) if parsed == "all" else [int(bus) for bus in parsed]
        return [bus for bus in buses if bus in (available or set(buses))]

    def _expected_artifacts(self, buses: list[int]) -> list[str]:
        artifacts: list[str] = []
        for bus in buses:
            artifacts.append(f"plots/node_{bus}_v_(a,b,c)_mag_vs_time.png")
            if self.config.include_current_plots:
                artifacts.append(f"plots/node_{bus}_i_(a,b,c)_mag_vs_time.png")
            if self.config.include_frequency_plots:
                artifacts.append(f"plots/node_{bus}_freq_rocof_vs_time.png")
        artifacts.append("plots/ieee39_event_diagram.png")
        artifacts.append("engineering_review.md")
        return artifacts

    def _artifacts_exist(self, scenario_dir: Path, artifacts: list[str]) -> bool:
        return all((scenario_dir / artifact).exists() for artifact in artifacts)

    def _update_scenario_artifacts(
        self,
        scenario_json: Path,
        scenario: dict[str, Any],
        artifacts: list[str],
    ) -> str:
        if scenario.get("review_artifacts") == artifacts or self.config.dry_run:
            return "skipped_complete"
        scenario["review_artifacts"] = artifacts
        write_json(scenario_json, scenario)
        return "metadata_updated"

    def _csv_files(self, scenario: dict[str, Any], scenario_dir: Path) -> dict[int, str]:
        raw = scenario.get("csv_files", {})
        csv_files = {int(bus): str(path) for bus, path in raw.items() if str(bus).isdigit()}
        if csv_files:
            return csv_files
        discovered: dict[int, str] = {}
        for path in sorted((scenario_dir / "csv").glob("*Competition_Data_nanmask.csv")):
            match = BUS_FILE_RE.search(path.name)
            if match:
                discovered[int(match.group(1))] = path.relative_to(scenario_dir).as_posix()
        if not discovered:
            raise FileNotFoundError(f"No bus CSV files found under {scenario_dir}")
        return discovered

    def _read_frames(self, scenario_dir: Path, csv_files: dict[int, str], buses: list[int]) -> dict[int, pd.DataFrame]:
        frames: dict[int, pd.DataFrame] = {}
        for bus in buses:
            if bus not in csv_files:
                raise FileNotFoundError(f"Scenario is missing CSV metadata for bus {bus}")
            path = scenario_dir / csv_files[bus]
            if not path.exists():
                raise FileNotFoundError(f"Scenario CSV does not exist: {path}")
            frame = pd.read_csv(path)
            expected = bus_columns(bus)
            if list(frame.columns) != expected:
                raise ValueError(f"{path} does not match the expected bus {bus} schema")
            frames[bus] = frame
        return frames

    def _scenario_config(self, scenario: dict[str, Any], buses: list[int]) -> dict[str, Any]:
        config = json.loads(json.dumps(self.config.generator_config))
        config["pmu_buses"] = [int(bus) for bus in scenario.get("pmu_buses", config.get("pmu_buses", PMU_BUSES))]
        review = config.setdefault("review", {})
        review["plot_buses_per_scenario"] = buses
        review["include_current_plots"] = self.config.include_current_plots
        review["include_frequency_plots"] = self.config.include_frequency_plots
        return config


class ManifestUpdater:
    """Keep manifest and validation plot counts aligned with backfilled artifacts."""

    def __init__(self, dataset_dir: Path, dry_run: bool) -> None:
        self.dataset_dir = dataset_dir
        self.dry_run = dry_run
        self.manifest_path = dataset_dir / "synthetic_dataset_manifest.json"
        self.validation_path = dataset_dir / "validation_summary.json"

    def update(self, results: list[PlotBackfillResult]) -> dict[str, Any]:
        result_by_id = {
            result.scenario_id: result
            for result in results
            if result.status in {"written", "metadata_updated", "skipped_complete"}
        }
        if self.manifest_path.exists() and not self.dry_run:
            manifest = load_json(self.manifest_path)
            for item in manifest.get("scenarios", []):
                result = result_by_id.get(str(item.get("scenario_id")))
                if result and result.artifacts:
                    item["review_artifacts"] = result.artifacts
            manifest["artifact_counts"] = self._artifact_counts_from_manifest(manifest)
            write_json(self.manifest_path, manifest)

        actual_counts = self._count_actual_artifacts()
        summary = {
            "dataset_root": str(self.dataset_dir.resolve()),
            "scenario_results": self._status_counts(results),
            "selected_scenarios": len(results),
            "review_png_files_on_disk": actual_counts["review_png_files"],
            "engineering_review_markdown_on_disk": actual_counts["engineering_review_markdown"],
            "errors": [
                {
                    "scenario_id": result.scenario_id,
                    "scenario_json": result.scenario_json,
                    "error": result.error,
                }
                for result in results
                if result.error
            ],
            "passed": not any(result.error for result in results),
        }
        if not self.dry_run:
            write_json(self.dataset_dir / "plot_generation_summary.json", summary)
            self._update_validation_summary(actual_counts)
        return summary

    def _artifact_counts_from_manifest(self, manifest: dict[str, Any]) -> dict[str, int]:
        scenarios = manifest.get("scenarios", [])
        return {
            "scenario_dirs": len(scenarios),
            "bus_csv_files": int(manifest.get("artifact_counts", {}).get("bus_csv_files", 0)),
            "review_png_files": sum(
                1
                for scenario in scenarios
                for artifact in scenario.get("review_artifacts", [])
                if str(artifact).endswith(".png")
            ),
            "engineering_review_markdown": sum(
                1
                for scenario in scenarios
                for artifact in scenario.get("review_artifacts", [])
                if str(artifact).endswith(".md")
            ),
        }

    def _count_actual_artifacts(self) -> dict[str, int]:
        png_count = 0
        report_count = 0
        for scenario_dir in self.dataset_dir.glob("SIM_*"):
            png_count += sum(1 for _ in (scenario_dir / "plots").glob("*.png"))
            report_count += int((scenario_dir / "engineering_review.md").exists())
        return {
            "review_png_files": png_count,
            "engineering_review_markdown": report_count,
        }

    def _update_validation_summary(self, counts: dict[str, int]) -> None:
        if not self.validation_path.exists():
            return
        validation = load_json(self.validation_path)
        validation["review_png_files"] = counts["review_png_files"]
        validation["engineering_review_markdown"] = counts["engineering_review_markdown"]
        validation["plot_generation_summary"] = "plot_generation_summary.json"
        write_json(self.validation_path, validation)

    def _status_counts(self, results: list[PlotBackfillResult]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return dict(sorted(counts.items()))


class PlotBackfillPipeline:
    """Find scenario JSONs, backfill plots, and update dataset-level summaries."""

    def __init__(self, config: PlotBackfillConfig) -> None:
        self.config = config
        self.backfiller = ScenarioPlotBackfiller(config)
        self.updater = ManifestUpdater(config.dataset_dir, config.dry_run)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PlotBackfillPipeline":
        dataset_dir = project_path(args.dataset_dir)
        config_path = project_path(args.config)
        manifest_path = dataset_dir / "synthetic_dataset_manifest.json"
        if manifest_path.exists():
            manifest = load_json(manifest_path)
            generator_config = manifest.get("config", {})
        else:
            generator_config = load_config(config_path)
        pmu_buses = [int(bus) for bus in generator_config.get("pmu_buses", PMU_BUSES)]
        plot_buses = parse_plot_bus_option(args.plot_buses, pmu_buses)
        config = PlotBackfillConfig(
            dataset_dir=dataset_dir,
            generator_config=generator_config,
            pattern=args.pattern,
            plot_buses=plot_buses,
            include_current_plots=not args.no_current_plots,
            include_frequency_plots=not args.no_frequency_plots,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            max_scenarios=args.max_scenarios,
            start_index=args.start_index,
            end_index=args.end_index,
        )
        return cls(config)

    def run(self) -> dict[str, Any]:
        scenario_files = self._scenario_files()
        if not scenario_files:
            raise FileNotFoundError(f"No scenario JSON files found with pattern {self.config.dataset_dir / self.config.pattern}")
        results: list[PlotBackfillResult] = []
        for index, scenario_json in enumerate(scenario_files, start=1):
            result = self.backfiller.run(scenario_json)
            results.append(result)
            if index == 1 or index % 100 == 0 or result.status == "error":
                print(f"[{index}/{len(scenario_files)}] {result.scenario_id}: {result.status}")
        return self.updater.update(results)

    def _scenario_files(self) -> list[Path]:
        files = sorted(self.config.dataset_dir.glob(self.config.pattern))
        selected: list[Path] = []
        for path in files:
            number = scenario_number(path)
            if self.config.start_index is not None and (number is None or number < self.config.start_index):
                continue
            if self.config.end_index is not None and (number is None or number > self.config.end_index):
                continue
            selected.append(path)
        if self.config.max_scenarios is not None:
            selected = selected[: self.config.max_scenarios]
        return selected


def main() -> None:
    pipeline = PlotBackfillPipeline.from_args(parse_args())
    summary = pipeline.run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
