
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

import argparse
import sys
from pathlib import Path

from src.config.config import AnalysisConfig


def get_default_input_dir() -> Path:
    """Return default input directory (data/RAW0001) relative to project root."""
    # Try relative to current working directory
    cwd = Path.cwd()
    candidates = [
        cwd / "data" / "RAW0001",
        cwd / "V4" / "data" / "RAW0001",
        cwd.parent / "data" / "RAW0001",
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    # If none found, return the first candidate as default (will raise error later)
    return candidates[0]


def get_default_output_dir() -> Path:
    """Return default output directory (outputs/analysis)."""
    cwd = Path.cwd()
    return cwd / "outputs" / "analysis"


def parse_args() -> AnalysisConfig:
    parser = argparse.ArgumentParser(
        description="Analyze an IEEE-39 PMU RAW scenario into dataset/general/buses/events outputs."
    )
    parser.add_argument("--input-dir", type=Path, default=get_default_input_dir(), help="Folder containing Bus*_Competition_Data*.csv files. Default: data/RAW0001")
    parser.add_argument("--output-dir", type=Path, default=get_default_output_dir(), help="Folder where analysis outputs will be written. Default: outputs/analysis")
    parser.add_argument("--pattern", type=str, default="Bus*_Competition_Data*.csv", help="Glob pattern for PMU CSV files.")
    parser.add_argument("--trend-window-seconds", type=float, default=1.0, help="Rolling trend window in seconds for noise analysis.")
    parser.add_argument("--event-baseline-seconds", type=float, default=1.0, help="Pre-event baseline window in seconds.")
    parser.add_argument("--event-post-seconds", type=float, default=1.0, help="Post-event recovery window in seconds.")
    parser.add_argument("--event-context-seconds", type=float, default=1.0, help="Context padding around event windows for plots/exports.")
    parser.add_argument("--artifact-z-threshold", type=float, default=6.0, help="Robust z-score threshold for artifact detection.")
    parser.add_argument("--plots-dpi", type=int, default=220, help="Default DPI for standard plots.")
    parser.add_argument("--plots-zoom-dpi", type=int, default=400, help="DPI for event/context plots.")
    parser.add_argument(
        "--save-per-bus-json",
        action="store_true",
        default=True,
        help="Save per-bus JSON outputs. Default: enabled.",
    )
    parser.add_argument(
        "--no-save-per-bus-json",
        action="store_false",
        dest="save_per_bus_json",
        help="Disable per-bus JSON outputs.",
    )
    parser.add_argument(
        "--save-event-bus-csvs",
        action="store_true",
        default=True,
        help="Save event-per-bus cropped CSVs. Default: enabled.",
    )
    parser.add_argument(
        "--no-save-event-bus-csvs",
        action="store_false",
        dest="save_event_bus_csvs",
        help="Disable event-per-bus cropped CSVs.",
    )

    args = parser.parse_args()

    # Validate input directory exists
    if not args.input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {args.input_dir}\n"
            f"Please provide a valid --input-dir or place CSV files in {get_default_input_dir()}"
        )
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = AnalysisConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        event_context_seconds=args.event_context_seconds,
        trend_window_seconds=args.trend_window_seconds,
        event_baseline_seconds=args.event_baseline_seconds,
        event_post_seconds=args.event_post_seconds,
        artifact_z_threshold=args.artifact_z_threshold,
        plots_dpi=args.plots_dpi,
        plots_zoom_dpi=args.plots_zoom_dpi,
        save_per_bus_json=args.save_per_bus_json,
        save_event_bus_csvs=args.save_event_bus_csvs,
    )
    config.validate()
    return config