#!/usr/bin/env python3
from __future__ import annotations

import warnings
from pathlib import Path

from src.analysis.loader import load_all_buses
from src.analysis.events import summarize_event_spans
from src.config.config import AnalysisConfig
from src.exporters.bus_exporter import export_per_bus
from src.exporters.dataset_exporter import export_dataset_level
from src.exporters.event_exporter import export_events
from src.exporters.general_exporter import export_general_normal_operation
from src.exporters.metadata_exporter import write_event_dictionary, write_outputs_readme
from src.io.output_layout import build_root_dirs
from src.utils.cli import parse_args
from src.utils.filesystem import ensure_dir
from src.utils.serialization import save_json



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:

    config = parse_args()
    ensure_dir(config.output_dir)

    write_event_dictionary(config)
    roots = build_root_dirs(config)

    buses = load_all_buses(config.input_dir, config.pattern)
    
    dataset_spans, correlations, rankings = export_dataset_level(buses=buses, config=config, roots=roots)
    general_json = export_general_normal_operation(buses=buses, config=config, roots=roots)
    per_bus_full_json = export_per_bus(buses=buses, config=config)
    export_events(buses=buses, dataset_spans=dataset_spans, config=config)

    recipe_seed = {
        "global": {
            "event_library": summarize_event_spans(dataset_spans),
            "recommended_pipeline_order": [
                "ANDES clean physics",
                "PMU observation/filter layer",
                "bus/channel operating baselines",
                "colored stochastic noise",
                "heavy-tail residual shaping",
                "artifact / bad-data bursts",
                "missing-data bursts",
                "event-duration relabeling at PMU-visible level",
            ],
            "notes": [
                "This analyzer focuses on PMU feature realism, not raw waveform harmonics.",
                "Low-frequency spectral analysis is valid for these ~30 fps synchrophasor CSVs.",
            ],
        },
        "per_bus_normal_operation": general_json,
        "per_bus_full_timeline": per_bus_full_json,
        "cross_bus_rankings": rankings,
        "cross_bus_correlations": correlations,
    }
    save_json(roots["dataset"] / "json" / "simulation_copy_recipe_seed.json", recipe_seed)

    write_outputs_readme(config)
    print("Done.")
    print(f"Input: {config.input_dir}")
    print(f"Output: {config.output_dir}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        main()
