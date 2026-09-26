"""Adapter for m0 API using migrated src implementation (phase-2 slice b)."""

from __future__ import annotations

from src.application.use_cases.chunk_raw_data import run_chunk_raw_data_use_case
from src.data_engineering.chunking import find_chunk_boundaries, get_chunk_metadata, load_and_synchronize_data


def run_chunk_pipeline(
    input_dir: str | None = None,
    output_dir: str | None = None,
    enable_sanity_check: bool = True,
    generate_plots: bool = False,
) -> None:
    """Run migrated chunk pipeline while preserving legacy defaults."""
    run_chunk_raw_data_use_case(
        input_dir=input_dir or "data/RAW0001/",
        output_dir=output_dir or "output/SCENARIO_RAW0001/",
        enable_sanity_check=enable_sanity_check,
        generate_plots=generate_plots,
    )


__all__ = [
    "run_chunk_pipeline",
    "load_and_synchronize_data",
    "find_chunk_boundaries",
    "get_chunk_metadata",
]
