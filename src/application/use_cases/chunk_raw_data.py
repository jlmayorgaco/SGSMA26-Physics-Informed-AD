"""Use case for chunking raw PMU data with legacy-compatible outputs."""

from __future__ import annotations

from src.data_engineering.chunking import ChunkPipelinePaths, run_chunk_pipeline


def run_chunk_raw_data_use_case(
    input_dir: str,
    output_dir: str,
    enable_sanity_check: bool = True,
    generate_plots: bool = False,
) -> None:
    """Execute migrated m0 chunking flow."""
    paths = ChunkPipelinePaths(input_dir=input_dir, output_dir=output_dir)
    run_chunk_pipeline(
        paths=paths,
        enable_sanity_check=enable_sanity_check,
        generate_plots=generate_plots,
    )
