"""Phase-1 CLI to run legacy data generation entrypoints."""

from __future__ import annotations

import argparse

from src.infrastructure.legacy.m0_adapter import run_chunk_pipeline
from src.infrastructure.legacy.m1_adapter import run_normalization_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate chunked and normalized data via legacy adapters.")
    parser.add_argument("--skip-normalization", action="store_true", help="Only run chunking.")
    parser.add_argument("--plots", action="store_true", help="Enable legacy plots.")
    args = parser.parse_args()

    run_chunk_pipeline(generate_plots=args.plots)
    if not args.skip_normalization:
        run_normalization_pipeline(generate_plots=args.plots)


if __name__ == "__main__":
    main()
