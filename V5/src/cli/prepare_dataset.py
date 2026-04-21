"""Phase-1 CLI for dataset preparation wiring."""

from __future__ import annotations

import argparse

from src.infrastructure.legacy.m0_adapter import run_chunk_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare chunked dataset using legacy chunker.")
    parser.add_argument("--plots", action="store_true", help="Enable legacy plots.")
    args = parser.parse_args()

    run_chunk_pipeline(generate_plots=args.plots)


if __name__ == "__main__":
    main()
