from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.application.use_cases.generate_chunk_plots import (
    run_generate_chunk_plots_use_case,
)


LOGGER = logging.getLogger("m0_pipeline_chunk_plots")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-chunk PMU plots from chunked Bus*.csv files."
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("output/M0_RAW0001_NEWARCH/chunks"),
        help="Directory containing chunkXX_event_*/Bus*.csv files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    summary = run_generate_chunk_plots_use_case(
        chunks_dir=args.chunks_dir.resolve(),
    )

    LOGGER.info("Chunk plot generation completed.")
    LOGGER.info("chunks_dir=%s", summary["chunks_dir"])
    LOGGER.info("chunk_count=%s", summary["chunk_count"])
    LOGGER.info("total_bus_plots=%s", summary["total_bus_plots"])
    LOGGER.info("total_event_plots=%s", summary["total_event_plots"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())