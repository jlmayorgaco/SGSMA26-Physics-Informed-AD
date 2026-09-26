from __future__ import annotations

import logging
from pathlib import Path

from src.evaluation.chunk_plotter import plot_chunk_directory


LOGGER = logging.getLogger(__name__)


def run_generate_chunk_plots_use_case(chunks_dir: str | Path) -> dict[str, object]:
    chunks_root = Path(chunks_dir).resolve()

    if not chunks_root.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_root}")
    if not chunks_root.is_dir():
        raise NotADirectoryError(f"Chunks path is not a directory: {chunks_root}")

    chunk_dirs = sorted([p for p in chunks_root.iterdir() if p.is_dir()])
    if not chunk_dirs:
        raise RuntimeError(f"No chunk directories found inside: {chunks_root}")

    total_chunk_dirs = 0
    total_bus_plots = 0
    total_event_plots = 0
    per_chunk_summary: list[dict[str, object]] = []

    for chunk_dir in chunk_dirs:
        LOGGER.info("Processing chunk: %s", chunk_dir.name)
        result = plot_chunk_directory(chunk_dir)

        created_bus_plots = result["bus_plots"]
        event_plot = result["event_plot"]

        bus_plot_count = len(created_bus_plots)
        event_plot_count = 1 if event_plot is not None else 0

        total_chunk_dirs += 1
        total_bus_plots += bus_plot_count
        total_event_plots += event_plot_count

        per_chunk_summary.append(
            {
                "chunk_dir": str(chunk_dir),
                "chunk_name": chunk_dir.name,
                "bus_plot_count": bus_plot_count,
                "event_plot_created": event_plot is not None,
                "event_plot_path": str(event_plot) if event_plot is not None else None,
            }
        )

        LOGGER.info(
            "Chunk %s done. bus_plots=%d event_plot=%s",
            chunk_dir.name,
            bus_plot_count,
            "yes" if event_plot is not None else "no",
        )

    summary = {
        "chunks_dir": str(chunks_root),
        "chunk_count": total_chunk_dirs,
        "total_bus_plots": total_bus_plots,
        "total_event_plots": total_event_plots,
        "per_chunk_summary": per_chunk_summary,
    }

    LOGGER.info(
        "Chunk plotting completed. chunk_count=%d total_bus_plots=%d total_event_plots=%d",
        total_chunk_dirs,
        total_bus_plots,
        total_event_plots,
    )
    return summary