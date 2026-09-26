"""Thin adapter around legacy `m1_preprocessing.py`."""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from importlib import import_module
from pathlib import Path
from typing import Any, Iterator
import sys


@contextmanager
def _temporary_attr(obj: Any, attr: str, value: Any) -> Iterator[None]:
    had_attr = hasattr(obj, attr)
    old_value = getattr(obj, attr, None)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        if had_attr:
            setattr(obj, attr, old_value)
        else:
            delattr(obj, attr)


def _legacy_m1():
    workspace_root = Path(__file__).resolve().parents[4]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return import_module("m1_preprocessing")


def run_normalization_pipeline(
    input_dir: str | None = None,
    output_dir: str | None = None,
    generate_plots: bool = False,
) -> None:
    """Run legacy normalization pipeline with optional path overrides."""
    m1 = _legacy_m1()

    if input_dir is None and output_dir is None:
        m1.process_normalized_pipeline(generate_plots=generate_plots)
        return

    contexts = []
    if input_dir is not None:
        contexts.append(_temporary_attr(m1, "INPUT_DIR", input_dir))
    if output_dir is not None:
        contexts.extend(
            [
                _temporary_attr(m1, "OUTPUT_DIR", output_dir),
                _temporary_attr(m1, "CHUNKS_DIR", str(Path(output_dir) / "chunks")),
                _temporary_attr(m1, "BASELINES_CSV", str(Path(output_dir) / "normalization_baselines.csv")),
                _temporary_attr(m1, "CHUNK_INDEX_JSON", str(Path(output_dir) / "normalized_chunk_index.json")),
            ]
        )

    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        m1.process_normalized_pipeline(generate_plots=generate_plots)


def normalize_bus_data(bus_data: dict, event_df):
    """Proxy for legacy normalization internals."""
    return _legacy_m1().normalize_bus_data(bus_data, event_df)


def load_and_synchronize_data(input_dir: str):
    """Proxy for legacy raw loading + synchronization."""
    return _legacy_m1().load_and_synchronize_data(input_dir)


__all__ = [
    "run_normalization_pipeline",
    "load_and_synchronize_data",
    "normalize_bus_data",
]
