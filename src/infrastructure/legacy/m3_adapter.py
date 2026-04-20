"""Thin adapter around legacy `m3_andes_calibration_raw.py`."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
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


def _legacy_m3():
    workspace_root = Path(__file__).resolve().parents[4]
    root_str = str(workspace_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return import_module("m3_andes_calibration_raw")


def run_raw_event0_calibration(
    raw_chunks_dir: str | None = None,
    raw_profile_file: str | None = None,
    output_dir: str | None = None,
    event_label: int | None = None,
) -> None:
    """Run legacy raw event-0 calibration with optional path/label overrides."""
    m3 = _legacy_m3()
    contexts = []
    if raw_chunks_dir is not None:
        contexts.append(_temporary_attr(m3, "RAW_CHUNKS_DIR", raw_chunks_dir))
    if raw_profile_file is not None:
        contexts.append(_temporary_attr(m3, "RAW_PROFILE_FILE", raw_profile_file))
    if output_dir is not None:
        contexts.append(_temporary_attr(m3, "OUTPUT_DIR", output_dir))
    if event_label is not None:
        contexts.append(_temporary_attr(m3, "EVENT_LABEL", int(event_label)))
    with ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        m3.run_raw_event0_calibration()


def load_signal_specs():
    """Return legacy signal specs."""
    return _legacy_m3().build_signal_specs()


def load_noise_layer(profile_path: str | None = None):
    """Build legacy RawPMUNoiseLayer."""
    m3 = _legacy_m3()
    return m3.RawPMUNoiseLayer(profile_path if profile_path is not None else m3.RAW_PROFILE_FILE)


__all__ = [
    "run_raw_event0_calibration",
    "load_signal_specs",
    "load_noise_layer",
]
