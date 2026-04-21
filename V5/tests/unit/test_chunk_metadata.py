from __future__ import annotations

import pandas as pd

from src.infrastructure.legacy.m0_adapter import get_chunk_metadata


def test_chunk_metadata_contains_required_fields() -> None:
    event_slice = pd.DataFrame({"Bus10": [0, 1], "Bus19": [0, 0]})
    meta = get_chunk_metadata(event_slice, start_t=0.0, end_t=1.0, order_idx=1, dominant_label=1)

    assert "affected_buses" in meta
    assert "labels_present" in meta
    assert "per_bus_labels" in meta
    assert "category" in meta
    assert "Bus10" in meta["affected_buses"]
    assert meta["per_bus_labels"]["Bus10"] == 1
