from __future__ import annotations

import pandas as pd

from src.infrastructure.legacy.m0_adapter import get_chunk_metadata


def test_get_chunk_metadata_missing_data_case() -> None:
    event_df_slice = pd.DataFrame(
        {
            "Bus10": [0, 0, 0],
            "Bus19": [5, 5, 5],
            "Bus22": [0, 0, 0],
        }
    )

    meta = get_chunk_metadata(
        event_df_slice=event_df_slice,
        start_t=10.0,
        end_t=10.1,
        order_idx=2,
        dominant_label=5,
    )

    assert meta["chunk_order"] == 2
    assert meta["label"] == 5
    assert meta["chunk_type"] == "event"
    assert meta["category"] == "Cyber"
    assert meta["affected_buses"] == ["Bus19"]
    assert meta["per_bus_labels"]["Bus19"] == 5
    assert sorted(meta["labels_present"]) == [0, 5]
    assert meta["duration_s"] == 0.1