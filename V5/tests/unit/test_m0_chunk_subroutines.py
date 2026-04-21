from __future__ import annotations

import pandas as pd

from src.data_engineering.chunking import append_sanity_rows, chunk_mask


def test_chunk_mask_include_end_behavior() -> None:
    idx = pd.Index([0.0, 0.1, 0.2])
    mask_open = chunk_mask(idx, start_t=0.0, end_t=0.2, include_end=False)
    mask_closed = chunk_mask(idx, start_t=0.0, end_t=0.2, include_end=True)

    assert mask_open.tolist() == [True, True, False]
    assert mask_closed.tolist() == [True, True, True]


def test_append_sanity_rows_emits_expected_keys() -> None:
    df = pd.DataFrame(
        {
            "BUS10_VA_MAG": [1.0, 2.0],
            "BUS10_ROCOF": [0.1, 0.2],
            "DATA_PRESENT": [1, 1],
            "Event": [0, 0],
        },
        index=[0.0, 0.1],
    )

    rows: list[dict] = []
    append_sanity_rows(
        sanity_rows=rows,
        bus_slice=df,
        bus_name="Bus10",
        chunk_order=1,
        chunk_dir_name="chunk01_event_0_normal_operation",
        dominant_label=0,
    )

    assert len(rows) == 2
    required = {
        "chunk_order",
        "chunk_dir",
        "bus_id",
        "dominant_event_label",
        "signal_name",
        "n_valid",
        "n_nan",
        "mean",
        "std",
        "p50",
    }
    assert required.issubset(rows[0].keys())
