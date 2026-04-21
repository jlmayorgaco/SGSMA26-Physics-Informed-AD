from __future__ import annotations

import pandas as pd

from src.infrastructure.legacy.m0_adapter import find_chunk_boundaries


def test_find_chunk_boundaries_constant_event() -> None:
    idx = pd.Index([0.0, 0.1, 0.2, 0.3], name="TIMESTAMP")
    event_df = pd.DataFrame(
        {
            "Bus10": [0, 0, 0, 0],
            "Bus19": [0, 0, 0, 0],
            "Bus22": [0, 0, 0, 0],
        },
        index=idx,
    )

    boundaries, global_state = find_chunk_boundaries(event_df)

    assert boundaries == [0.0, 0.3]
    assert global_state.tolist() == [0, 0, 0, 0]


def test_find_chunk_boundaries_transition_0_5_0() -> None:
    idx = pd.Index([0.0, 0.1, 0.2, 0.3, 0.4], name="TIMESTAMP")
    event_df = pd.DataFrame(
        {
            "Bus10": [0, 0, 5, 0, 0],
            "Bus19": [0, 0, 0, 0, 0],
            "Bus22": [0, 0, 0, 0, 0],
        },
        index=idx,
    )

    boundaries, global_state = find_chunk_boundaries(event_df)

    assert boundaries == [0.0, 0.2, 0.3, 0.4]
    assert global_state.tolist() == [0, 0, 5, 0, 0]