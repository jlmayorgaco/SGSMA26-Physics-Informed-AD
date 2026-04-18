from __future__ import annotations

import pandas as pd

from src.infrastructure.legacy.m0_adapter import find_chunk_boundaries


def test_constant_label_yields_single_block() -> None:
    event_df = pd.DataFrame({"Bus10": [0, 0, 0]}, index=[0.0, 0.1, 0.2])
    boundaries, global_state = find_chunk_boundaries(event_df)
    assert boundaries == [0.0, 0.2]
    assert global_state.tolist() == [0, 0, 0]


def test_label_transition_0_1_0_boundaries() -> None:
    event_df = pd.DataFrame({"Bus10": [0, 0, 1, 1, 0]}, index=[0.0, 0.1, 0.2, 0.3, 0.4])
    boundaries, _ = find_chunk_boundaries(event_df)
    assert boundaries == [0.0, 0.2, 0.4, 0.4]
