from __future__ import annotations

import pandas as pd

from src.infrastructure.legacy.m0_adapter import (
    find_chunk_boundaries,
    get_chunk_metadata,
)


def test_find_chunk_boundaries_constant_normal_event() -> None:
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


def test_find_chunk_boundaries_detects_0_5_0_transition() -> None:
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


def test_find_chunk_boundaries_uses_global_max_across_buses() -> None:
    idx = pd.Index([0.0, 0.1, 0.2, 0.3], name="TIMESTAMP")
    event_df = pd.DataFrame(
        {
            "Bus10": [0, 0, 0, 0],
            "Bus19": [0, 5, 5, 0],
            "Bus22": [0, 0, 0, 0],
        },
        index=idx,
    )

    boundaries, global_state = find_chunk_boundaries(event_df)

    assert global_state.tolist() == [0, 5, 5, 0]
    assert boundaries == [0.0, 0.1, 0.3, 0.3] or boundaries == [0.0, 0.1, 0.3]
    # Si tu implementación siempre agrega el último índice aunque coincida con un boundary,
    # puede salir repetido. Si no, quedará la forma corta.


def test_get_chunk_metadata_for_missing_data_event() -> None:
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
    assert meta["label_name"] == "Missing data"
    assert meta["chunk_type"] == "event"
    assert meta["start_time_s"] == 10.0
    assert meta["end_time_s"] == 10.1
    assert meta["duration_s"] == 0.1
    assert meta["category"] == "Cyber"
    assert meta["affected_buses"] == ["Bus19"]
    assert meta["labels_present"] == [0, 5]
    assert meta["per_bus_labels"] == {
        "Bus10": 0,
        "Bus19": 5,
        "Bus22": 0,
    }


def test_get_chunk_metadata_for_normal_chunk() -> None:
    event_df_slice = pd.DataFrame(
        {
            "Bus10": [0, 0, 0],
            "Bus19": [0, 0, 0],
            "Bus22": [0, 0, 0],
        }
    )

    meta = get_chunk_metadata(
        event_df_slice=event_df_slice,
        start_t=0.0,
        end_t=0.2,
        order_idx=1,
        dominant_label=0,
    )

    assert meta["chunk_order"] == 1
    assert meta["label"] == 0
    assert meta["label_name"] == "Normal operation"
    assert meta["chunk_type"] == "normal"
    assert meta["duration_s"] == 0.2
    assert meta["affected_buses"] == []
    assert meta["labels_present"] == [0]
    assert meta["category"] is None


def test_get_chunk_metadata_for_concurrent_labels_preserves_all_present_labels() -> None:
    event_df_slice = pd.DataFrame(
        {
            "Bus10": [6, 6, 6],
            "Bus19": [0, 0, 0],
            "Bus22": [3, 3, 3],
        }
    )

    meta = get_chunk_metadata(
        event_df_slice=event_df_slice,
        start_t=20.0,
        end_t=20.2,
        order_idx=4,
        dominant_label=6,
    )

    assert meta["label"] == 6
    assert meta["chunk_type"] == "event"
    assert meta["category"] == "Cyber Physical"
    assert sorted(meta["labels_present"]) == [0, 3, 6]
    assert sorted(meta["affected_buses"]) == ["Bus10", "Bus22"]
    assert meta["per_bus_labels"]["Bus10"] == 6
    assert meta["per_bus_labels"]["Bus22"] == 3


def test_get_chunk_metadata_duration_in_minutes_is_consistent() -> None:
    event_df_slice = pd.DataFrame(
        {
            "Bus10": [1, 1, 1],
            "Bus19": [0, 0, 0],
            "Bus22": [0, 0, 0],
        }
    )

    meta = get_chunk_metadata(
        event_df_slice=event_df_slice,
        start_t=120.0,
        end_t=126.0,
        order_idx=3,
        dominant_label=1,
    )

    assert meta["duration_s"] == 6.0
    assert meta["start_time_min"] == 2.0
    assert meta["end_time_min"] == 2.1
    assert meta["duration_min"] == 0.1


def test_get_chunk_metadata_per_bus_labels_uses_buswise_max_label() -> None:
    event_df_slice = pd.DataFrame(
        {
            "Bus10": [0, 1, 0],
            "Bus19": [0, 0, 0],
            "Bus22": [0, 4, 4],
        }
    )

    meta = get_chunk_metadata(
        event_df_slice=event_df_slice,
        start_t=30.0,
        end_t=30.2,
        order_idx=5,
        dominant_label=4,
    )

    assert meta["per_bus_labels"] == {
        "Bus10": 1,
        "Bus19": 0,
        "Bus22": 4,
    }
    assert sorted(meta["affected_buses"]) == ["Bus10", "Bus22"]
    assert sorted(meta["labels_present"]) == [0, 1, 4]