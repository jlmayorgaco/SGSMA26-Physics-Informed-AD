from __future__ import annotations

from src.data_engineering.normalized_index import build_normalized_chunk_index


def test_build_normalized_chunk_index_contract() -> None:
    chunks = [
        {
            "chunk_id": "chunk01_event_0_normal_operation",
            "chunk_path": "tmp/chunk01_event_0_normal_operation",
            "label": 0,
            "label_name": "Normal operation",
            "duration_s": 0.2,
            "start_time_s": 0.0,
            "end_time_s": 0.2,
            "buses_available": ["Bus10"],
            "bus_columns": {"Bus10": ["DATA_PRESENT", "Event", "BUS10_VA_MAG"]},
        },
        {
            "chunk_id": "chunk02_event_1_fault",
            "chunk_path": "tmp/chunk02_event_1_fault",
            "label": 1,
            "label_name": "Fault",
            "duration_s": 0.2,
            "start_time_s": 0.2,
            "end_time_s": 0.4,
            "buses_available": ["Bus10"],
            "bus_columns": {"Bus10": ["DATA_PRESENT", "Event", "BUS10_VA_MAG"]},
        },
    ]

    payload = build_normalized_chunk_index(chunks)
    assert payload["focus_event_label"] == 0
    assert "event0_chunk_ids" in payload
    assert "chunks" in payload
    assert payload["event0_chunk_ids"] == ["chunk01_event_0_normal_operation"]

    required = {
        "chunk_id",
        "chunk_path",
        "label",
        "label_name",
        "duration_s",
        "start_time_s",
        "end_time_s",
        "buses_available",
        "bus_columns",
    }
    for entry in payload["chunks"]:
        assert required.issubset(set(entry.keys()))
