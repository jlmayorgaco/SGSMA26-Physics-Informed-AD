"""Helpers for normalized chunk index payload."""

from __future__ import annotations

from typing import Any


def build_normalized_chunk_index(chunk_index_entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Build normalized chunk index contract."""
    event0_chunk_ids = [
        entry["chunk_id"] for entry in chunk_index_entries if int(entry.get("label", -1)) == 0
    ]
    return {
        "focus_event_label": 0,
        "event0_chunk_ids": event0_chunk_ids,
        "chunks": chunk_index_entries,
    }
