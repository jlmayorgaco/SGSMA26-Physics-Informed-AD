from __future__ import annotations

import numpy as np

from src.detectors.postprocessing.event_chunker import EventChunker


def test_event_chunker_builds_intervals() -> None:
    y = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=int)
    ts = np.arange(len(y), dtype=float) * 0.1
    p = np.linspace(0.1, 0.9, len(y))
    chunker = EventChunker(min_chunk_frames=2, pre_padding_frames=1, post_padding_frames=1)
    chunks = chunker.build_chunks(y, ts, p)
    assert len(chunks) == 2
    assert chunks[0]["frame_count"] >= 2

