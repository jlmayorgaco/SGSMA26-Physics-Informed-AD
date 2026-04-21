from __future__ import annotations

import numpy as np

from src.detectors.configs import PostprocessingConfig
from src.detectors.models import EventChunk


def chunk_events(binary: np.ndarray, timestamps: np.ndarray, scores: np.ndarray, config: PostprocessingConfig) -> list[EventChunk]:
    chunks: list[EventChunk] = []
    i = 0
    while i < len(binary):
        if binary[i] == 1:
            j = i
            while j < len(binary) and binary[j] == 1:
                j += 1
            frames = j - i
            if frames >= config.min_chunk_frames:
                seg_scores = scores[i:j]
                start_t = float(timestamps[i]) if i < len(timestamps) else float(i)
                end_t = float(timestamps[j - 1]) if (j - 1) < len(timestamps) else float(j - 1)
                chunks.append(
                    EventChunk(
                        start_time_s=start_t,
                        end_time_s=end_t,
                        duration_s=max(0.0, end_t - start_t),
                        mean_score=float(seg_scores.mean()) if len(seg_scores) else 0.0,
                        max_score=float(seg_scores.max()) if len(seg_scores) else 0.0,
                        frame_count=frames,
                    )
                )
            i = j
        else:
            i += 1
    return chunks

