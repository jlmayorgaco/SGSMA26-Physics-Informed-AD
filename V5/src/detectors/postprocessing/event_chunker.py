from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EventChunker:
    min_chunk_frames: int = 2
    pre_padding_frames: int = 0
    post_padding_frames: int = 0

    def build_chunks(self, binary: np.ndarray, timestamps: np.ndarray, probabilities: np.ndarray) -> list[dict[str, float | int]]:
        y = np.asarray(binary, dtype=int)
        ts = np.asarray(timestamps, dtype=float)
        probs = np.asarray(probabilities, dtype=float)
        chunks: list[dict[str, float | int]] = []
        i = 0
        while i < len(y):
            if y[i] == 1:
                j = i
                while j < len(y) and y[j] == 1:
                    j += 1
                start = max(0, i - self.pre_padding_frames)
                end = min(len(y) - 1, j - 1 + self.post_padding_frames)
                frame_count = end - start + 1
                if frame_count >= self.min_chunk_frames:
                    seg = probs[start : end + 1]
                    chunks.append(
                        {
                            "start_index": int(start),
                            "end_index": int(end),
                            "frame_count": int(frame_count),
                            "start_time_s": float(ts[start]) if len(ts) > start else float(start),
                            "end_time_s": float(ts[end]) if len(ts) > end else float(end),
                            "duration_s": float(max(0.0, (ts[end] - ts[start]) if len(ts) > end else (end - start))),
                            "mean_p_abnormal": float(seg.mean()) if len(seg) else 0.0,
                            "max_p_abnormal": float(seg.max()) if len(seg) else 0.0,
                        }
                    )
                i = j
            else:
                i += 1
        return chunks

