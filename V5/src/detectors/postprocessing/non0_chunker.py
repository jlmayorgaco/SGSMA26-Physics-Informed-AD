from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Non0Chunker:
    min_chunk_frames: int = 2

    def build_chunks(self, y_stable: np.ndarray, timestamps: np.ndarray, p_non0: np.ndarray) -> list[dict[str, float | int]]:
        y = np.asarray(y_stable, dtype=int)
        ts = np.asarray(timestamps, dtype=float)
        p = np.asarray(p_non0, dtype=float)
        chunks: list[dict[str, float | int]] = []
        i = 0
        while i < len(y):
            if y[i] == 0:
                i += 1
                continue
            j = i
            while j < len(y) and y[j] == 1:
                j += 1
            frame_count = j - i
            if frame_count >= self.min_chunk_frames:
                chunks.append(
                    {
                        "start_index": int(i),
                        "end_index": int(j - 1),
                        "frame_count": int(frame_count),
                        "start_time_s": float(ts[i]),
                        "end_time_s": float(ts[j - 1]),
                        "duration_s": float(max(0.0, ts[j - 1] - ts[i])),
                        "mean_p_non0": float(p[i:j].mean()),
                        "max_p_non0": float(p[i:j].max()),
                    }
                )
            i = j
        return chunks

