from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Non0StateMachine:
    start_threshold: float = 0.74
    stop_threshold: float = 0.52
    min_on_frames: int = 3
    min_off_frames: int = 4
    warmup_frames: int = 2
    quiet_reset_threshold: float = 0.45
    quiet_reset_frames: int = 3

    def run(
        self,
        probabilities: np.ndarray,
        *,
        p_event5: np.ndarray | None = None,
        p_event7: np.ndarray | None = None,
        p_physical: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        p = np.asarray(probabilities, dtype=float)
        y = np.zeros((len(p),), dtype=int)
        e5 = np.asarray(p_event5, dtype=float) if p_event5 is not None else np.zeros_like(p)
        e7 = np.asarray(p_event7, dtype=float) if p_event7 is not None else np.zeros_like(p)
        phy = np.asarray(p_physical, dtype=float) if p_physical is not None else np.zeros_like(p)
        states: list[str] = []
        on_count = 0
        off_count = 0
        quiet_count = 0
        active = False
        for i, score in enumerate(p):
            if i < self.warmup_frames:
                states.append("WARMUP")
                continue
            quiet = bool(score <= self.quiet_reset_threshold and e5[i] <= self.quiet_reset_threshold and e7[i] <= self.quiet_reset_threshold and phy[i] <= self.quiet_reset_threshold)
            if not active:
                if score >= self.start_threshold:
                    on_count += 1
                    if on_count >= self.min_on_frames:
                        active = True
                        y[i] = 1
                        quiet_count = 0
                else:
                    on_count = 0
            else:
                y[i] = 1
                if quiet:
                    quiet_count += 1
                else:
                    quiet_count = 0
                if quiet_count >= self.quiet_reset_frames:
                    active = False
                    y[i] = 0
                    on_count = 0
                    off_count = 0
                    quiet_count = 0
                elif score <= self.stop_threshold:
                    off_count += 1
                    if off_count >= self.min_off_frames:
                        active = False
                        y[i] = 0
                        on_count = 0
                        off_count = 0
                else:
                    off_count = 0
            states.append("ABNORMAL_ACTIVE" if active else "NORMAL")
        return y, states
