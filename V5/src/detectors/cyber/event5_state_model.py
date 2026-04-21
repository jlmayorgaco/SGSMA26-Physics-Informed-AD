from __future__ import annotations

from dataclasses import dataclass


EVENT5_NORMAL = "NORMAL"
EVENT5_PARTIAL_DROPOUT = "PARTIAL_DROPOUT"
EVENT5_FULL_DROPOUT = "FULL_DROPOUT"
EVENT5_RECOVERY = "RECOVERY"


@dataclass(slots=True)
class Event5StateModel:
    partial_start_threshold: float = 0.22
    full_start_threshold: float = 0.92
    recovery_threshold: float = 0.08
    min_partial_frames: int = 2
    min_full_frames: int = 1
    min_recovery_frames: int = 2

    def run(self, *, partial_ratio: list[float], full_ratio: list[float]) -> tuple[list[str], list[float]]:
        n = min(len(partial_ratio), len(full_ratio))
        states: list[str] = []
        probs: list[float] = []
        state = EVENT5_NORMAL
        partial_count = 0
        full_count = 0
        recovery_count = 0
        for i in range(n):
            partial = float(max(0.0, min(1.0, partial_ratio[i])))
            full = float(max(0.0, min(1.0, full_ratio[i])))
            score = max(full, partial)

            if state == EVENT5_NORMAL:
                if full >= self.full_start_threshold:
                    full_count += 1
                    if full_count >= self.min_full_frames:
                        state = EVENT5_FULL_DROPOUT
                elif partial >= self.partial_start_threshold:
                    partial_count += 1
                    if partial_count >= self.min_partial_frames:
                        state = EVENT5_PARTIAL_DROPOUT
                else:
                    partial_count = 0
                    full_count = 0
            elif state == EVENT5_PARTIAL_DROPOUT:
                if full >= self.full_start_threshold:
                    state = EVENT5_FULL_DROPOUT
                    full_count = 0
                elif score <= self.recovery_threshold:
                    recovery_count += 1
                    if recovery_count >= self.min_recovery_frames:
                        state = EVENT5_RECOVERY
                else:
                    recovery_count = 0
            elif state == EVENT5_FULL_DROPOUT:
                if score <= self.recovery_threshold:
                    recovery_count += 1
                    if recovery_count >= self.min_recovery_frames:
                        state = EVENT5_RECOVERY
                else:
                    recovery_count = 0
            else:
                if score <= self.recovery_threshold:
                    recovery_count += 1
                    if recovery_count >= self.min_recovery_frames:
                        state = EVENT5_NORMAL
                        partial_count = 0
                        full_count = 0
                elif full >= self.full_start_threshold:
                    state = EVENT5_FULL_DROPOUT
                    recovery_count = 0
                elif partial >= self.partial_start_threshold:
                    state = EVENT5_PARTIAL_DROPOUT
                    recovery_count = 0

            states.append(state)
            probs.append(score)
        return states, probs

