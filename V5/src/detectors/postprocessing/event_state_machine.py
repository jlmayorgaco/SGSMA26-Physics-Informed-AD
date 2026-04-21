from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STATE_NORMAL = "NORMAL"
STATE_ABNORMAL_ACTIVE = "ABNORMAL_ACTIVE"
STATE_RECOVERY = "RECOVERY"


@dataclass(slots=True)
class EventStateMachine:
    start_threshold: float = 0.62
    stop_threshold: float = 0.42
    min_on_frames: int = 2
    min_off_frames: int = 2
    warmup_frames: int = 0
    quiet_abnormal_max: float = 0.58
    quiet_cyber_max: float = 0.46
    quiet_physical_max: float = 0.28
    quiet_frames_to_reset: int = 1
    frame_normal_abnormal_max: float = 0.68
    frame_normal_cyber_max: float = 0.45
    frame_normal_frames_to_reset: int = 1

    def run(
        self,
        probabilities: np.ndarray,
        *,
        cyber_probabilities: np.ndarray | None = None,
        physical_probabilities: np.ndarray | None = None,
        frame_predictions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        probs = np.asarray(probabilities, dtype=float)
        n = len(probs)
        if n == 0:
            return np.zeros((0,), dtype=int), []
        cyber = (
            np.asarray(cyber_probabilities, dtype=float)
            if cyber_probabilities is not None
            else np.full((n,), 1.0, dtype=float)
        )
        physical = (
            np.asarray(physical_probabilities, dtype=float)
            if physical_probabilities is not None
            else np.full((n,), 1.0, dtype=float)
        )
        frame_pred = (
            np.asarray(frame_predictions, dtype=int)
            if frame_predictions is not None
            else np.ones((n,), dtype=int)
        )
        if len(cyber) != n or len(physical) != n or len(frame_pred) != n:
            raise ValueError("cyber/physical/frame prediction arrays must match probabilities length")
        binary = np.zeros((n,), dtype=int)
        states: list[str] = []
        state = STATE_NORMAL
        on_count = 0
        off_count = 0
        quiet_count = 0
        frame_normal_count = 0
        for i, p in enumerate(probs):
            quiet = bool(
                p <= self.quiet_abnormal_max
                and cyber[i] <= self.quiet_cyber_max
                and physical[i] <= self.quiet_physical_max
            )
            frame_normal = bool(
                frame_pred[i] <= 0
                and p <= self.frame_normal_abnormal_max
                and cyber[i] <= self.frame_normal_cyber_max
            )
            if i < self.warmup_frames:
                binary[i] = 0
                states.append(state)
                continue
            if state == STATE_NORMAL:
                frame_normal_count = 0
                if p >= self.start_threshold:
                    on_count += 1
                    if on_count >= self.min_on_frames:
                        state = STATE_ABNORMAL_ACTIVE
                        binary[i] = 1
                    else:
                        binary[i] = 0
                else:
                    on_count = 0
                    binary[i] = 0
            elif state == STATE_ABNORMAL_ACTIVE:
                if frame_normal:
                    frame_normal_count += 1
                    quiet_count = 0
                    if frame_normal_count >= self.frame_normal_frames_to_reset:
                        state = STATE_RECOVERY
                        binary[i] = 0
                        off_count = 0
                    else:
                        binary[i] = 1
                elif quiet:
                    frame_normal_count = 0
                    quiet_count += 1
                    if quiet_count >= self.quiet_frames_to_reset:
                        state = STATE_RECOVERY
                        binary[i] = 0
                        off_count = 0
                    else:
                        binary[i] = 1
                else:
                    frame_normal_count = 0
                    quiet_count = 0
                    binary[i] = 1
                if binary[i] == 1 and p <= self.stop_threshold:
                    off_count += 1
                    if off_count >= self.min_off_frames:
                        state = STATE_RECOVERY
                else:
                    off_count = 0
            else:  # RECOVERY
                binary[i] = 0
                quiet_count = 0
                frame_normal_count = 0
                if p >= self.start_threshold:
                    state = STATE_ABNORMAL_ACTIVE
                    binary[i] = 1
                    off_count = 0
                    on_count = self.min_on_frames
                elif p <= self.stop_threshold:
                    off_count += 1
                    if off_count >= self.min_off_frames:
                        state = STATE_NORMAL
                        off_count = 0
                        on_count = 0
            states.append(state)
        return binary, states
