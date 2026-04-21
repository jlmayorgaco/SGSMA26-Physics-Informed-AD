from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Event7ModeModel:
    spike_threshold: float = 0.65
    drift_threshold: float = 0.35
    stuck_threshold: float = 0.60
    replay_threshold: float = 0.50

    def classify(self, *, jump_rate: float, drift_strength: float, stuck_rate: float, replay_similarity: float) -> str:
        if jump_rate >= self.spike_threshold:
            return "SPIKE"
        if stuck_rate >= self.stuck_threshold:
            return "STUCK"
        if replay_similarity >= self.replay_threshold:
            return "REPLAY_LIKE"
        if drift_strength >= self.drift_threshold:
            return "BIAS_DRIFT"
        return "NORMAL"

