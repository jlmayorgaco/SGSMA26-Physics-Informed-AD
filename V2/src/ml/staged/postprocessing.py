"""Rule-based guardrails for staged event predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PostProcessConfig:
    """Transparent temporal and label constraints for event segments."""

    max_merge_gap_sec: float = 1.0
    min_duration_by_label: dict[int, float] = field(
        default_factory=lambda: {
            1: 1.0,
            2: 1.0,
            3: 5.0,
            4: 5.0,
            5: 0.75,
            6: 0.75,
            7: 0.50,
            8: 1.0,
        }
    )
    min_windows_by_label: dict[int, int] = field(default_factory=lambda: {7: 2})
    min_confidence_by_label: dict[int, float] = field(default_factory=lambda: {7: 0.45})


class EventPostProcessor:
    """Merge adjacent segments and remove physically implausible micro-events."""

    def __init__(self, config: PostProcessConfig | None = None) -> None:
        self.config = config or PostProcessConfig()

    def process(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged = self._merge_adjacent(sorted(segments, key=lambda item: float(item.get("start_sec", 0.0))))
        return [segment for segment in merged if self._is_valid(segment)]

    def _merge_adjacent(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for segment in segments:
            if not result:
                result.append(dict(segment))
                continue
            previous = result[-1]
            same_kind = int(previous["label"]) == int(segment["label"]) and str(previous["location"]) == str(segment["location"])
            gap = float(segment["start_sec"]) - float(previous["end_sec"])
            if same_kind and gap <= self.config.max_merge_gap_sec:
                self._merge_into(previous, segment)
            else:
                result.append(dict(segment))
        return result

    def _merge_into(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        left_n = int(target.get("window_count", 0))
        right_n = int(source.get("window_count", 0))
        total_windows = left_n + right_n
        target["end_sec"] = max(float(target["end_sec"]), float(source["end_sec"]))
        target["window_count"] = total_windows
        left_conf = float(target.get("confidence_mean", 0.0))
        right_conf = float(source.get("confidence_mean", 0.0))
        target["confidence_mean"] = float((left_conf * left_n + right_conf * right_n) / max(total_windows, 1))
        target["confidence_max"] = max(float(target.get("confidence_max", 0.0)), float(source.get("confidence_max", 0.0)))
        active = {int(item["bus"]): int(item["state"]) for item in target.get("active_buses", [])}
        for item in source.get("active_buses", []):
            active[int(item["bus"])] = max(active.get(int(item["bus"]), 0), int(item["state"]))
        target["active_buses"] = [{"bus": bus, "state": state} for bus, state in sorted(active.items()) if state != 0]

    def _is_valid(self, segment: dict[str, Any]) -> bool:
        label = int(segment.get("label", 0))
        if label == 0:
            return False
        duration = float(segment.get("end_sec", 0.0)) - float(segment.get("start_sec", 0.0))
        if duration < float(self.config.min_duration_by_label.get(label, 0.0)):
            return False
        if int(segment.get("window_count", 0)) < int(self.config.min_windows_by_label.get(label, 1)):
            return False
        if float(segment.get("confidence_mean", 0.0)) < float(self.config.min_confidence_by_label.get(label, 0.0)):
            return False
        return True
