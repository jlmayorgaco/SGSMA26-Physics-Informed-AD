"""Physics-aware localization scoring for staged PMU inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ml.models.ieee39_physics import ALL_BUSES, PMU_BUSES, bus_index, electrical_distance_matrix, finite_or_zero


@dataclass(frozen=True)
class CandidateScore:
    bus: int
    score: float
    ybus_score: float
    state_score: float
    zbus_prior: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "bus": int(self.bus),
            "score": float(self.score),
            "ybus_score": float(self.ybus_score),
            "state_score": float(self.state_score),
            "zbus_prior": float(self.zbus_prior),
        }


class PhysicsAwareLocalizer:
    """Rank candidate fault buses using ML state output plus Ybus/Zbus evidence."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = int(top_k)

    def rank_segment(
        self,
        segment: dict[str, Any],
        centers: np.ndarray,
        features: pd.DataFrame,
        bus_states: np.ndarray,
    ) -> list[dict[str, float | int]]:
        mask = (centers >= float(segment["start_sec"])) & (centers <= float(segment["end_sec"]))
        if not np.any(mask):
            return []
        ybus = self._bus_ybus_scores(features.loc[mask])
        state = self._bus_state_scores(bus_states[mask])
        zbus = self._zbus_prior(features.loc[mask])
        combined = 0.45 * ybus + 0.35 * state + 0.20 * zbus
        candidates = [
            CandidateScore(
                bus=bus,
                score=float(combined[bus_index(bus)]),
                ybus_score=float(ybus[bus_index(bus)]),
                state_score=float(state[bus_index(bus)]),
                zbus_prior=float(zbus[bus_index(bus)]),
            )
            for bus in ALL_BUSES
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        return [candidate.to_dict() for candidate in candidates[: self.top_k]]

    def _bus_ybus_scores(self, features: pd.DataFrame) -> np.ndarray:
        values = np.array(
            [
                float(pd.to_numeric(features.get(f"phys_bus_{bus:02d}_ybus_residual", 0.0), errors="coerce").mean())
                for bus in ALL_BUSES
            ],
            dtype=float,
        )
        return self._normalize(values)

    def _bus_state_scores(self, bus_states: np.ndarray) -> np.ndarray:
        states = np.asarray(bus_states, dtype=int)
        if states.size == 0:
            return np.zeros(len(ALL_BUSES), dtype=float)
        active = states != 0
        top_label = int(np.nanmax(states)) if states.size else 0
        weights = np.where(states == top_label, 1.0, active.astype(float) * 0.65)
        return finite_or_zero(np.mean(weights, axis=0))

    def _zbus_prior(self, features: pd.DataFrame) -> np.ndarray:
        pmu_scores = []
        for bus in PMU_BUSES:
            score = 0.0
            for name in [
                f"phys_pmu_{bus:02d}_kcl_residual",
                f"BUS{bus}_missing_fraction",
                f"BUS{bus}_VA_MAG_relative_max_abs_delta",
                f"BUS{bus}_IA_MAG_relative_max_abs_delta",
                f"BUS{bus}_ROCOF_max_abs_nominal_dev",
            ]:
                if name in features:
                    score += float(pd.to_numeric(features[name], errors="coerce").mean())
            pmu_scores.append(score)
        pmu_scores = self._normalize(np.asarray(pmu_scores, dtype=float))
        distances = electrical_distance_matrix()[:, [bus_index(bus) for bus in PMU_BUSES]]
        scale = max(float(np.nanmedian(distances)), 1e-6)
        prior = np.exp(-distances / scale) @ pmu_scores
        return self._normalize(prior)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        arr = finite_or_zero(values)
        low = float(np.nanmin(arr)) if arr.size else 0.0
        high = float(np.nanmax(arr)) if arr.size else 0.0
        if high - low <= 1e-12:
            return np.zeros_like(arr, dtype=float)
        return (arr - low) / (high - low)
