"""RAW-informed Event 5 dropout state process."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES


EVENT5_STATES = ("NORMAL", "PARTIAL_DROPOUT", "FULL_DROPOUT")


def _measurement_columns(bus: str) -> list[str]:
    return [f"{bus}_{suffix}" for suffix in PMU_MEASUREMENT_SUFFIXES]


def _channel_family(channel: str) -> str:
    name = str(channel).upper()
    if name.endswith("FREQ"):
        return "frequency"
    if name.endswith("ROCOF"):
        return "rocof"
    suffix = name.split("_", maxsplit=1)[-1]
    if suffix.startswith("V"):
        return "voltage"
    if suffix.startswith("I"):
        return "current"
    return "other"


def _as_probability_row(raw_row: dict[str, Any] | None) -> np.ndarray:
    if raw_row is None:
        return np.asarray([0.84, 0.10, 0.06], dtype=float)
    values = np.asarray(
        [
            float(raw_row.get("NORMAL", 0.84)),
            float(raw_row.get("PARTIAL_DROPOUT", 0.10)),
            float(raw_row.get("FULL_DROPOUT", 0.06)),
        ],
        dtype=float,
    )
    values = np.clip(values, 1e-9, None)
    values = values / np.sum(values)
    return values


def _state_runs(states: list[str]) -> list[tuple[str, int, int]]:
    if not states:
        return []
    out: list[tuple[str, int, int]] = []
    start = 0
    current = states[0]
    for idx, state in enumerate(states[1:], start=1):
        if state != current:
            out.append((current, start, idx - 1))
            current = state
            start = idx
    out.append((current, start, len(states) - 1))
    return out


@dataclass(slots=True)
class Event5DropoutProcess:
    """Semi-Markov dropout process with PMU-conditioned transitions."""

    params: dict[str, Any]
    seed: int = 12345
    _rng: np.random.Generator = field(init=False, repr=False)
    _matrix: dict[str, Any] = field(init=False, repr=False)
    _pmu: dict[str, Any] = field(init=False, repr=False)
    _families: dict[str, Any] = field(init=False, repr=False)
    _burst: dict[str, Any] = field(init=False, repr=False)
    _inter: dict[str, Any] = field(init=False, repr=False)
    _full_prob: float = field(init=False, repr=False)
    _partial_prob: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._matrix = dict(self.params.get("state_transition_matrix", {}))
        self._pmu = dict(self.params.get("pmu_specific_dropout_tendencies", {}))
        self._families = dict(self.params.get("channel_family_dropout_tendencies", {}))
        self._burst = dict(self.params.get("dropout_burst_length_distribution", {}))
        self._inter = dict(self.params.get("inter_burst_interval_distribution", {}))
        split = dict(self.params.get("full_vs_partial_dropout_fraction", {}))
        full = float(split.get("full", 0.7))
        partial = float(split.get("partial", 0.3))
        total = max(full + partial, 1e-9)
        self._full_prob = full / total
        self._partial_prob = partial / total

    def _dwell_sample(self, state: str, bus: str) -> int:
        if state == "NORMAL":
            pmu = self._pmu.get(bus, {})
            inter_mean = float(pmu.get("interburst_mean_frames", self._inter.get("mean", 12.0)))
            base = max(1.0, inter_mean)
            sampled = int(max(1.0, round(self._rng.normal(base, max(1.0, 0.25 * base)))))
            return sampled
        if state == "FULL_DROPOUT":
            pmu = self._pmu.get(bus, {})
            burst = float(pmu.get("dropout_burst_mean_frames", self._burst.get("mean", 8.0)))
            base = max(1.0, burst)
            sampled = int(max(1.0, round(self._rng.normal(base, max(1.0, 0.30 * base)))))
            return sampled
        # PARTIAL_DROPOUT
        base = max(1.0, float(self._burst.get("p50", self._burst.get("mean", 5.0))))
        sampled = int(max(1.0, round(self._rng.normal(base, max(1.0, 0.25 * base)))))
        return sampled

    def _next_state(self, current: str) -> str:
        row = _as_probability_row(self._matrix.get(current))
        index = int(self._rng.choice(np.arange(3), p=row))
        return EVENT5_STATES[index]

    def _partial_channels(self, *, bus: str, channels: list[str]) -> list[str]:
        family_weights = {
            "voltage": float(self._families.get("voltage", {}).get("nan_fraction_mean", 0.30)),
            "current": float(self._families.get("current", {}).get("nan_fraction_mean", 0.30)),
            "frequency": float(self._families.get("frequency", {}).get("nan_fraction_mean", 0.10)),
            "rocof": float(self._families.get("rocof", {}).get("nan_fraction_mean", 0.15)),
            "other": 0.05,
        }
        grouped: dict[str, list[str]] = {}
        for column in channels:
            grouped.setdefault(_channel_family(column), []).append(column)
        chosen: list[str] = []
        for family, cols in grouped.items():
            probability = np.clip(family_weights.get(family, 0.15), 0.05, 0.95)
            local_pick = [column for column in cols if self._rng.random() <= probability]
            chosen.extend(local_pick)
        if not chosen and channels:
            chosen = [channels[int(self._rng.integers(0, len(channels)))]]
        return chosen

    def sample_states(self, *, n_frames: int, bus: str) -> list[str]:
        if n_frames <= 0:
            return []
        states: list[str] = []
        current = "NORMAL"
        cursor = 0
        while cursor < n_frames:
            # Keep Event 5 dropout behavior bursty by forcing dropout with calibrated probabilities.
            if current == "NORMAL":
                current = str(
                    self._rng.choice(
                        ["NORMAL", "PARTIAL_DROPOUT", "FULL_DROPOUT"],
                        p=np.asarray([0.70, self._partial_prob * 0.30, self._full_prob * 0.30], dtype=float),
                    )
                )
            dwell = self._dwell_sample(current, bus=bus)
            end = min(n_frames, cursor + dwell)
            states.extend([current] * (end - cursor))
            cursor = end
            current = self._next_state(current)
        if not any(state != "NORMAL" for state in states):
            start = max(0, int(0.35 * n_frames))
            width = max(1, int(self._dwell_sample("FULL_DROPOUT", bus)))
            for idx in range(start, min(n_frames, start + width)):
                states[idx] = "FULL_DROPOUT"
        return states

    def apply(
        self,
        *,
        bus: str,
        frame: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        physical_event_mask: np.ndarray | None = None,
        event_label_when_physical: int = 6,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        out = frame.copy()
        channels = [column for column in _measurement_columns(bus) if column in out.columns]
        if not channels:
            return out, []
        local_start = max(0, int(start_idx))
        local_end = min(int(end_idx), len(out) - 1)
        if local_end < local_start:
            return out, []

        segment_size = local_end - local_start + 1
        states = self.sample_states(n_frames=segment_size, bus=bus)
        global_mask = np.zeros((len(out),), dtype=bool)
        global_mask[local_start : local_end + 1] = True
        local_indices = np.arange(local_start, local_end + 1, dtype=int)
        physical = (
            np.asarray(physical_event_mask, dtype=bool)
            if physical_event_mask is not None
            else np.zeros((len(out),), dtype=bool)
        )

        latent_rows: list[dict[str, Any]] = []
        for run_state, run_start, run_end in _state_runs(states):
            run_indices = local_indices[run_start : run_end + 1]
            if run_state == "FULL_DROPOUT":
                out.loc[run_indices, channels] = np.nan
                out.loc[run_indices, "DATA_PRESENT"] = 0
                event_values = np.where(physical[run_indices], event_label_when_physical, 5)
                out.loc[run_indices, "Event"] = event_values
            elif run_state == "PARTIAL_DROPOUT":
                partial_cols = self._partial_channels(bus=bus, channels=channels)
                if partial_cols:
                    out.loc[run_indices, partial_cols] = np.nan
                # Partial dropout can keep packet presence while some channels are missing.
                if self._rng.random() < 0.20:
                    out.loc[run_indices, "DATA_PRESENT"] = 0
                else:
                    out.loc[run_indices, "DATA_PRESENT"] = 1
                event_values = np.where(physical[run_indices], event_label_when_physical, 5)
                out.loc[run_indices, "Event"] = event_values
            else:
                # NORMAL: keep baseline values unchanged.
                pass

            for idx in run_indices:
                latent_rows.append(
                    {
                        "bus": bus,
                        "frame_index": int(idx),
                        "timestamp": float(out.iloc[int(idx)]["TIMESTAMP"]),
                        "state": run_state,
                    }
                )

        return out, latent_rows
