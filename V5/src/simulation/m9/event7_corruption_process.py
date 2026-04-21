"""RAW-informed Event 7 corruption switching process."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.simulation.m9.constants import PMU_MEASUREMENT_SUFFIXES


EVENT7_MODES = ("NORMAL", "SPIKE", "BIAS_DRIFT", "STUCK", "REPLAY_LIKE")
ANGLE_SUFFIXES = tuple(suffix for suffix in PMU_MEASUREMENT_SUFFIXES if suffix.endswith("ANG"))


def _measurement_columns(bus: str) -> list[str]:
    return [f"{bus}_{suffix}" for suffix in PMU_MEASUREMENT_SUFFIXES]


def _channel_family(column: str) -> str:
    upper = str(column).upper()
    if upper.endswith("FREQ"):
        return "frequency"
    if upper.endswith("ROCOF"):
        return "rocof"
    suffix = upper.split("_", maxsplit=1)[-1]
    if suffix.startswith("V"):
        return "voltage"
    if suffix.startswith("I"):
        return "current"
    return "other"


def _wrap_deg(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return ((arr + 180.0) % 360.0) - 180.0


def _state_runs(states: list[str]) -> list[tuple[str, int, int]]:
    if not states:
        return []
    out: list[tuple[str, int, int]] = []
    start = 0
    current = states[0]
    for idx, state in enumerate(states[1:], start=1):
        if state != current:
            out.append((current, start, idx - 1))
            start = idx
            current = state
    out.append((current, start, len(states) - 1))
    return out


@dataclass(slots=True)
class Event7CorruptionProcess:
    """Switching corruption process for Event 7."""

    params: dict[str, Any]
    noise_baseline: dict[str, Any]
    seed: int = 12345
    _rng: np.random.Generator = field(init=False, repr=False)
    _mode_prior: dict[str, Any] = field(init=False, repr=False)
    _pmu: dict[str, Any] = field(init=False, repr=False)
    _family: dict[str, Any] = field(init=False, repr=False)
    _spike_dist: dict[str, Any] = field(init=False, repr=False)
    _spike_width: dict[str, Any] = field(init=False, repr=False)
    _drift_dist: dict[str, Any] = field(init=False, repr=False)
    _stuck_dist: dict[str, Any] = field(init=False, repr=False)
    _replay_dist: dict[str, Any] = field(init=False, repr=False)
    _noise: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._mode_prior = dict(self.params.get("mode_prior", {}))
        self._pmu = dict(self.params.get("pmu_specific_corruption_tendencies", {}))
        self._family = dict(self.params.get("channel_family_corruption_tendencies", {}))
        self._spike_dist = dict(self.params.get("spike_amplitude_distribution", {}))
        self._spike_width = dict(self.params.get("spike_duration_distribution", {}))
        self._drift_dist = dict(self.params.get("drift_slope_distribution", {}))
        self._stuck_dist = dict(self.params.get("stuck_run_length_distribution", {}))
        self._replay_dist = dict(self.params.get("replay_segment_length_distribution", {}))
        self._noise = dict(self.noise_baseline.get("per_pmu_channel", {}))

    def _mode_probabilities(self) -> np.ndarray:
        values = np.asarray(
            [
                float(self._mode_prior.get("SPIKE", 0.30)),
                float(self._mode_prior.get("BIAS_DRIFT", 0.30)),
                float(self._mode_prior.get("STUCK", 0.20)),
                float(self._mode_prior.get("REPLAY_LIKE", 0.20)),
            ],
            dtype=float,
        )
        values = np.clip(values, 1e-9, None)
        values = values / np.sum(values)
        return values

    def _sample_mode_sequence(self, n_frames: int) -> list[str]:
        if n_frames <= 0:
            return []
        probs = self._mode_probabilities()
        modes = ["NORMAL"] * n_frames
        cursor = 0
        while cursor < n_frames:
            base_mode = str(self._rng.choice(["SPIKE", "BIAS_DRIFT", "STUCK", "REPLAY_LIKE"], p=probs))
            if base_mode == "SPIKE":
                base = max(1.0, float(self._spike_width.get("mean", 2.0)))
            elif base_mode == "BIAS_DRIFT":
                base = max(2.0, float(self._drift_dist.get("p50", 6.0)))
            elif base_mode == "STUCK":
                base = max(2.0, float(self._stuck_dist.get("mean", 5.0)))
            else:
                base = max(2.0, float(self._replay_dist.get("mean", 5.0)))
            dwell = int(max(1.0, round(self._rng.normal(base, max(1.0, 0.25 * base)))))
            end = min(n_frames, cursor + dwell)
            for idx in range(cursor, end):
                modes[idx] = base_mode
            cursor = end + int(max(0, round(self._rng.normal(1.5, 0.8))))
        # Guarantee at least one corruption mode appears.
        if all(mode == "NORMAL" for mode in modes):
            anchor = max(0, int(0.40 * n_frames))
            modes[anchor] = "SPIKE"
        return modes

    def _channel_scale(self, *, bus: str, column: str) -> float:
        key = f"{bus}::{column.split('_', maxsplit=1)[-1].upper()}"
        stats = self._noise.get(key, {})
        scale = float(stats.get("robust_sigma", stats.get("std", 1.0)))
        return max(scale, 1e-6)

    def _family_scale(self, column: str) -> float:
        family = _channel_family(column)
        f = self._family.get(family, {})
        jump = float(f.get("jump_rate_mean", 0.05))
        outlier = float(f.get("outlier_rate_mean", 0.01))
        return max(0.25, min(3.0, 1.0 + 2.0 * jump + 3.0 * outlier))

    def _spike_amplitude(self) -> float:
        mean = float(self._spike_dist.get("mean", 5.0))
        p95 = float(self._spike_dist.get("p95", max(mean * 1.8, mean + 1.0)))
        sigma = max(0.25, (p95 - mean) / 2.5)
        return float(max(1.5, self._rng.normal(mean, sigma)))

    def _drift_slope(self) -> float:
        mean = float(self._drift_dist.get("mean", 0.0))
        std = max(1e-5, float(self._drift_dist.get("std", max(abs(mean), 0.03))))
        return float(self._rng.normal(mean, std))

    def apply(
        self,
        *,
        bus: str,
        frame: pd.DataFrame,
        start_idx: int,
        end_idx: int,
        physical_event_mask: np.ndarray | None = None,
        event_label_when_physical: int = 8,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        out = frame.copy()
        cols = [column for column in _measurement_columns(bus) if column in out.columns]
        if not cols:
            return out, []
        local_start = max(0, int(start_idx))
        local_end = min(int(end_idx), len(out) - 1)
        if local_end < local_start:
            return out, []
        n_frames = local_end - local_start + 1
        modes = self._sample_mode_sequence(n_frames)
        local_indices = np.arange(local_start, local_end + 1, dtype=int)
        physical = (
            np.asarray(physical_event_mask, dtype=bool)
            if physical_event_mask is not None
            else np.zeros((len(out),), dtype=bool)
        )

        latent_rows: list[dict[str, Any]] = []
        for mode, run_start, run_end in _state_runs(modes):
            run_indices = local_indices[run_start : run_end + 1]
            if mode == "NORMAL":
                for idx in run_indices:
                    latent_rows.append(
                        {"bus": bus, "frame_index": int(idx), "timestamp": float(out.iloc[int(idx)]["TIMESTAMP"]), "mode": mode}
                    )
                continue

            for column in cols:
                series = pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float)
                scale = self._channel_scale(bus=bus, column=column) * self._family_scale(column)
                if mode == "SPIKE":
                    amplitude = self._spike_amplitude() * scale
                    signs = self._rng.choice(np.asarray([-1.0, 1.0], dtype=float), size=len(run_indices))
                    series[run_indices] = series[run_indices] + signs * amplitude
                elif mode == "BIAS_DRIFT":
                    slope = self._drift_slope() * scale
                    offset = float(self._rng.normal(0.0, 0.45 * scale))
                    drift = np.linspace(0.0, slope * max(1, len(run_indices) - 1), len(run_indices), dtype=float)
                    series[run_indices] = series[run_indices] + offset + drift
                elif mode == "STUCK":
                    anchor = max(0, int(run_indices[0]) - 1)
                    base = float(series[anchor]) if np.isfinite(series[anchor]) else float(np.nanmedian(series))
                    series[run_indices] = base
                elif mode == "REPLAY_LIKE":
                    lag_mean = max(2.0, float(self._replay_dist.get("mean", 5.0)))
                    lag = int(np.clip(round(self._rng.normal(lag_mean, 1.2)), 2, 12))
                    src = np.maximum(run_indices - lag, 0)
                    series[run_indices] = series[src]
                if column.split("_", maxsplit=1)[-1] in ANGLE_SUFFIXES:
                    series[run_indices] = _wrap_deg(series[run_indices])
                out[column] = series

            out.loc[run_indices, "DATA_PRESENT"] = 1
            labels = np.where(physical[run_indices], event_label_when_physical, 7)
            out.loc[run_indices, "Event"] = labels
            for idx in run_indices:
                latent_rows.append(
                    {"bus": bus, "frame_index": int(idx), "timestamp": float(out.iloc[int(idx)]["TIMESTAMP"]), "mode": mode}
                )

        return out, latent_rows
