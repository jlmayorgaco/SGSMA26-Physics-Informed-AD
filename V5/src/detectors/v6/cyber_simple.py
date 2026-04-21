from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_META_COLUMNS = {"TIMESTAMP", "DATA_PRESENT", "Event", "EVENT"}
_EPS = 1e-9


@dataclass(slots=True)
class CyberV6Config:
    event5_missing_threshold: float = 0.80
    normal_quantile: float = 0.95
    threshold_margin: float = 1.35
    outlier_z_threshold: float = 4.5
    jump_z_threshold: float = 4.5
    replay_tolerance_scale: float = 0.03
    replay_lags: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12)
    event7_min_bus_score: float = 0.90
    event7_min_feature_hits: int = 1
    event7_max_active_buses: int = 2


@dataclass(slots=True)
class CyberV6Thresholds:
    outlier_rate: float
    jump_rate: float
    stuck_rate: float
    replay_rate: float


@dataclass(slots=True)
class Event5Decision:
    predicted: bool
    chunk_score: float
    per_bus_score: dict[str, float]
    missing_buses: list[str]


@dataclass(slots=True)
class BusCorruptionScores:
    bus: str
    outlier_rate: float
    jump_rate: float
    stuck_rate: float
    replay_rate: float
    feature_hits: int
    composite_score: float


@dataclass(slots=True)
class Event7Decision:
    predicted: bool
    top_bus: str | None
    top_score: float
    active_buses: list[str]
    thresholds: CyberV6Thresholds
    per_bus: list[BusCorruptionScores]


@dataclass(slots=True)
class CyberChunkDecision:
    predicted_label: int
    event5: Event5Decision
    event7: Event7Decision


@dataclass(slots=True)
class CyberV6Detector:
    config: CyberV6Config = field(default_factory=CyberV6Config)
    thresholds: CyberV6Thresholds = field(
        default_factory=lambda: CyberV6Thresholds(
            outlier_rate=0.08,
            jump_rate=0.08,
            stuck_rate=0.35,
            replay_rate=0.35,
        )
    )

    @staticmethod
    def load_chunk_frames(chunk_dir: Path) -> dict[str, pd.DataFrame]:
        if not chunk_dir.exists() or not chunk_dir.is_dir():
            raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")
        frames: dict[str, pd.DataFrame] = {}
        for csv_path in sorted(chunk_dir.glob("Bus*.csv")):
            bus = csv_path.stem
            frames[bus] = pd.read_csv(csv_path)
        if not frames:
            raise ValueError(f"No bus CSV files found under {chunk_dir}")
        return frames

    @staticmethod
    def _measurement_columns(frame: pd.DataFrame) -> list[str]:
        return [column for column in frame.columns if column not in _META_COLUMNS]

    @staticmethod
    def _robust_scale(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return 1.0
        med = float(np.nanmedian(finite))
        mad = float(np.nanmedian(np.abs(finite - med)))
        return max(1.4826 * mad, _EPS)

    @staticmethod
    def _safe_quantile(values: Iterable[float], q: float, fallback: float) -> float:
        arr = np.asarray([float(v) for v in values], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return fallback
        return float(np.nanquantile(arr, q))

    def _compute_channel_corruption_features(self, series: np.ndarray) -> tuple[float, float, float, float]:
        x = np.asarray(series, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 4:
            return 0.0, 0.0, 0.0, 0.0

        scale_x = self._robust_scale(x)
        med_x = float(np.nanmedian(x))
        abs_z = np.abs((x - med_x) / scale_x)
        outlier_rate = float(np.mean(abs_z > self.config.outlier_z_threshold))

        dx = np.diff(x)
        if dx.size < 3:
            return outlier_rate, 0.0, 0.0, 0.0

        scale_dx = self._robust_scale(dx)
        med_dx = float(np.nanmedian(dx))
        jump_z = np.abs((dx - med_dx) / scale_dx)
        jump_rate = float(np.mean(jump_z > self.config.jump_z_threshold))

        stuck_tol = max(scale_dx * 0.08, _EPS)
        stuck_rate = float(np.mean(np.abs(dx) <= stuck_tol))

        replay_rates: list[float] = []
        for lag in self.config.replay_lags:
            if lag <= 0 or lag >= x.size:
                continue
            delta = np.abs(x[lag:] - x[:-lag])
            tol = max(self.config.replay_tolerance_scale * scale_x, _EPS)
            replay_rates.append(float(np.mean(delta <= tol)))
        replay_rate = float(max(replay_rates) if replay_rates else 0.0)

        return outlier_rate, jump_rate, stuck_rate, replay_rate

    def _compute_bus_features(self, bus: str, frame: pd.DataFrame) -> tuple[float, float, float, float]:
        measurement_columns = self._measurement_columns(frame)
        if not measurement_columns:
            return 0.0, 0.0, 0.0, 0.0
        outlier_rates: list[float] = []
        jump_rates: list[float] = []
        stuck_rates: list[float] = []
        replay_rates: list[float] = []
        for column in measurement_columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            outlier_rate, jump_rate, stuck_rate, replay_rate = self._compute_channel_corruption_features(values)
            outlier_rates.append(outlier_rate)
            jump_rates.append(jump_rate)
            stuck_rates.append(stuck_rate)
            replay_rates.append(replay_rate)
        q = 0.80
        return (
            float(np.nanquantile(outlier_rates, q)) if outlier_rates else 0.0,
            float(np.nanquantile(jump_rates, q)) if jump_rates else 0.0,
            float(np.nanquantile(stuck_rates, q)) if stuck_rates else 0.0,
            float(np.nanquantile(replay_rates, q)) if replay_rates else 0.0,
        )

    def fit_event7_thresholds(self, normal_chunks: Iterable[dict[str, pd.DataFrame]]) -> CyberV6Thresholds:
        outlier_values: list[float] = []
        jump_values: list[float] = []
        stuck_values: list[float] = []
        replay_values: list[float] = []

        for chunk in normal_chunks:
            for bus, frame in chunk.items():
                outlier_rate, jump_rate, stuck_rate, replay_rate = self._compute_bus_features(bus=bus, frame=frame)
                outlier_values.append(outlier_rate)
                jump_values.append(jump_rate)
                stuck_values.append(stuck_rate)
                replay_values.append(replay_rate)

        def _threshold(values: list[float], floor: float) -> float:
            if not values:
                return floor
            arr = np.asarray(values, dtype=float)
            q = float(np.nanquantile(arr, self.config.normal_quantile))
            med = float(np.nanmedian(arr))
            mad = float(np.nanmedian(np.abs(arr - med)))
            robust = med + self.config.threshold_margin * 1.4826 * max(mad, _EPS)
            return max(floor, q, robust)

        self.thresholds = CyberV6Thresholds(
            outlier_rate=_threshold(outlier_values, floor=0.01),
            jump_rate=_threshold(jump_values, floor=0.01),
            stuck_rate=_threshold(stuck_values, floor=0.10),
            replay_rate=_threshold(replay_values, floor=0.10),
        )
        return self.thresholds

    def detect_event5(self, chunk_frames: dict[str, pd.DataFrame]) -> Event5Decision:
        per_bus: dict[str, float] = {}
        for bus, frame in chunk_frames.items():
            measurement_columns = self._measurement_columns(frame)
            if not measurement_columns:
                per_bus[bus] = 0.0
                continue
            bus_nan_ratio = float(frame[measurement_columns].isna().to_numpy(dtype=float).mean())
            if "DATA_PRESENT" in frame.columns:
                dp = pd.to_numeric(frame["DATA_PRESENT"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
                bus_data_missing = float(np.mean(dp <= 0.0))
            else:
                bus_data_missing = 0.0
            per_bus[bus] = max(bus_nan_ratio, bus_data_missing)
        chunk_score = max(per_bus.values()) if per_bus else 0.0
        missing_buses = [bus for bus, score in per_bus.items() if score >= self.config.event5_missing_threshold]
        return Event5Decision(
            predicted=chunk_score >= self.config.event5_missing_threshold,
            chunk_score=float(chunk_score),
            per_bus_score=per_bus,
            missing_buses=sorted(missing_buses),
        )

    def detect_event7(self, chunk_frames: dict[str, pd.DataFrame], *, thresholds: CyberV6Thresholds | None = None) -> Event7Decision:
        active_thresholds = thresholds or self.thresholds
        per_bus_scores: list[BusCorruptionScores] = []
        for bus, frame in sorted(chunk_frames.items()):
            outlier_rate, jump_rate, stuck_rate, replay_rate = self._compute_bus_features(bus=bus, frame=frame)
            normalized_primary = np.asarray(
                [
                    outlier_rate / max(active_thresholds.outlier_rate, _EPS),
                    jump_rate / max(active_thresholds.jump_rate, _EPS),
                    stuck_rate / max(active_thresholds.stuck_rate, _EPS),
                ],
                dtype=float,
            )
            replay_norm = replay_rate / max(active_thresholds.replay_rate, _EPS)
            feature_hits = int(np.sum(normalized_primary >= 1.0))
            composite = float(max(float(np.max(normalized_primary)), 0.85 * replay_norm))
            per_bus_scores.append(
                BusCorruptionScores(
                    bus=bus,
                    outlier_rate=outlier_rate,
                    jump_rate=jump_rate,
                    stuck_rate=stuck_rate,
                    replay_rate=replay_rate,
                    feature_hits=feature_hits,
                    composite_score=composite,
                )
            )
        per_bus_scores.sort(key=lambda row: (-row.composite_score, row.bus))
        top_bus = per_bus_scores[0].bus if per_bus_scores else None
        top_score = float(per_bus_scores[0].composite_score) if per_bus_scores else 0.0
        active_buses = [
            row.bus
            for row in per_bus_scores
            if row.composite_score >= self.config.event7_min_bus_score and row.feature_hits >= self.config.event7_min_feature_hits
        ]
        predicted = 1 <= len(active_buses) <= self.config.event7_max_active_buses
        return Event7Decision(
            predicted=predicted,
            top_bus=top_bus,
            top_score=top_score,
            active_buses=active_buses,
            thresholds=active_thresholds,
            per_bus=per_bus_scores,
        )

    def classify_chunk(self, chunk_frames: dict[str, pd.DataFrame], *, thresholds: CyberV6Thresholds | None = None) -> CyberChunkDecision:
        event5 = self.detect_event5(chunk_frames)
        if event5.predicted:
            event7 = Event7Decision(
                predicted=False,
                top_bus=event5.missing_buses[0] if event5.missing_buses else None,
                top_score=0.0,
                active_buses=[],
                thresholds=thresholds or self.thresholds,
                per_bus=[],
            )
            return CyberChunkDecision(predicted_label=5, event5=event5, event7=event7)

        event7 = self.detect_event7(chunk_frames, thresholds=thresholds)
        label = 7 if event7.predicted else 0
        return CyberChunkDecision(predicted_label=label, event5=event5, event7=event7)
