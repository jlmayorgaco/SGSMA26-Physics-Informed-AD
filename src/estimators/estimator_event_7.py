from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.classes.report_models import Event7BusThresholds
from src.helpers.raw_loader import measurement_columns
from src.utils.math_utils import EPS, robust_median, robust_scale, safe_quantile


@dataclass(slots=True)
class Event7EstimatorConfig:
    baseline_event_value: int = 0
    outlier_z_threshold: float = 4.5
    jump_z_threshold: float = 4.5
    stuck_quantile: float = 0.05
    bus_threshold_quantile: float = 0.995
    min_active_buses: int = 1
    max_active_buses: int = 2
    min_frame_score: float = 0.05
    weight_outlier: float = 0.35
    weight_jump: float = 0.45
    weight_stuck: float = 0.20


@dataclass(slots=True)
class Event7EstimatorResult:
    frame_max_score: pd.Series
    frame_prediction: pd.Series
    frame_top_bus: pd.Series
    frame_active_bus_count: pd.Series
    per_bus_score: pd.DataFrame
    per_bus_outlier_ratio: pd.DataFrame
    per_bus_jump_ratio: pd.DataFrame
    per_bus_stuck_flag: pd.DataFrame
    thresholds_by_bus: dict[str, Event7BusThresholds]


@dataclass(slots=True)
class Event7Estimator:
    config: Event7EstimatorConfig = field(default_factory=Event7EstimatorConfig)

    def _baseline_mask(self, frame: pd.DataFrame) -> np.ndarray:
        event = pd.to_numeric(frame["Event"], errors="coerce").to_numpy(dtype=float)
        data_present = pd.to_numeric(frame["DATA_PRESENT"], errors="coerce").to_numpy(dtype=float)
        base = (event == float(self.config.baseline_event_value)) & np.isfinite(data_present) & (data_present > 0.0)
        if int(np.sum(base)) < 100:
            base = np.isfinite(data_present) & (data_present > 0.0)
        return base

    def _per_bus_features(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, Event7BusThresholds]:
        cols = measurement_columns(frame)
        if not cols:
            raise ValueError("Event7 estimator requires PMU measurement columns.")
        values = frame[cols].apply(pd.to_numeric, errors="coerce")
        baseline_mask = self._baseline_mask(frame)

        outlier_hits: list[np.ndarray] = []
        jump_hits: list[np.ndarray] = []
        delta_abs_table: list[np.ndarray] = []

        for col in cols:
            x = values[col].to_numpy(dtype=float)
            baseline_x = x[baseline_mask]
            med = robust_median(baseline_x)
            scale = robust_scale(baseline_x)
            z = np.abs((x - med) / max(scale, EPS))
            outlier_hits.append((z > self.config.outlier_z_threshold).astype(float))

            prepend_value = x[0] if x.size > 0 else np.nan
            dx = np.diff(x, prepend=prepend_value)
            baseline_dx = dx[baseline_mask]
            med_dx = robust_median(baseline_dx)
            scale_dx = robust_scale(baseline_dx)
            z_dx = np.abs((dx - med_dx) / max(scale_dx, EPS))
            jump_hits.append((z_dx > self.config.jump_z_threshold).astype(float))
            delta_abs_table.append(np.abs(dx))

        outlier_ratio = np.nanmean(np.vstack(outlier_hits), axis=0)
        jump_ratio = np.nanmean(np.vstack(jump_hits), axis=0)
        delta_abs_stack = np.vstack(delta_abs_table)
        valid_counts = np.sum(np.isfinite(delta_abs_stack), axis=0).astype(float)
        sums = np.nansum(delta_abs_stack, axis=0)
        delta_abs_mean = np.divide(
            sums,
            valid_counts,
            out=np.full_like(sums, fill_value=np.inf, dtype=float),
            where=valid_counts > 0,
        )
        stuck_threshold = safe_quantile(delta_abs_mean[baseline_mask], q=self.config.stuck_quantile, fallback=0.0)
        stuck_flag = (delta_abs_mean <= stuck_threshold).astype(float)

        composite = (
            self.config.weight_outlier * outlier_ratio
            + self.config.weight_jump * jump_ratio
            + self.config.weight_stuck * stuck_flag
        )

        outlier_q95 = safe_quantile(outlier_ratio[baseline_mask], q=0.95, fallback=0.0)
        jump_q95 = safe_quantile(jump_ratio[baseline_mask], q=0.95, fallback=0.0)
        stuck_q95 = safe_quantile(stuck_flag[baseline_mask], q=0.95, fallback=0.0)
        composite_q995 = safe_quantile(composite[baseline_mask], q=self.config.bus_threshold_quantile, fallback=1.0)
        thresholds = Event7BusThresholds(
            outlier_rate_q95=float(outlier_q95),
            jump_rate_q95=float(jump_q95),
            stuck_flag_q95=float(stuck_q95),
            composite_q995=float(max(composite_q995, self.config.min_frame_score)),
        )

        features = pd.DataFrame(
            {
                "outlier_ratio": outlier_ratio,
                "jump_ratio": jump_ratio,
                "stuck_flag": stuck_flag,
                "composite": composite,
            },
            index=frame.index,
        )
        return features, thresholds

    def estimate(
        self,
        aligned_frames: dict[str, pd.DataFrame],
        event5_prediction: pd.Series | np.ndarray | None = None,
    ) -> Event7EstimatorResult:
        if not aligned_frames:
            raise ValueError("aligned_frames is empty.")
        timeline = next(iter(aligned_frames.values())).index

        per_bus_score: dict[str, pd.Series] = {}
        per_bus_outlier: dict[str, pd.Series] = {}
        per_bus_jump: dict[str, pd.Series] = {}
        per_bus_stuck: dict[str, pd.Series] = {}
        thresholds_by_bus: dict[str, Event7BusThresholds] = {}
        active_flags: dict[str, np.ndarray] = {}

        for bus, frame in sorted(aligned_frames.items()):
            features, thresholds = self._per_bus_features(frame=frame)
            thresholds_by_bus[bus] = thresholds
            per_bus_score[bus] = features["composite"].rename(bus)
            per_bus_outlier[bus] = features["outlier_ratio"].rename(bus)
            per_bus_jump[bus] = features["jump_ratio"].rename(bus)
            per_bus_stuck[bus] = features["stuck_flag"].rename(bus)
            feature_gate = (
                (features["outlier_ratio"].to_numpy(dtype=float) > thresholds.outlier_rate_q95)
                | (features["jump_ratio"].to_numpy(dtype=float) > thresholds.jump_rate_q95)
                | (features["stuck_flag"].to_numpy(dtype=float) > thresholds.stuck_flag_q95)
            )
            score_gate = features["composite"].to_numpy(dtype=float) >= thresholds.composite_q995
            active_flags[bus] = (score_gate & feature_gate).astype(int)

        score_df = pd.DataFrame(per_bus_score, index=timeline)
        outlier_df = pd.DataFrame(per_bus_outlier, index=timeline)
        jump_df = pd.DataFrame(per_bus_jump, index=timeline)
        stuck_df = pd.DataFrame(per_bus_stuck, index=timeline)

        active_df = pd.DataFrame(active_flags, index=timeline).astype(int)
        active_bus_count = active_df.sum(axis=1).astype(int).rename("event7_active_bus_count")
        frame_max_score = score_df.max(axis=1).astype(float).rename("event7_max_score")
        frame_top_bus = score_df.idxmax(axis=1).astype(str).rename("event7_top_bus")

        frame_pred = (
            (active_bus_count >= int(self.config.min_active_buses))
            & (active_bus_count <= int(self.config.max_active_buses))
            & (frame_max_score >= float(self.config.min_frame_score))
        )
        if event5_prediction is not None:
            event5_arr = np.asarray(event5_prediction, dtype=int)
            if event5_arr.shape[0] != frame_pred.shape[0]:
                raise ValueError("event5_prediction length does not match timeline length.")
            frame_pred = frame_pred & (event5_arr == 0)
        frame_prediction = frame_pred.astype(int).rename("pred_event7")

        return Event7EstimatorResult(
            frame_max_score=frame_max_score,
            frame_prediction=frame_prediction,
            frame_top_bus=frame_top_bus,
            frame_active_bus_count=active_bus_count,
            per_bus_score=score_df,
            per_bus_outlier_ratio=outlier_df,
            per_bus_jump_ratio=jump_df,
            per_bus_stuck_flag=stuck_df,
            thresholds_by_bus=thresholds_by_bus,
        )
