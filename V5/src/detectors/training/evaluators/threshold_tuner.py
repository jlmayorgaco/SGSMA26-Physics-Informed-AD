from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.detectors.postprocessing.event_state_machine import EventStateMachine
from src.detectors.postprocessing.alert_smoother import exponential_smooth
from src.detectors.training.evaluators.detector_evaluator import BinaryDetectorEvaluator


@dataclass(slots=True)
class ThresholdTuningConfig:
    thresholds: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
    margins: tuple[float, ...] = (0.05, 0.10)
    min_on_values: tuple[int, ...] = (1, 2)
    min_off_values: tuple[int, ...] = (1, 2)
    warmup_values: tuple[int, ...] = (0, 1, 2)
    quiet_cyber_max_values: tuple[float, ...] = (0.42, 0.46, 0.50)
    quiet_physical_max_values: tuple[float, ...] = (0.20, 0.25, 0.30)
    quiet_abnormal_max_values: tuple[float, ...] = (0.52, 0.58, 0.64)
    quiet_frames_to_reset_values: tuple[int, ...] = (1, 2)
    smoothing_alpha: float = 0.35
    objective_f1_weight: float = 1.0
    objective_recall_weight: float = 0.20
    objective_cyber_recall_weight: float = 0.45
    objective_normal_fp_per_min_weight: float = 0.18
    objective_fp_per_min_weight: float = 0.03
    objective_delay_weight: float = 0.01


class ThresholdTuner:
    def __init__(self, config: ThresholdTuningConfig | None = None) -> None:
        self.config = config or ThresholdTuningConfig()
        self.evaluator = BinaryDetectorEvaluator()

    def sweep(
        self,
        *,
        y_true: np.ndarray,
        p_abnormal: np.ndarray,
        p_cyber: np.ndarray | None,
        p_physical: np.ndarray | None,
        frame_predictions: np.ndarray | None,
        timestamps: np.ndarray,
        scenario_ids: np.ndarray | None,
        frame: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        scores = exponential_smooth(np.asarray(p_abnormal, dtype=float), alpha=self.config.smoothing_alpha)
        cyber_scores = np.asarray(p_cyber, dtype=float) if p_cyber is not None else None
        physical_scores = np.asarray(p_physical, dtype=float) if p_physical is not None else None
        frame_scores = np.asarray(frame_predictions, dtype=int) if frame_predictions is not None else None
        rows: list[dict[str, float | int | None]] = []
        for threshold in self.config.thresholds:
            for margin in self.config.margins:
                stop_threshold = float(threshold - margin)
                if stop_threshold <= 0.0:
                    continue
                for min_on in self.config.min_on_values:
                    for min_off in self.config.min_off_values:
                        for warmup in self.config.warmup_values:
                            for quiet_cyber_max in self.config.quiet_cyber_max_values:
                                for quiet_physical_max in self.config.quiet_physical_max_values:
                                    for quiet_abnormal_max in self.config.quiet_abnormal_max_values:
                                        for quiet_frames_to_reset in self.config.quiet_frames_to_reset_values:
                                            machine = EventStateMachine(
                                                start_threshold=float(threshold),
                                                stop_threshold=stop_threshold,
                                                min_on_frames=int(min_on),
                                                min_off_frames=int(min_off),
                                                warmup_frames=int(warmup),
                                                quiet_cyber_max=float(quiet_cyber_max),
                                                quiet_physical_max=float(quiet_physical_max),
                                                quiet_abnormal_max=float(quiet_abnormal_max),
                                                quiet_frames_to_reset=int(quiet_frames_to_reset),
                                            )
                                            y_pred = np.zeros_like(np.asarray(y_true, dtype=int))
                                            if scenario_ids is not None and len(scenario_ids) == len(scores):
                                                sid_array = np.asarray(scenario_ids)
                                                for sid in pd.unique(sid_array):
                                                    mask = sid_array == sid
                                                    idx = np.where(mask)[0]
                                                    if len(idx) == 0:
                                                        continue
                                                    order = idx[np.argsort(np.asarray(timestamps, dtype=float)[idx])]
                                                    local_pred, _ = machine.run(
                                                        scores[order],
                                                        cyber_probabilities=None if cyber_scores is None else cyber_scores[order],
                                                        physical_probabilities=None if physical_scores is None else physical_scores[order],
                                                        frame_predictions=None if frame_scores is None else frame_scores[order],
                                                    )
                                                    y_pred[order] = local_pred
                                            else:
                                                y_pred, _ = machine.run(
                                                    scores,
                                                    cyber_probabilities=cyber_scores,
                                                    physical_probabilities=physical_scores,
                                                    frame_predictions=frame_scores,
                                                )
                                            metrics = self.evaluator.metric_bundle(
                                                y_true=np.asarray(y_true, dtype=int),
                                                y_pred=y_pred,
                                                y_score=scores,
                                                timestamps=np.asarray(timestamps, dtype=float),
                                                scenario_ids=scenario_ids,
                                            )
                                            normal_fp_per_min = float(metrics["false_positives_per_minute"])
                                            cyber_recall = 0.0
                                            physical_recall = 0.0
                                            concurrent_recall = 0.0
                                            if frame is not None and not frame.empty:
                                                tmp = frame.copy()
                                                tmp["y_pred_stable"] = y_pred
                                                tmp["p_abnormal"] = scores
                                                if cyber_scores is not None:
                                                    tmp["p_cyber"] = cyber_scores
                                                if physical_scores is not None:
                                                    tmp["p_physical"] = physical_scores
                                                familywise = self.evaluator.familywise_metrics(tmp)
                                                normal_fp_per_min = float(
                                                    familywise.get("normal", {}).get("false_positives_per_minute", normal_fp_per_min) or normal_fp_per_min
                                                )
                                                cyber_recall = float(familywise.get("cyber_heavy", {}).get("recall_abnormal", 0.0) or 0.0)
                                                physical_recall = float(familywise.get("physical_heavy", {}).get("recall_abnormal", 0.0) or 0.0)
                                                concurrent_recall = float(familywise.get("concurrent_heavy", {}).get("recall_abnormal", 0.0) or 0.0)
                                            delay = float(metrics["detection_delay_s"]) if metrics["detection_delay_s"] is not None else 0.0
                                            objective = (
                                                self.config.objective_f1_weight * float(metrics["f1_abnormal"])
                                                + self.config.objective_recall_weight * float(metrics["recall_abnormal"])
                                                + self.config.objective_cyber_recall_weight * cyber_recall
                                                - self.config.objective_fp_per_min_weight * float(metrics["false_positives_per_minute"])
                                                - self.config.objective_normal_fp_per_min_weight * normal_fp_per_min
                                                - self.config.objective_delay_weight * delay
                                                - 0.10 * max(0.0, 0.75 - physical_recall)
                                                - 0.08 * max(0.0, 0.70 - concurrent_recall)
                                            )
                                            rows.append(
                                                {
                                                    "threshold": float(threshold),
                                                    "start_threshold": float(threshold),
                                                    "stop_threshold": stop_threshold,
                                                    "margin": float(margin),
                                                    "min_on_frames": int(min_on),
                                                    "min_off_frames": int(min_off),
                                                    "warmup_frames": int(warmup),
                                                    "quiet_cyber_max": float(quiet_cyber_max),
                                                    "quiet_physical_max": float(quiet_physical_max),
                                                    "quiet_abnormal_max": float(quiet_abnormal_max),
                                                    "quiet_frames_to_reset": int(quiet_frames_to_reset),
                                                    "normal_false_positives_per_minute": float(normal_fp_per_min),
                                                    "cyber_heavy_recall": float(cyber_recall),
                                                    "physical_heavy_recall": float(physical_recall),
                                                    "concurrent_heavy_recall": float(concurrent_recall),
                                                    **metrics,
                                                    "objective": float(objective),
                                                }
                                            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(
                [
                    {
                        "threshold": 0.50,
                        "start_threshold": 0.50,
                        "stop_threshold": 0.40,
                        "margin": 0.10,
                        "min_on_frames": 2,
                        "min_off_frames": 2,
                        "warmup_frames": 0,
                        "quiet_cyber_max": 0.46,
                        "quiet_physical_max": 0.28,
                        "quiet_abnormal_max": 0.58,
                        "quiet_frames_to_reset": 1,
                        "f1_abnormal": 0.0,
                        "recall_abnormal": 0.0,
                        "precision_abnormal": 0.0,
                        "balanced_accuracy": 0.0,
                        "false_positives_per_minute": 0.0,
                        "detection_delay_s": None,
                        "objective": 0.0,
                    }
                ]
            )
        return frame.sort_values(
            [
                "objective",
                "normal_false_positives_per_minute",
                "false_positives_per_minute",
                "f1_abnormal",
                "recall_abnormal",
                "precision_abnormal",
                "threshold",
                "min_off_frames",
                "min_on_frames",
            ],
            ascending=[False, True, True, False, False, False, False, False, False],
        ).reset_index(drop=True)

    def select_best(self, sweep: pd.DataFrame) -> dict[str, float | int | None]:
        if sweep.empty:
            return {
                "threshold": 0.50,
                "start_threshold": 0.50,
                "stop_threshold": 0.40,
                "margin": 0.10,
                "min_on_frames": 2,
                "min_off_frames": 2,
                "warmup_frames": 0,
                "quiet_cyber_max": 0.46,
                "quiet_physical_max": 0.28,
                "quiet_abnormal_max": 0.58,
                "quiet_frames_to_reset": 1,
                "objective": 0.0,
            }
        row = sweep.iloc[0].to_dict()
        return {
            "threshold": float(row.get("threshold", 0.50)),
            "start_threshold": float(row.get("start_threshold", row.get("threshold", 0.50))),
            "stop_threshold": float(row.get("stop_threshold", 0.40)),
            "margin": float(row.get("margin", 0.10)),
            "min_on_frames": int(row.get("min_on_frames", 2)),
            "min_off_frames": int(row.get("min_off_frames", 2)),
            "warmup_frames": int(row.get("warmup_frames", 0)),
            "quiet_cyber_max": float(row.get("quiet_cyber_max", 0.46)),
            "quiet_physical_max": float(row.get("quiet_physical_max", 0.28)),
            "quiet_abnormal_max": float(row.get("quiet_abnormal_max", 0.58)),
            "quiet_frames_to_reset": int(row.get("quiet_frames_to_reset", 1)),
            "objective": float(row.get("objective", 0.0)),
        }

    def tune(
        self,
        *,
        y_true: np.ndarray,
        p_abnormal: np.ndarray,
        p_cyber: np.ndarray | None,
        p_physical: np.ndarray | None,
        frame_predictions: np.ndarray | None,
        timestamps: np.ndarray,
        scenario_ids: np.ndarray | None,
        frame: pd.DataFrame | None = None,
    ) -> tuple[dict[str, float | int | None], pd.DataFrame]:
        sweep = self.sweep(
            y_true=y_true,
            p_abnormal=p_abnormal,
            p_cyber=p_cyber,
            p_physical=p_physical,
            frame_predictions=frame_predictions,
            timestamps=timestamps,
            scenario_ids=scenario_ids,
            frame=frame,
        )
        return self.select_best(sweep), sweep
