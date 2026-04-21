from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import pickle
from typing import Sequence

import numpy as np
import pandas as pd

from src.detectors.cyber.services.cyber_branch_service import CyberBranchService
from src.detectors.domain.interfaces.detector import Detector
from src.detectors.domain.models.detection_input import DetectionInput
from src.detectors.domain.models.detection_output import DetectionOutput
from src.detectors.domain.models.detector_config import DetectorConfigV2
from src.detectors.domain.models.window_sample import WindowSample
from src.detectors.fusion.services.detector_fusion_service import DetectorFusionService
from src.detectors.physical.services.physical_branch_service import PhysicalBranchService
from src.detectors.postprocessing import EventChunker, EventStateMachine, exponential_smooth
from src.detectors.training.evaluators.calibration_models import ProbabilityCalibrator


class HybridEventDetector(Detector):
    def __init__(self, config: DetectorConfigV2 | None = None) -> None:
        self.config = config or DetectorConfigV2()
        self.cyber = CyberBranchService()
        self.physical = PhysicalBranchService()
        self.fusion = DetectorFusionService(decision_threshold=self.config.fusion.decision_threshold)
        self.state_machine = EventStateMachine(
            start_threshold=self.config.postprocessing.start_threshold,
            stop_threshold=self.config.postprocessing.stop_threshold,
            min_on_frames=self.config.postprocessing.min_on_frames,
            min_off_frames=self.config.postprocessing.min_off_frames,
            warmup_frames=self.config.postprocessing.warmup_frames,
            quiet_abnormal_max=self.config.postprocessing.quiet_abnormal_max,
            quiet_cyber_max=self.config.postprocessing.quiet_cyber_max,
            quiet_physical_max=self.config.postprocessing.quiet_physical_max,
            quiet_frames_to_reset=self.config.postprocessing.quiet_frames_to_reset,
            frame_normal_abnormal_max=self.config.postprocessing.frame_normal_abnormal_max,
            frame_normal_cyber_max=self.config.postprocessing.frame_normal_cyber_max,
            frame_normal_frames_to_reset=self.config.postprocessing.frame_normal_frames_to_reset,
        )
        self.chunker = EventChunker(min_chunk_frames=self.config.postprocessing.min_chunk_frames)
        self.feature_names: list[str] = []
        self.preprocessing_state: object | None = None
        self.calibrator: ProbabilityCalibrator | None = None

    def set_preprocessing_state(self, state: object, feature_columns: list[str]) -> None:
        self.preprocessing_state = state
        self.feature_names = list(feature_columns)

    def fit(self, samples: Sequence[WindowSample], config: DetectorConfigV2 | None = None) -> None:
        if config is not None:
            self.config = config
        if not samples:
            return
        x = np.stack([s.x for s in samples], axis=0)
        y = np.asarray([s.y_binary for s in samples], dtype=int)
        timestamps = np.asarray([s.center_timestamp for s in samples], dtype=float)
        feature_count = x.shape[2]
        feature_names = self.feature_names if self.feature_names else [f"f{i}" for i in range(feature_count)]
        metadata = pd.DataFrame(
            [
                {
                    "scenario_id": s.scenario_id,
                    "split": s.split,
                    "y_binary": s.y_binary,
                    "event_coarse": s.event_coarse,
                    "event_family": s.event_family.value,
                    "is_cyber_event": s.is_cyber_event,
                    "is_physical_event": s.is_physical_event,
                    "is_concurrent_event": s.is_concurrent_event,
                    "data_present_ratio": s.data_present_ratio,
                    **s.metadata,
                }
                for s in samples
            ]
        )
        inputs = DetectionInput(
            scenario_id=str(samples[0].scenario_id),
            split=str(samples[0].split),
            x_windows=x,
            timestamps=timestamps,
            feature_names=feature_names,
            metadata=metadata,
        )
        metadata["y_binary"] = y
        self.cyber.fit(inputs)
        self.physical.fit(inputs)

    def fit_inputs(self, train_input: DetectionInput) -> None:
        self.feature_names = list(train_input.feature_names)
        self.cyber.fit(train_input)
        self.physical.fit(train_input)

    def predict(self, inputs: DetectionInput) -> DetectionOutput:
        cyber_score = self.cyber.score(inputs)
        physical_score = self.physical.score(inputs)
        fused = self.fusion.fuse(cyber_score, physical_score)
        calibrated = self.calibrator.transform(fused.p_abnormal) if self.calibrator is not None else fused.p_abnormal
        smoothed = np.zeros_like(calibrated, dtype=float)
        stable_binary = np.zeros((len(calibrated),), dtype=int)
        states: list[str] = ["NORMAL"] * len(calibrated)
        chunks: list[dict[str, object]] = []

        if "scenario_id" in inputs.metadata.columns:
            scenario_series = inputs.metadata["scenario_id"].astype(str)
            for scenario_id in scenario_series.dropna().unique().tolist():
                mask = scenario_series.eq(str(scenario_id)).to_numpy()
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue
                order = idx[np.argsort(inputs.timestamps[idx])]
                local_probs = calibrated[order]
                local_cyber = fused.p_cyber[order]
                local_physical = fused.p_physical[order]
                local_alpha = np.full((len(order),), self.config.postprocessing.smoothing_alpha, dtype=float)
                strong = (local_cyber >= self.config.postprocessing.strong_cyber_threshold) | (
                    local_physical >= self.config.postprocessing.strong_physical_threshold
                )
                local_alpha[strong] = self.config.postprocessing.strong_alpha
                local_smoothed = exponential_smooth(local_probs, alpha=local_alpha)
                local_binary, local_states = self.state_machine.run(
                    local_smoothed,
                    cyber_probabilities=local_cyber,
                    physical_probabilities=local_physical,
                    frame_predictions=fused.y_pred_frame[order],
                )
                smoothed[order] = local_smoothed
                stable_binary[order] = local_binary
                for pos, state in zip(order.tolist(), local_states):
                    states[int(pos)] = state
                local_chunks = self.chunker.build_chunks(local_binary, inputs.timestamps[order], local_smoothed)
                for item in local_chunks:
                    mapped = dict(item)
                    mapped["scenario_id"] = str(scenario_id)
                    mapped["start_index"] = int(order[int(item["start_index"])])
                    mapped["end_index"] = int(order[int(item["end_index"])])
                    chunks.append(mapped)
        else:
            alpha = np.full((len(calibrated),), self.config.postprocessing.smoothing_alpha, dtype=float)
            strong = (fused.p_cyber >= self.config.postprocessing.strong_cyber_threshold) | (
                fused.p_physical >= self.config.postprocessing.strong_physical_threshold
            )
            alpha[strong] = self.config.postprocessing.strong_alpha
            smoothed = exponential_smooth(calibrated, alpha=alpha)
            stable_binary, states = self.state_machine.run(
                smoothed,
                cyber_probabilities=fused.p_cyber,
                physical_probabilities=fused.p_physical,
                frame_predictions=fused.y_pred_frame,
            )
            chunks = self.chunker.build_chunks(stable_binary, inputs.timestamps, smoothed)

        evidence = []
        for i in range(len(smoothed)):
            row = dict(fused.evidence[i]) if i < len(fused.evidence) else {}
            row["cyber_backend"] = cyber_score.backend
            row["physical_backend"] = physical_score.backend
            row["state"] = states[i] if i < len(states) else "NORMAL"
            evidence.append(row)
        return DetectionOutput(
            p_abnormal=smoothed,
            p_cyber=fused.p_cyber,
            p_physical=fused.p_physical,
            y_pred_frame=fused.y_pred_frame,
            y_pred_stable=stable_binary,
            evidence=evidence,
            chunks=chunks,
            diagnostics={
                **fused.diagnostics,
                "states": states,
                "cyber_details": cyber_score.details,
                "physical_details": physical_score.details,
            },
        )

    def save(self, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as fh:
            pickle.dump(self, fh)
        self.config.cyber.backend = self.cyber.model.backend
        self.config.physical.backend = self.physical.temporal_model.backend
        self.config.fusion.strategy = str(self.fusion.strategy.__class__.__name__)
        metadata = {
            "config": asdict(self.config),
            "feature_names": self.feature_names,
            "has_preprocessing_state": self.preprocessing_state is not None,
            "runtime_backends": {
                "cyber": self.cyber.model.backend,
                "physical": self.physical.temporal_model.backend,
                "fusion": self.fusion.strategy.__class__.__name__,
            },
        }
        (model_path.with_suffix(".json")).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def load(model_path: Path) -> "HybridEventDetector":
        with model_path.open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, HybridEventDetector):
            raise TypeError(f"Unexpected model at {model_path}: {type(obj)!r}")
        return obj
