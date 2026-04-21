from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.detectors.cyber.event7_mode_model import Event7ModeModel
from src.detectors.cyber.residual_change_detector import ResidualChangeDetector
from src.detectors.domain.models.detection_input import DetectionInput


@dataclass(slots=True)
class Event7DetectorOutput:
    p_event7: np.ndarray
    modes: list[str]
    evidence: list[dict[str, float | str]]


@dataclass(slots=True)
class Event7Detector:
    mode_model: Event7ModeModel = field(default_factory=Event7ModeModel)
    residual: ResidualChangeDetector = field(default_factory=ResidualChangeDetector)
    normal_jump_q: float = 0.02
    normal_outlier_q: float = 0.01
    normal_stuck_q: float = 0.10
    normal_drift_q: float = 0.10

    def fit(self, normal_inputs: DetectionInput) -> None:
        x = np.asarray(normal_inputs.x_windows, dtype=float)
        if x.size == 0:
            self.residual.fit(np.zeros((1,), dtype=float))
            return
        delta = np.abs(np.diff(x, axis=1, prepend=x[:, :1, :]))
        self.residual.fit(delta.mean(axis=(1, 2)))
        self.normal_jump_q = float(np.nanquantile((delta > np.nanquantile(delta, 0.98)).mean(axis=(1, 2)), 0.95))
        self.normal_outlier_q = float(np.nanquantile((np.abs(x) > np.nanquantile(np.abs(x), 0.995)).mean(axis=(1, 2)), 0.95))
        self.normal_stuck_q = float(np.nanquantile((delta < 1e-4).mean(axis=(1, 2)), 0.95))
        self.normal_drift_q = float(np.nanquantile(np.abs(x[:, -1, :] - x[:, 0, :]).mean(axis=1), 0.95))

    def detect(self, inputs: DetectionInput) -> Event7DetectorOutput:
        x = np.asarray(inputs.x_windows, dtype=float)
        if x.size == 0:
            return Event7DetectorOutput(p_event7=np.zeros((0,), dtype=float), modes=[], evidence=[])
        delta = np.abs(np.diff(x, axis=1, prepend=x[:, :1, :]))
        jump_rate = np.clip((delta > np.nanquantile(delta, 0.98)).mean(axis=(1, 2)), 0.0, 1.0)
        outlier_rate = np.clip((np.abs(x) > np.nanquantile(np.abs(x), 0.995)).mean(axis=(1, 2)), 0.0, 1.0)
        drift_strength = np.clip(np.abs(x[:, -1, :] - x[:, 0, :]).mean(axis=1), 0.0, 1.0)
        stuck_rate = np.clip((delta < 1e-4).mean(axis=(1, 2)), 0.0, 1.0)
        replay_similarity = np.zeros((x.shape[0],), dtype=float)
        half = x.shape[1] // 2
        if half >= 2:
            a = x[:, :half, :].reshape(x.shape[0], -1)
            b = x[:, -half:, :].reshape(x.shape[0], -1)
            num = np.sum(a * b, axis=1)
            den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9
            replay_similarity = np.clip((num / den + 1.0) * 0.5, 0.0, 1.0)
        residual_score = self.residual.score(delta.mean(axis=(1, 2)))
        # Event7 selectivity: require persistence beyond normal-quantile baselines.
        jump_excess = np.clip((jump_rate - self.normal_jump_q) / 0.25, 0.0, 1.0)
        outlier_excess = np.clip((outlier_rate - self.normal_outlier_q) / 0.20, 0.0, 1.0)
        stuck_excess = np.clip((stuck_rate - self.normal_stuck_q) / 0.40, 0.0, 1.0)
        drift_excess = np.clip((drift_strength - self.normal_drift_q) / 0.40, 0.0, 1.0)
        persistence = np.clip(0.50 * jump_excess + 0.20 * outlier_excess + 0.20 * stuck_excess + 0.10 * drift_excess, 0.0, 1.0)
        p_event7 = np.clip(
            0.30 * jump_excess
            + 0.25 * outlier_excess
            + 0.20 * stuck_excess
            + 0.15 * residual_score
            + 0.10 * drift_excess,
            0.0,
            1.0,
        )
        p_event7 = np.where(persistence < 0.30, 0.35 * p_event7, p_event7)
        p_event7 = np.where((jump_excess < 0.15) & (outlier_excess < 0.15) & (stuck_excess < 0.15), np.minimum(p_event7, 0.25), p_event7)
        modes: list[str] = []
        evidence: list[dict[str, float | str]] = []
        for i in range(len(p_event7)):
            mode = self.mode_model.classify(
                jump_rate=float(jump_rate[i]),
                drift_strength=float(drift_strength[i]),
                stuck_rate=float(stuck_rate[i]),
                replay_similarity=float(replay_similarity[i]),
            )
            modes.append(mode)
            evidence.append(
                {
                    "jump_rate": float(jump_rate[i]),
                    "outlier_rate": float(outlier_rate[i]),
                    "drift_strength": float(drift_strength[i]),
                    "stuck_rate": float(stuck_rate[i]),
                    "replay_similarity": float(replay_similarity[i]),
                    "residual_change_score": float(residual_score[i]),
                    "jump_excess": float(jump_excess[i]),
                    "outlier_excess": float(outlier_excess[i]),
                    "stuck_excess": float(stuck_excess[i]),
                    "drift_excess": float(drift_excess[i]),
                    "persistence_score": float(persistence[i]),
                    "mode": mode,
                }
            )
        return Event7DetectorOutput(p_event7=p_event7, modes=modes, evidence=evidence)
