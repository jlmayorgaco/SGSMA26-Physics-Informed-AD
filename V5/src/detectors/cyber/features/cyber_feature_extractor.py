from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.detectors.cyber.heuristics.missing_data_rules import missing_data_evidence
from src.detectors.cyber.heuristics.spike_rules import spike_evidence
from src.detectors.cyber.heuristics.stuck_value_rules import stuck_value_evidence
from src.detectors.cyber.heuristics.timestamp_rules import timestamp_evidence
from src.detectors.domain.models.detection_input import DetectionInput


@dataclass(slots=True)
class CyberFeatureBatch:
    x: np.ndarray
    feature_names: list[str]
    rule_evidence: dict[str, np.ndarray]
    rule_score: np.ndarray


def _summary_features(windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    mean_abs_delta = np.abs(np.diff(windows, axis=1, prepend=windows[:, :1, :])).mean(axis=1)
    matrix = np.concatenate([mean, std, mean_abs_delta], axis=1)
    n_feat = windows.shape[2]
    names = (
        [f"mean_{i}" for i in range(n_feat)]
        + [f"std_{i}" for i in range(n_feat)]
        + [f"madelta_{i}" for i in range(n_feat)]
    )
    return matrix, names


def _stuck_after_missing(windows: np.ndarray, feature_names: list[str]) -> np.ndarray:
    nan_cols = [i for i, name in enumerate(feature_names) if name.endswith("__is_nan")]
    if not nan_cols:
        return np.zeros((windows.shape[0],), dtype=float)
    base_cols = [i for i, name in enumerate(feature_names) if not name.endswith("__is_nan")]
    if not base_cols:
        return np.zeros((windows.shape[0],), dtype=float)
    x = windows[:, :, base_cols]
    nan_avg = np.clip(windows[:, :, nan_cols].mean(axis=2), 0.0, 1.0)
    # Compare movement before/after high missingness frames.
    delta = np.abs(np.diff(x, axis=1, prepend=x[:, :1, :])).mean(axis=2)
    score = np.zeros((windows.shape[0],), dtype=float)
    for i in range(windows.shape[0]):
        missing_frames = nan_avg[i] > 0.5
        if not np.any(missing_frames):
            continue
        after_mask = np.roll(missing_frames, 1)
        after_mask[0] = False
        if not np.any(after_mask):
            continue
        post_delta = float(delta[i][after_mask].mean())
        overall_delta = float(delta[i].mean())
        if overall_delta <= 1e-9:
            score[i] = 0.0
        else:
            score[i] = float(np.clip(1.0 - post_delta / overall_delta, 0.0, 1.0))
    return score


class CyberFeatureExtractor:
    def extract(self, inputs: DetectionInput) -> CyberFeatureBatch:
        windows = inputs.x_windows
        data_present_index = inputs.feature_names.index("DATA_PRESENT") if "DATA_PRESENT" in inputs.feature_names else None
        missing = missing_data_evidence(windows, feature_names=inputs.feature_names, data_present_index=data_present_index)
        stuck = stuck_value_evidence(windows)
        spike = spike_evidence(windows)
        ts = timestamp_evidence(inputs.timestamps)
        summary_matrix, summary_names = _summary_features(windows)
        stuck_after_missing = _stuck_after_missing(windows, inputs.feature_names)
        rule_matrix = np.column_stack(
            [
                missing["missing_ratio"],
                missing["all_missing_flag"],
                missing["nan_ratio"],
                missing["data_present_transition_rate"],
                missing["nan_burst_count"],
                missing["nan_burst_max_len"],
                missing["missing_asymmetry"],
                missing["partial_dropout_ratio"],
                stuck["stuck_feature_fraction"],
                stuck["low_delta_fraction"],
                stuck_after_missing,
                spike["spike_ratio"],
                spike["max_abs_z"],
                ts["score"],
                ts["irregular_flag"],
            ]
        )
        rule_names = [
            "missing_ratio",
            "all_missing_flag",
            "nan_ratio",
            "data_present_transition_rate",
            "nan_burst_count",
            "nan_burst_max_len",
            "missing_asymmetry",
            "partial_dropout_ratio",
            "stuck_feature_fraction",
            "low_delta_fraction",
            "stuck_after_missing",
            "spike_ratio",
            "max_abs_z",
            "timestamp_score",
            "timestamp_irregular_flag",
        ]
        full_x = np.concatenate([rule_matrix, summary_matrix], axis=1)
        rule_score = np.clip(
            0.45 * missing["score"]
            + 0.15 * stuck["score"]
            + 0.15 * spike["score"]
            + 0.10 * ts["score"]
            + 0.15 * stuck_after_missing,
            0.0,
            1.0,
        )
        return CyberFeatureBatch(
            x=full_x.astype(float),
            feature_names=rule_names + summary_names,
            rule_evidence={
                **missing,
                **stuck,
                **spike,
                **ts,
            },
            rule_score=rule_score,
        )
