from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.detectors.domain.models.detection_output import DetectionOutput


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    yt = y_true.astype(int)
    yp = y_pred.astype(int)
    return {
        "tp": int(((yp == 1) & (yt == 1)).sum()),
        "tn": int(((yp == 0) & (yt == 0)).sum()),
        "fp": int(((yp == 1) & (yt == 0)).sum()),
        "fn": int(((yp == 0) & (yt == 1)).sum()),
    }


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    yt = y_true.astype(int)
    if len(np.unique(yt)) < 2:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score), dtype=float)
    pos = yt == 1
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    rank_sum = float(ranks[pos].sum())
    return float((rank_sum - n_pos * (n_pos - 1.0) / 2.0) / max(n_pos * n_neg, 1.0))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    yt = y_true.astype(int)
    if len(np.unique(yt)) < 2:
        return None
    order = np.argsort(-y_score)
    y = yt[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(float((yt == 1).sum()), 1.0)
    return float(np.trapezoid(precision, recall))


def duration_minutes(timestamps: np.ndarray, scenario_ids: np.ndarray | None = None) -> float:
    if len(timestamps) == 0:
        return 0.0
    if scenario_ids is None or len(scenario_ids) != len(timestamps):
        span = float(np.max(timestamps) - np.min(timestamps)) if len(timestamps) > 1 else 0.0
        return max(safe_div(span, 60.0), safe_div(len(timestamps), 60.0))
    total = 0.0
    frame = pd.DataFrame({"timestamp": timestamps, "scenario_id": scenario_ids})
    for _, group in frame.groupby("scenario_id"):
        ts = group["timestamp"].to_numpy(dtype=float)
        if len(ts) <= 1:
            total += safe_div(len(ts), 60.0)
        else:
            span = float(np.max(ts) - np.min(ts))
            total += max(safe_div(span, 60.0), safe_div(len(ts), 60.0))
    return total


def detection_delay_seconds(y_true: np.ndarray, y_pred: np.ndarray, timestamps: np.ndarray) -> float | None:
    true_idx = np.where(y_true == 1)[0]
    if len(true_idx) == 0:
        return None
    first_true = true_idx[0]
    pred_idx = np.where((y_pred == 1) & (np.arange(len(y_pred)) >= first_true))[0]
    if len(pred_idx) == 0:
        return None
    t0 = float(timestamps[first_true])
    t1 = float(timestamps[pred_idx[0]])
    return max(0.0, t1 - t0)


def metric_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    timestamps: np.ndarray,
    scenario_ids: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    cm = confusion_counts(y_true, y_pred)
    precision = safe_div(cm["tp"], cm["tp"] + cm["fp"])
    recall = safe_div(cm["tp"], cm["tp"] + cm["fn"])
    tnr = safe_div(cm["tn"], cm["tn"] + cm["fp"])
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    fp_per_min = safe_div(cm["fp"], duration_minutes(timestamps, scenario_ids=scenario_ids))
    return {
        "f1_abnormal": f1,
        "precision_abnormal": precision,
        "recall_abnormal": recall,
        "balanced_accuracy": 0.5 * (recall + tnr),
        "false_positives_per_minute": fp_per_min,
        "detection_delay_s": detection_delay_seconds(y_true, y_pred, timestamps),
        "roc_auc": roc_auc(y_true, y_score),
        "pr_auc": pr_auc(y_true, y_score),
        "confusion": cm,
    }


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray, timestamps: np.ndarray, scenario_ids: np.ndarray | None) -> tuple[dict, pd.DataFrame]:
    rows = []
    best: dict | None = None
    best_obj = -1e9
    for thr in np.linspace(0.3, 0.8, 11):
        pred = (y_score >= thr).astype(int)
        m = metric_bundle(y_true, pred, y_score, timestamps, scenario_ids=scenario_ids)
        obj = float(m["f1_abnormal"]) + 0.2 * float(m["recall_abnormal"]) - 0.05 * float(m["false_positives_per_minute"])
        row = {"threshold": float(thr), **m, "objective": obj}
        rows.append(row)
        if obj > best_obj:
            best_obj = obj
            best = row
    frame = pd.DataFrame(rows).sort_values(["objective", "f1_abnormal"], ascending=False).reset_index(drop=True)
    if best is None:
        best = {"threshold": 0.5, "objective": 0.0}
    return best, frame


def build_frame_output(
    output: DetectionOutput,
    y_true: np.ndarray,
    timestamps: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true": y_true.astype(int),
            "p_abnormal": output.p_abnormal,
            "p_cyber": output.p_cyber,
            "p_physical": output.p_physical,
            "y_pred_frame": output.y_pred_frame,
            "y_pred_stable": output.y_pred_stable,
            "is_abnormal": output.is_abnormal,
            "confidence": output.confidence,
        }
    )
    if not metadata.empty:
        frame = pd.concat([frame, metadata.reset_index(drop=True)], axis=1)
    return frame


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

