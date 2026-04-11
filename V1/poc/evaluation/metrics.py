"""Metrics for the POC ablation table."""

from __future__ import annotations

import math

import numpy as np


def classification_report_numbers(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return macro/weighted/per-class F1 and confusion matrix."""

    from sklearn.metrics import confusion_matrix, f1_score

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = [label for label in range(9) if label in present]
    if not labels:
        labels = list(range(9))
    per_class = {
        int(label): float(f1_score(y_true, y_pred, labels=[label], average="macro", zero_division=0))
        for label in range(9)
    }
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class_f1": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(9))),
    }


def detection_metrics(pred_frames: np.ndarray, true_frames: np.ndarray, fps: float = 30.0, tolerance_s: float = 5.0) -> dict:
    """Frame-level event-onset matching for precision/recall/F1."""

    pred = sorted(int(x) for x in np.asarray(pred_frames, dtype=int).tolist())
    true = sorted(int(x) for x in np.asarray(true_frames, dtype=int).tolist())
    tol = int(round(tolerance_s * fps))
    matched: set[int] = set()
    tp = fp = 0
    delays: list[float] = []
    for p in pred:
        candidates = [(abs(p - t), j, t) for j, t in enumerate(true) if abs(p - t) <= tol and j not in matched]
        if candidates:
            _, j, t = min(candidates)
            matched.add(j)
            tp += 1
            delays.append(max(0, p - t) / fps)
        else:
            fp += 1
    fn = len(true) - len(matched)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_delay_sec": float(np.mean(delays)) if delays else float("nan"),
    }


def localization_metrics(top3: list[list[int]], true_buses: list[int]) -> dict:
    """Top-1 and Top-3 localization accuracy."""

    if not true_buses:
        return {"top1": 0.0, "top3": 0.0}
    top1_hit = 0
    top3_hit = 0
    for pred, truth in zip(top3, true_buses):
        if pred and pred[0] == truth:
            top1_hit += 1
        if truth in pred[:3]:
            top3_hit += 1
    n = len(true_buses)
    return {"top1": top1_hit / n, "top3": top3_hit / n}


def competition_score(macro_f1_real: float, params: int, lam: float = 0.03) -> float:
    """SGSMA-style compactness-adjusted score."""

    return float(macro_f1_real - lam * math.log10(max(int(params), 1)))

