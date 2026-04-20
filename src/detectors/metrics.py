from __future__ import annotations

import numpy as np
import pandas as pd

from src.detectors.models import DetectorMetrics


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score), dtype=float)
    pos = y_true == 1
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    rank_sum = float(ranks[pos].sum())
    return (rank_sum - n_pos * (n_pos - 1.0) / 2.0) / max(n_pos * n_neg, 1.0)


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(float((y_true == 1).sum()), 1.0)
    return float(np.trapezoid(precision, recall))


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> DetectorMetrics:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    balanced = 0.5 * (recall + specificity)
    return DetectorMetrics(
        accuracy=_safe_div(tp + tn, len(y_true)),
        precision=precision,
        recall=recall,
        f1=f1,
        balanced_accuracy=balanced,
        roc_auc=_roc_auc(y_true, y_prob),
        pr_auc=_pr_auc(y_true, y_prob),
        support_positive=int((y_true == 1).sum()),
        support_negative=int((y_true == 0).sum()),
    )


def metrics_to_frame(metrics: DetectorMetrics) -> pd.DataFrame:
    return pd.DataFrame([metrics.__dict__])
