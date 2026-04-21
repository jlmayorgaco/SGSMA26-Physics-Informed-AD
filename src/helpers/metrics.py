from __future__ import annotations

import numpy as np

from src.classes.report_models import BinaryMetrics


def compute_binary_metrics(true_values: np.ndarray, pred_values: np.ndarray) -> BinaryMetrics:
    true_int = np.asarray(true_values, dtype=int)
    pred_int = np.asarray(pred_values, dtype=int)
    tp = int(np.sum((true_int == 1) & (pred_int == 1)))
    tn = int(np.sum((true_int == 0) & (pred_int == 0)))
    fp = int(np.sum((true_int == 0) & (pred_int == 1)))
    fn = int(np.sum((true_int == 1) & (pred_int == 0)))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)

    return BinaryMetrics(
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        accuracy=float(accuracy),
    )

