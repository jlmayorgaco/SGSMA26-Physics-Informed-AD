"""Evaluation metrics for the SGSMA 2026 competition (spec §10).

Three metric blocks
-------------------
1. **Detection**     — precision, recall, F1, FP/min, detection delay (seconds)
2. **Classification** — macro-F1, weighted-F1, per-class P/R/F1, 9×9 confusion matrix
3. **Localization**  — Top-1 accuracy, Top-3 accuracy, mean electrical-distance error

All functions accept ground-truth and predicted arrays aligned at the event level
(one row per detected alarm onset).  The caller is responsible for aligning
predicted outputs to ground-truth events via timestamp matching.

Detection alignment
-------------------
A prediction fires a True Positive if there exists a ground-truth event window
within `tol_sec` seconds.  Multiple predictions inside the same event window count
as one TP plus (n-1) FPs.  Missed ground-truth events count as FNs.
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# All event labels (0 = normal, used for confusion matrix rows/cols)
ALL_LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8]


# ── Detection metrics ─────────────────────────────────────────────────────────

def detection_metrics(
    alarm_times: np.ndarray,
    gt_event_windows: list[tuple[float, float]],
    total_duration_sec: float,
    tol_sec: float = 5.0,
) -> dict:
    """Compute detection precision, recall, F1, FP/min, and mean detection delay.

    Args:
        alarm_times:       (K,) array of predicted alarm onset times in seconds.
        gt_event_windows:  List of (start_sec, end_sec) ground-truth event windows.
                           A prediction is a TP if it falls within one of these windows
                           (or within tol_sec before start_sec for early detection).
        total_duration_sec: Duration of the evaluated segment (for FP/min).
        tol_sec:           Tolerance window around event start (seconds).

    Returns:
        dict with keys: tp, fp, fn, precision, recall, f1, fp_per_min,
                        mean_delay_sec, delays_sec.
    """
    alarm_times = np.asarray(alarm_times, dtype=float)
    tp = 0
    fp = 0
    delays: list[float] = []
    matched_gt: set[int] = set()

    for t_alarm in sorted(alarm_times):
        matched = False
        for gi, (t_start, t_end) in enumerate(gt_event_windows):
            # Allow prediction up to tol_sec before the event start
            if (t_start - tol_sec) <= t_alarm <= (t_end + tol_sec):
                if gi not in matched_gt:
                    tp += 1
                    matched_gt.add(gi)
                    delays.append(max(0.0, t_alarm - t_start))
                    matched = True
                    break
                else:
                    # Already matched — second alarm in same window → FP
                    fp += 1
                    matched = True
                    break
        if not matched:
            fp += 1

    fn = len(gt_event_windows) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    fp_per_min = fp / (total_duration_sec / 60.0) if total_duration_sec > 0 else float("inf")
    mean_delay = float(np.mean(delays)) if delays else float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_per_min": fp_per_min,
        "mean_delay_sec": mean_delay,
        "delays_sec": delays,
    }


# ── Classification metrics ────────────────────────────────────────────────────

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[int] | None = None,
) -> dict:
    """Compute macro-F1, weighted-F1, per-class P/R/F1, and 9×9 confusion matrix.

    Args:
        y_true:  (N,) ground-truth event labels.
        y_pred:  (N,) predicted event labels.
        labels:  Label set for the confusion matrix.  Defaults to ALL_LABELS (0-8).

    Returns:
        dict with keys: macro_f1, weighted_f1, per_class (dict), confusion_matrix.
    """
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, confusion_matrix,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if labels is None:
        present = sorted(set(y_true) | set(y_pred))
        labels  = [l for l in ALL_LABELS if l in present]

    macro_f1    = float(f1_score(y_true, y_pred, labels=labels, average="macro",   zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

    per_class: dict[int, dict[str, float]] = {}
    prec_all = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec_all  = recall_score   (y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_all   = f1_score       (y_true, y_pred, labels=labels, average=None, zero_division=0)
    for i, lbl in enumerate(labels):
        per_class[lbl] = {
            "precision": float(prec_all[i]),
            "recall":    float(rec_all[i]),
            "f1":        float(f1_all[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=list(ALL_LABELS))

    return {
        "macro_f1":        macro_f1,
        "weighted_f1":     weighted_f1,
        "per_class":       per_class,
        "confusion_matrix": cm,
        "labels":          labels,
    }


# ── Localization metrics ──────────────────────────────────────────────────────

def localization_metrics(
    top3_buses: list[list[int]],
    true_buses: list[int],
    zbus: np.ndarray | None = None,
    ext_bus_order: list[int] | None = None,
) -> dict:
    """Compute Top-1, Top-3 accuracy and mean electrical-distance error.

    Args:
        top3_buses:    List of up to 3 predicted bus numbers per event (Top-3 list).
                       Each inner list should be in descending confidence order.
        true_buses:    Ground-truth origin bus per event (competition bus number).
        zbus:          (N×N) bus impedance matrix for electrical-distance error.
                       Optional — if None, only Top-1/Top-3 accuracy is computed.
        ext_bus_order: Competition bus number for each Zbus row/column index.
                       Required when zbus is provided.

    Returns:
        dict with keys: top1_acc, top3_acc, n_events, mean_elec_dist_error
                        (latter only if zbus provided).
    """
    n = len(true_buses)
    if n == 0:
        return {"top1_acc": 0.0, "top3_acc": 0.0, "n_events": 0}

    top1_hits = 0
    top3_hits = 0
    elec_dists: list[float] = []

    for top3, true_bus in zip(top3_buses, true_buses):
        if not top3:
            continue
        top1 = top3[0]
        if top1 == true_bus:
            top1_hits += 1
        if true_bus in top3:
            top3_hits += 1

        # Electrical distance from Top-1 prediction to true bus
        if zbus is not None and ext_bus_order is not None:
            try:
                i = ext_bus_order.index(top1)
                j = ext_bus_order.index(true_bus)
                # d_ij = |Z_ii + Z_jj - 2 Z_ij|
                d = abs(zbus[i, i] + zbus[j, j] - 2 * zbus[i, j])
                elec_dists.append(float(abs(d)))
            except (ValueError, IndexError):
                pass

    result: dict = {
        "top1_acc": top1_hits / n,
        "top3_acc": top3_hits / n,
        "n_events": n,
    }
    if elec_dists:
        result["mean_elec_dist_error"] = float(np.mean(elec_dists))
    return result


# ── Aggregate report ──────────────────────────────────────────────────────────

def compute_all_metrics(
    df: pd.DataFrame,
    alarm_times: np.ndarray,
    predicted_labels: np.ndarray,
    top3_buses: list[list[int]],
    gt_event_windows: list[tuple[float, float]],
    true_labels_at_alarms: np.ndarray,
    true_buses_at_alarms: list[int],
    zbus: np.ndarray | None = None,
    ext_bus_order: list[int] | None = None,
    tol_sec: float = 5.0,
) -> dict:
    """Convenience wrapper computing all three metric blocks at once.

    Args:
        df:                    Full merged DataFrame (used for total_duration_sec).
        alarm_times:           Predicted alarm onset times (seconds).
        predicted_labels:      Classifier output at each alarm.
        top3_buses:            Localizer Top-3 per alarm.
        gt_event_windows:      Ground-truth (start_sec, end_sec) event intervals.
        true_labels_at_alarms: Ground-truth label at each predicted alarm.
        true_buses_at_alarms:  Ground-truth origin bus at each predicted alarm.
        zbus, ext_bus_order:   For electrical-distance computation.
        tol_sec:               Detection tolerance window.

    Returns:
        dict with "detection", "classification", "localization" sub-dicts.
    """
    ts = df["TIMESTAMP"].to_numpy(float)
    total_sec = float(ts[-1] - ts[0]) if len(ts) > 1 else 1.0

    det = detection_metrics(alarm_times, gt_event_windows, total_sec, tol_sec)
    clf = classification_metrics(true_labels_at_alarms, predicted_labels)
    loc = localization_metrics(top3_buses, true_buses_at_alarms, zbus, ext_bus_order)

    return {"detection": det, "classification": clf, "localization": loc}


def print_metrics(metrics: dict) -> None:
    """Pretty-print the metric report to stdout."""
    det = metrics.get("detection", {})
    clf = metrics.get("classification", {})
    loc = metrics.get("localization", {})

    print("\n=== Detection ===")
    print(f"  Precision:      {det.get('precision', 0):.3f}")
    print(f"  Recall:         {det.get('recall', 0):.3f}")
    print(f"  F1:             {det.get('f1', 0):.3f}")
    print(f"  FP/min:         {det.get('fp_per_min', float('nan')):.3f}")
    print(f"  Mean delay:     {det.get('mean_delay_sec', float('nan')):.2f} s")
    print(f"  TP/FP/FN:       {det.get('tp',0)}/{det.get('fp',0)}/{det.get('fn',0)}")

    print("\n=== Classification ===")
    print(f"  Macro-F1:       {clf.get('macro_f1', 0):.3f}")
    print(f"  Weighted-F1:    {clf.get('weighted_f1', 0):.3f}")
    pc = clf.get("per_class", {})
    if pc:
        print("  Per-class:")
        for lbl in sorted(pc):
            d = pc[lbl]
            print(f"    Label {lbl}: P={d['precision']:.2f}  R={d['recall']:.2f}  F1={d['f1']:.2f}")
    cm = clf.get("confusion_matrix")
    if cm is not None:
        print("  Confusion matrix (rows=true, cols=pred, labels 0-8):")
        header = "       " + "  ".join(f"{l:2d}" for l in ALL_LABELS)
        print(f"  {header}")
        for i, row in enumerate(cm):
            cells = "  ".join(f"{v:2d}" for v in row)
            print(f"  {ALL_LABELS[i]:4d}:  {cells}")

    print("\n=== Localization ===")
    print(f"  Top-1 accuracy: {loc.get('top1_acc', 0):.3f}  ({int(loc.get('top1_acc',0)*loc.get('n_events',0))}/{loc.get('n_events',0)})")
    print(f"  Top-3 accuracy: {loc.get('top3_acc', 0):.3f}  ({int(loc.get('top3_acc',0)*loc.get('n_events',0))}/{loc.get('n_events',0)})")
    if "mean_elec_dist_error" in loc:
        print(f"  Mean elec dist: {loc['mean_elec_dist_error']:.4f} p.u.")
    print()
