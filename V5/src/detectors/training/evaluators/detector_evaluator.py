from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.detectors.training.evaluators.calibration_evaluator import CalibrationEvaluator


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


@dataclass(slots=True)
class BinaryDetectorEvaluator:
    calibration_bins: int = 10

    def confusion_counts(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_pred, dtype=int)
        return {
            "tp": int(((yp == 1) & (yt == 1)).sum()),
            "tn": int(((yp == 0) & (yt == 0)).sum()),
            "fp": int(((yp == 1) & (yt == 0)).sum()),
            "fn": int(((yp == 0) & (yt == 1)).sum()),
        }

    def roc_auc(self, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
        yt = np.asarray(y_true, dtype=int)
        ys = np.asarray(y_score, dtype=float)
        if len(np.unique(yt)) < 2:
            return None
        order = np.argsort(ys)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(ys), dtype=float)
        pos = yt == 1
        n_pos = float(pos.sum())
        n_neg = float((~pos).sum())
        rank_sum = float(ranks[pos].sum())
        return float((rank_sum - n_pos * (n_pos - 1.0) / 2.0) / max(n_pos * n_neg, 1.0))

    def pr_auc(self, y_true: np.ndarray, y_score: np.ndarray) -> float | None:
        yt = np.asarray(y_true, dtype=int)
        ys = np.asarray(y_score, dtype=float)
        if len(np.unique(yt)) < 2:
            return None
        order = np.argsort(-ys)
        y = yt[order]
        tp = np.cumsum(y == 1)
        fp = np.cumsum(y == 0)
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / max(float((yt == 1).sum()), 1.0)
        return float(np.trapezoid(precision, recall))

    def duration_minutes(self, timestamps: np.ndarray, scenario_ids: np.ndarray | None = None) -> float:
        ts = np.asarray(timestamps, dtype=float)
        if len(ts) == 0:
            return 0.0
        if scenario_ids is None or len(scenario_ids) != len(ts):
            span = float(ts.max() - ts.min()) if len(ts) > 1 else 0.0
            return max(safe_div(span, 60.0), safe_div(len(ts), 60.0))
        total = 0.0
        frame = pd.DataFrame({"timestamp": ts, "scenario_id": scenario_ids})
        for _, group in frame.groupby("scenario_id"):
            gts = group["timestamp"].to_numpy(dtype=float)
            if len(gts) <= 1:
                total += safe_div(len(gts), 60.0)
            else:
                span = float(gts.max() - gts.min())
                total += max(safe_div(span, 60.0), safe_div(len(gts), 60.0))
        return total

    def detection_delay_seconds(self, y_true: np.ndarray, y_pred: np.ndarray, timestamps: np.ndarray) -> float | None:
        yt = np.asarray(y_true, dtype=int)
        yp = np.asarray(y_pred, dtype=int)
        ts = np.asarray(timestamps, dtype=float)
        true_idx = np.where(yt == 1)[0]
        if len(true_idx) == 0:
            return None
        first_true = int(true_idx[0])
        pred_idx = np.where((yp == 1) & (np.arange(len(yp)) >= first_true))[0]
        if len(pred_idx) == 0:
            return None
        return max(0.0, float(ts[int(pred_idx[0])] - ts[first_true]))

    def metric_bundle(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray,
        timestamps: np.ndarray,
        scenario_ids: np.ndarray | None,
    ) -> dict[str, float | int | None | dict[str, int]]:
        cm = self.confusion_counts(y_true, y_pred)
        precision = safe_div(cm["tp"], cm["tp"] + cm["fp"])
        recall = safe_div(cm["tp"], cm["tp"] + cm["fn"])
        tnr = safe_div(cm["tn"], cm["tn"] + cm["fp"])
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        fp_per_min = safe_div(cm["fp"], self.duration_minutes(timestamps, scenario_ids=scenario_ids))
        calibration = CalibrationEvaluator(n_bins=self.calibration_bins).evaluate(y_true=y_true, y_prob=y_score)
        return {
            "f1_abnormal": float(f1),
            "precision_abnormal": float(precision),
            "recall_abnormal": float(recall),
            "balanced_accuracy": float(0.5 * (recall + tnr)),
            "false_positives": int(cm["fp"]),
            "false_positives_per_minute": float(fp_per_min),
            "detection_delay_s": self.detection_delay_seconds(y_true, y_pred, timestamps),
            "roc_auc": self.roc_auc(y_true, y_score),
            "pr_auc": self.pr_auc(y_true, y_score),
            "brier_score": float(calibration.brier),
            "support_abnormal": int((np.asarray(y_true, dtype=int) == 1).sum()),
            "support_normal": int((np.asarray(y_true, dtype=int) == 0).sum()),
            "confusion": cm,
        }

    def curve_points(self, y_true: np.ndarray, y_score: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
        thresholds = np.unique(np.round(np.asarray(y_score, dtype=float), 6))
        if len(thresholds) == 0:
            thresholds = np.array([0.5])
        roc_rows: list[dict[str, float]] = []
        pr_rows: list[dict[str, float]] = []
        yt = np.asarray(y_true, dtype=int)
        ys = np.asarray(y_score, dtype=float)
        for threshold in sorted(thresholds, reverse=True):
            pred = (ys >= threshold).astype(int)
            cm = self.confusion_counts(yt, pred)
            tpr = safe_div(cm["tp"], cm["tp"] + cm["fn"])
            fpr = safe_div(cm["fp"], cm["fp"] + cm["tn"])
            precision = safe_div(cm["tp"], cm["tp"] + cm["fp"])
            roc_rows.append({"threshold": float(threshold), "fpr": float(fpr), "tpr": float(tpr)})
            pr_rows.append({"threshold": float(threshold), "precision": float(precision), "recall": float(tpr)})
        return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)

    def familywise_metrics(self, frame: pd.DataFrame) -> dict[str, dict[str, float | int | None | dict[str, int]]]:
        event_coarse = pd.to_numeric(frame.get("event_coarse", pd.Series(np.nan, index=frame.index)), errors="coerce")
        groups = {
            "normal": event_coarse.eq(0).to_numpy(),
            "physical_heavy": event_coarse.isin([1, 2, 3, 4]).to_numpy(),
            "cyber_heavy": event_coarse.isin([5, 7]).to_numpy(),
            "concurrent_heavy": event_coarse.isin([6, 8]).to_numpy(),
        }
        output: dict[str, dict[str, float | int | None | dict[str, int]]] = {}
        for name, mask in groups.items():
            if int(mask.sum()) == 0:
                output[name] = {"windows": 0}
                continue
            group = frame.loc[mask]
            output[name] = {
                "windows": int(mask.sum()),
                **self.metric_bundle(
                    y_true=group["y_true"].to_numpy(dtype=int),
                    y_pred=group["y_pred_stable"].to_numpy(dtype=int),
                    y_score=group["p_abnormal"].to_numpy(dtype=float),
                    timestamps=group["timestamp"].to_numpy(dtype=float),
                    scenario_ids=group["scenario_id"].to_numpy() if "scenario_id" in group.columns else None,
                ),
            }

        if "scenario_family" in frame.columns:
            scenario_rows: dict[str, dict[str, float | int | None | dict[str, int]]] = {}
            for family_name, group in frame.groupby("scenario_family"):
                scenario_rows[str(family_name)] = {
                    "windows": int(len(group)),
                    **self.metric_bundle(
                        y_true=group["y_true"].to_numpy(dtype=int),
                        y_pred=group["y_pred_stable"].to_numpy(dtype=int),
                        y_score=group["p_abnormal"].to_numpy(dtype=float),
                        timestamps=group["timestamp"].to_numpy(dtype=float),
                        scenario_ids=group["scenario_id"].to_numpy() if "scenario_id" in group.columns else None,
                    ),
                }
            output["scenario_family"] = scenario_rows
        return output

    def per_scenario_metrics(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, float | int | str | None]] = []
        if "scenario_id" not in frame.columns:
            return pd.DataFrame(rows)
        for scenario_id, group in frame.groupby("scenario_id"):
            metrics = self.metric_bundle(
                y_true=group["y_true"].to_numpy(dtype=int),
                y_pred=group["y_pred_stable"].to_numpy(dtype=int),
                y_score=group["p_abnormal"].to_numpy(dtype=float),
                timestamps=group["timestamp"].to_numpy(dtype=float),
                scenario_ids=group["scenario_id"].to_numpy(),
            )
            rows.append(
                {
                    "scenario_id": str(scenario_id),
                    "split": str(group["split"].iloc[0]) if "split" in group.columns else "",
                    "template_name": str(group["template_name"].iloc[0]) if "template_name" in group.columns else "",
                    "difficulty": str(group["difficulty_level"].iloc[0]) if "difficulty_level" in group.columns else "",
                    "support_abnormal": int((group["y_true"].to_numpy(dtype=int) == 1).sum()),
                    "support_normal": int((group["y_true"].to_numpy(dtype=int) == 0).sum()),
                    "precision_abnormal": float(metrics["precision_abnormal"]),
                    "recall_abnormal": float(metrics["recall_abnormal"]),
                    "f1_abnormal": float(metrics["f1_abnormal"]),
                    "balanced_accuracy": float(metrics["balanced_accuracy"]),
                    "false_positives": int(metrics["false_positives"]),
                    "false_positives_per_minute": float(metrics["false_positives_per_minute"]),
                    "detection_delay_s": metrics["detection_delay_s"],
                    "average_probability": float(group["p_abnormal"].mean()),
                    "max_probability": float(group["p_abnormal"].max()),
                }
            )
        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(
                columns=[
                    "scenario_id",
                    "split",
                    "template_name",
                    "difficulty",
                    "support_abnormal",
                    "support_normal",
                    "precision_abnormal",
                    "recall_abnormal",
                    "f1_abnormal",
                    "balanced_accuracy",
                    "false_positives",
                    "false_positives_per_minute",
                    "detection_delay_s",
                    "average_probability",
                    "max_probability",
                ]
            )
        sort_columns = [col for col in ["split", "scenario_id"] if col in out.columns]
        if sort_columns:
            out = out.sort_values(sort_columns, ascending=[True] * len(sort_columns))
        return out.reset_index(drop=True)
