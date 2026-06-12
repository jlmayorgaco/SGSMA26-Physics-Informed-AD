from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from pipelines import _bootstrap  # type: ignore  # noqa: F401

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

from src.classes import PipelineResult
from src.models.localizer import electrical_distance
from src.utils.io import json_safe, write_json


DEFAULT_DATASET_DIR = Path("workbench") / "agnostic_training_set"
DEFAULT_OUT = Path("workbench") / "agnostic_models"
WINDOW_LABEL_COLUMNS = {
    "sample_id",
    "sim_id",
    "placement_id",
    "observed_pmus",
    "event_label",
    "abnormal_label",
    "location_label",
    "location_type",
}
CANDIDATE_LABEL_COLUMNS = {
    "sample_id",
    "sim_id",
    "event_label",
    "location_label",
    "candidate_label",
    "target",
    "observed_pmus",
}


def _model(n_estimators: int, random_state: int, class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=int(n_estimators),
                    random_state=int(random_state),
                    max_features="sqrt",
                    class_weight=class_weight,
                    n_jobs=1,
                ),
            ),
        ]
    )


def _feature_columns(frame: pd.DataFrame, label_columns: set[str]) -> list[str]:
    return [col for col in frame.columns if col not in label_columns and frame[col].dtype.kind in "bifc"]


def _split_by_sim(frame: pd.DataFrame, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    idx = np.arange(len(frame))
    train, test = next(splitter.split(idx, groups=frame["sim_id"].astype(str)))
    return train, test


def _detection_metrics(y_event: np.ndarray, pred_abnormal: np.ndarray, window_seconds: float = 30.0) -> dict[str, Any]:
    y = (np.asarray(y_event, dtype=int) != 0).astype(int)
    p = np.asarray(pred_abnormal, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, p, labels=[1], zero_division=0)
    normal = y == 0
    false_alarms = int(np.sum((p == 1) & normal))
    normal_minutes = max(float(np.sum(normal)) * float(window_seconds) / 60.0, 1e-9)
    return {
        "abnormal_precision": float(precision[0]),
        "abnormal_recall": float(recall[0]),
        "abnormal_f1": float(f1[0]),
        "false_alarm_rate_per_min": float(false_alarms / normal_minutes),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = list(range(9))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _rank_candidates(candidate_frame: pd.DataFrame, scores: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    scored = candidate_frame[["sample_id", "location_label", "candidate_label", "target", "event_label"]].copy()
    scored["score"] = np.asarray(scores, dtype=float)
    rows = []
    for sample_id, group in scored.groupby("sample_id", sort=False):
        ordered = group.sort_values("score", ascending=False).reset_index(drop=True)
        true_location = str(ordered["location_label"].iloc[0])
        pred = str(ordered["candidate_label"].iloc[0])
        top3 = ordered["candidate_label"].astype(str).head(3).tolist()
        rows.append(
            {
                "sample_id": sample_id,
                "event_label": int(ordered["event_label"].iloc[0]),
                "true_location": true_location,
                "pred_location": pred,
                "top3_hit": int(true_location in top3),
                "exact": int(true_location == pred),
                "electrical_distance": electrical_distance(true_location, pred) if true_location != "none" else 0.0,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result, {"evaluated": 0, "exact_accuracy": 0.0, "top3_accuracy": 0.0, "mean_electrical_distance": 0.0}
    metrics = {
        "evaluated": int(len(result)),
        "exact_accuracy": float(result["exact"].mean()),
        "top3_accuracy": float(result["top3_hit"].mean()),
        "mean_electrical_distance": float(result["electrical_distance"].mean()),
        "by_event": {
            str(event): {
                "support": int(len(group)),
                "exact_accuracy": float(group["exact"].mean()),
                "top3_accuracy": float(group["top3_hit"].mean()),
            }
            for event, group in result.groupby("event_label")
        },
    }
    return result, metrics


def _plot_confusion(matrix: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    arr = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    threshold = arr.max() / 2.0 if arr.size and arr.max() > 0 else 0.5
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="white" if arr[i, j] > threshold else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    out_dir: Path = DEFAULT_OUT,
    n_estimators: int = 250,
    random_state: int = 20260504,
) -> PipelineResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    windows = pd.read_csv(dataset_dir / "agnostic_windows.csv")
    candidates = pd.read_csv(dataset_dir / "agnostic_candidate_rows.csv")
    window_cols = _feature_columns(windows, WINDOW_LABEL_COLUMNS)
    candidate_cols = _feature_columns(candidates, CANDIDATE_LABEL_COLUMNS)
    train_idx, test_idx = _split_by_sim(windows, random_state)
    x_train = windows.iloc[train_idx].reindex(columns=window_cols)
    x_test = windows.iloc[test_idx].reindex(columns=window_cols)
    y_event_train = windows.iloc[train_idx]["event_label"].to_numpy(dtype=int)
    y_event_test = windows.iloc[test_idx]["event_label"].to_numpy(dtype=int)
    detector = _model(n_estimators, random_state)
    classifier = _model(n_estimators, random_state + 1)
    t0 = time.perf_counter()
    detector.fit(x_train, (y_event_train != 0).astype(int))
    classifier.fit(x_train, y_event_train)
    train_seconds = time.perf_counter() - t0
    pred_abnormal = detector.predict(x_test).astype(int)
    pred_event = classifier.predict(x_test).astype(int)
    pred_event = np.where(pred_abnormal == 0, 0, pred_event)

    train_sims = set(windows.iloc[train_idx]["sim_id"].astype(str))
    cand_train = candidates[candidates["sim_id"].astype(str).isin(train_sims)].copy()
    cand_test = candidates[~candidates["sim_id"].astype(str).isin(train_sims)].copy()
    ranker = _model(n_estimators, random_state + 2)
    ranker.fit(cand_train.reindex(columns=candidate_cols), cand_train["target"].astype(int))
    cand_scores = ranker.predict_proba(cand_test.reindex(columns=candidate_cols))[:, 1]
    loc_predictions, loc_metrics = _rank_candidates(cand_test, cand_scores)
    loc_predictions.to_csv(out_dir / "sim_agnostic_location_predictions.csv", index=False)

    model_paths = {
        "detector": out_dir / "detector.joblib",
        "classifier": out_dir / "classifier.joblib",
        "candidate_ranker": out_dir / "candidate_ranker.joblib",
    }
    joblib.dump({"model": detector, "feature_columns": window_cols}, model_paths["detector"])
    joblib.dump({"model": classifier, "feature_columns": window_cols}, model_paths["classifier"])
    joblib.dump({"model": ranker, "feature_columns": candidate_cols}, model_paths["candidate_ranker"])
    report = {
        "dataset_dir": str(dataset_dir.resolve()),
        "n_windows": int(len(windows)),
        "n_candidate_rows": int(len(candidates)),
        "window_feature_count": int(len(window_cols)),
        "candidate_feature_count": int(len(candidate_cols)),
        "feature_contract": {
            "window_contains_specific_bus_names": bool(any(col.startswith("BUS") or "LINE" in col for col in window_cols)),
            "candidate_contains_specific_bus_names": bool(any(col.startswith("BUS") or "LINE" in col for col in candidate_cols)),
        },
        "sim_test": {
            "detection": _detection_metrics(y_event_test, pred_abnormal),
            "classification": _classification_metrics(y_event_test, pred_event),
            "localization": loc_metrics,
        },
        "efficiency": {
            "train_seconds": float(train_seconds),
            "model_size_mb": float(sum(path.stat().st_size for path in model_paths.values()) / (1024 * 1024)),
            "hardware": platform.platform(),
        },
    }
    _plot_confusion(report["sim_test"]["detection"]["confusion_matrix"], ["normal", "abnormal"], "Agnostic Detector", plots_dir / "detector_confusion.png")
    _plot_confusion(report["sim_test"]["classification"]["confusion_matrix"], [str(i) for i in range(9)], "Agnostic Classifier", plots_dir / "classifier_confusion.png")
    report_path = out_dir / "p13_report.json"
    write_json(report_path, report)
    result = PipelineResult(
        name="p13_train_agnostic_models",
        status="completed",
        outputs={
            "detector": str(model_paths["detector"].resolve()),
            "classifier": str(model_paths["classifier"].resolve()),
            "candidate_ranker": str(model_paths["candidate_ranker"].resolve()),
            "report": str(report_path.resolve()),
            "location_predictions": str((out_dir / "sim_agnostic_location_predictions.csv").resolve()),
        },
        metrics={
            "sim_detector_abnormal_f1": report["sim_test"]["detection"]["abnormal_f1"],
            "sim_classifier_macro_f1": report["sim_test"]["classification"]["macro_f1"],
            "sim_localizer_top1": report["sim_test"]["localization"]["exact_accuracy"],
            "sim_localizer_top3": report["sim_test"]["localization"]["top3_accuracy"],
        },
        notes=[
            "Models train only on placement-agnostic feature columns.",
            "Location labels and candidate identifiers are retained only for ranking/evaluation, not model features.",
        ],
    )
    write_json(out_dir / "p13_pipeline_result.json", result.to_dict())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train detector/classifier/localizer on agnostic augmented tables.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--random-state", type=int, default=20260504)
    args = parser.parse_args()
    result = train(args.dataset_dir, args.out_dir, args.n_estimators, args.random_state)
    print(json.dumps(json_safe(result.to_dict()), indent=2))


if __name__ == "__main__":
    main()

