from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

from physics_feature_extractor import PhysicsFeatureExtractor, event_label_from_name  # noqa: E402


OUT_DIR = ROOT / "models" / "unified_event_detector"
MODEL_PATH = OUT_DIR / "unified_event_detector.pkl"
CONFIG_PATH = OUT_DIR / "unified_event_detector_config.json"
DATASET_PATH = OUT_DIR / "unified_event_detector_training_matrix.csv"
REPORT_PATH = OUT_DIR / "unified_event_detector_report.json"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    return value


def _classify_training_label(chunk_name: str, row: dict[str, Any]) -> int:
    label = event_label_from_name(chunk_name)
    if label is None:
        raise ValueError(f"Cannot parse event label from {chunk_name}")
    # Guideline-consistent proxy: missing data concurrent with a physical event
    # is event6-like. RAW0001 labels chunk19 as event3 but its BUS29 PMU is missing.
    if bool(row["missing_data"]) and label in {1, 2, 3, 4}:
        return 6
    return int(label)


def build_chunk_dataset() -> pd.DataFrame:
    extractor = PhysicsFeatureExtractor()
    rows: list[dict[str, Any]] = []
    for chunk_dir in sorted(
        [path for path in (ROOT / "data" / "chunked").iterdir() if path.is_dir() and "_event" in path.name],
        key=lambda path: int(path.name.split("_", 1)[0].replace("chunk", "")),
    ):
        row = extractor.summarize_chunk(chunk_dir)
        raw_label = event_label_from_name(chunk_dir.name)
        row["source"] = "raw0001_chunk"
        row["chunk_name"] = chunk_dir.name
        row["raw_event_label"] = raw_label
        row["target_event"] = _classify_training_label(chunk_dir.name, row)
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"sample_id", "source", "chunk_name", "raw_event_label", "target_event", "missing_data"}
    return [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]


def _train_models(df: pd.DataFrame, feature_cols: list[str]) -> tuple[str, Pipeline, dict[str, Any]]:
    X = df[feature_cols]
    y = df["target_event"].astype(int).to_numpy()
    models: dict[str, Pipeline] = {
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "multinomial_logistic": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    results: dict[str, Any] = {}
    best_name = ""
    best_model: Pipeline | None = None
    best_score = (-1.0, -1.0)
    for name, model in models.items():
        model.fit(X, y)
        pred = model.predict(X)
        labels = sorted(set(y.tolist()) | set(pred.tolist()))
        metrics = {
            "accuracy_train_resubstitution": float(accuracy_score(y, pred)),
            "macro_f1_train_resubstitution": float(f1_score(y, pred, average="macro", zero_division=0)),
            "labels": labels,
            "confusion_rows_true_cols_pred": confusion_matrix(y, pred, labels=labels).tolist(),
        }
        results[name] = metrics
        score = (metrics["macro_f1_train_resubstitution"], metrics["accuracy_train_resubstitution"])
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model
    if best_model is None:
        raise RuntimeError("No model trained")
    return best_name, best_model, results


def _top_features(model: Pipeline, feature_cols: list[str], n: int = 40) -> list[dict[str, Any]]:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.nanmean(np.abs(np.asarray(estimator.coef_, dtype=float)), axis=0)
    else:
        return []
    order = np.argsort(values)[::-1][:n]
    return [{"feature": feature_cols[idx], "importance": float(values[idx])} for idx in order if values[idx] > 0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_chunk_dataset()
    feature_cols = _feature_columns(df)
    best_name, model, model_results = _train_models(df, feature_cols)
    payload = {
        "model": model,
        "feature_columns": feature_cols,
        "label_meaning": {
            "0": "normal/event0",
            "1": "three-phase fault/event1",
            "2": "line outage/event2",
            "3": "generation change/event3",
            "4": "load change/event4",
            "5": "missing-data only/event5",
            "6": "missing-data plus physical event/event6 proxy",
            "7": "bad data/event7",
        },
    }
    with MODEL_PATH.open("wb") as handle:
        pickle.dump(payload, handle)
    df.to_csv(DATASET_PATH, index=False)

    report = {
        "model_path": str(MODEL_PATH.resolve()),
        "training_matrix_path": str(DATASET_PATH.resolve()),
        "best_model": best_name,
        "n_samples": int(len(df)),
        "n_features": int(len(feature_cols)),
        "raw001_target_distribution": {str(k): int(v) for k, v in df["target_event"].value_counts().sort_index().items()},
        "model_results": model_results,
        "top_features": _top_features(model, feature_cols),
        "validation_warning": "This first unified ML model is trained on RAW001 chunks only. It is physics-feature based, but RAW002 generalization still requires more simulated/profiled scenarios and holdout validation.",
    }
    REPORT_PATH.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    CONFIG_PATH.write_text(
        json.dumps(
            _json_safe(
                {
                    "schema_version": 1,
                    "model_name": "unified_physics_ml_event_detector",
                    "model_path": str(MODEL_PATH.resolve()),
                    "feature_columns": feature_cols,
                    "label_meaning": payload["label_meaning"],
                    "missing_data_rule": "DATA_PRESENT == 0 or any PMU measurement NaN",
                    "event6_rule_for_training": "missing_data AND raw physical event label in {1,2,3,4}",
                    "best_model": best_name,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(report), indent=2))


if __name__ == "__main__":
    main()
