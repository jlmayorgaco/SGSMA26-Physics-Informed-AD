"""Benchmark multiple event classifiers on the current feature pipeline.

This is an experiment harness, not the default submission path.  It compares a
small model zoo on the same physics-informed 44-feature event windows used by
the production LightGBM classifier, with synthetic events added only to the
training matrix.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier

from src.classifier.train_lgbm import (
    LGBM_DEFAULTS,
    _build_detector,
    _collect_transition_events,
    _load_synthetic_events,
)
from src.estimator.calibration import channel_cols
from src.eval.splits import make_splits
from src.grid.jacobians import bus_sensitivity_columns, compute_jacobians
from src.grid.load_case import load_case
from src.io.load_csv import load_all


def _model_zoo(seed: int) -> dict[str, object]:
    models: dict[str, object] = {}
    try:
        import lightgbm as lgb

        models["lightgbm"] = lgb.LGBMClassifier(
            **{
                **LGBM_DEFAULTS,
                "n_estimators": 300,
                "learning_rate": 0.04,
                "verbose": -1,
                "random_state": seed,
            }
        )
    except ImportError:
        pass

    models["extra_trees"] = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    models["random_forest"] = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    models["hist_gradient_boosting"] = make_pipeline(
        StandardScaler(),
        HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            l2_regularization=0.01,
            random_state=seed,
        ),
    )
    models["rbf_svm"] = make_pipeline(
        RobustScaler(),
        SVC(C=10.0, gamma="scale", class_weight="balanced", random_state=seed),
    )
    return models


def _labels_for_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[int]:
    return sorted(set(np.asarray(y_true, dtype=int).tolist()) | set(np.asarray(y_pred, dtype=int).tolist()))


def run_benchmark(
    *,
    data_dir: Path,
    raw_path: Path,
    synthetic_dir: Path | None,
    out_json: Path,
    out_csv: Path,
    seed: int = 42,
    fps: float = 30.0,
) -> list[dict]:
    df = load_all(data_dir)
    splits = make_splits(df, fps=fps)
    df_train = df.iloc[splits["train"]].reset_index(drop=True)
    df_val = df.iloc[splits["val"]].reset_index(drop=True)

    grid = load_case(raw_path)
    J_cols = bus_sensitivity_columns(compute_jacobians(grid), grid)
    from src.estimator.topology_state import TopologyStateEstimator

    state_estimator = TopologyStateEstimator(grid)

    det, h0_flat, offset, R = _build_detector(df_train)
    X_train, y_train = _collect_transition_events(
        df_train,
        J_cols,
        fps,
        window_sec=3.0,
        ignore_labels=set(),
        add_normal_samples=True,
        state_estimator=state_estimator,
    )
    if synthetic_dir is not None and synthetic_dir.exists():
        X_syn, y_syn = _load_synthetic_events(
            synthetic_dir,
            h0_flat,
            offset,
            R,
            channel_cols(),
            J_cols,
            fps,
            window_sec=3.0,
            state_estimator=state_estimator,
        )
        if len(y_syn):
            X_train = np.concatenate([X_train, X_syn], axis=0)
            y_train = np.concatenate([y_train, y_syn], axis=0)

    # The public file has very few validation-block events, so benchmark all
    # visible event transitions as a sanity probe. Hidden-test claims should be
    # made from held-out data only; this artifact is for model selection stress.
    X_eval, y_eval = _collect_transition_events(
        df,
        J_cols,
        fps,
        window_sec=3.0,
        ignore_labels=set(),
        add_normal_samples=True,
        state_estimator=state_estimator,
    )

    rows: list[dict] = []
    for name, model in _model_zoo(seed).items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        y_pred = model.predict(X_eval)
        predict_sec = time.perf_counter() - t1
        labels = _labels_for_metrics(y_eval, y_pred)
        rows.append(
            {
                "model": name,
                "train_windows": int(len(y_train)),
                "eval_windows": int(len(y_eval)),
                "train_labels": {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
                "eval_labels": {int(k): int(v) for k, v in zip(*np.unique(y_eval, return_counts=True))},
                "accuracy": float(accuracy_score(y_eval, y_pred)),
                "macro_f1": float(f1_score(y_eval, y_pred, labels=labels, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_eval, y_pred, labels=labels, average="weighted", zero_division=0)),
                "fit_sec": float(fit_sec),
                "predict_sec": float(predict_sec),
            }
        )

    rows.sort(key=lambda row: (row["macro_f1"], row["weighted_f1"]), reverse=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--raw", type=Path, default=Path("data/metadata/IEEE 39 Bus Power System.raw"))
    parser.add_argument("--synthetic", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--out-json", type=Path, default=Path("report/model_benchmark.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("report/model_benchmark.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = run_benchmark(
        data_dir=args.data,
        raw_path=args.raw,
        synthetic_dir=args.synthetic,
        out_json=args.out_json,
        out_csv=args.out_csv,
        seed=args.seed,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
