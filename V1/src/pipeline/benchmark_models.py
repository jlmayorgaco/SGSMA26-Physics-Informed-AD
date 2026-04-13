"""Benchmark compact explainable event classifiers.

This harness compares exactly ten small models on the same 44-feature
physics-informed event windows.  Synthetic examples are added only to the
training matrix; a deterministic synthetic holdout is used for stress scoring.
The selected model is saved with a registry entry that includes the model-size
penalty used by the competition-style score.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.classifier.features import FEATURE_NAMES
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


SELECTION_LAMBDA = 0.05


class PhysicsRulesClassifier(BaseEstimator, ClassifierMixin):
    """Tiny deterministic classifier used as an interpretable lower bound."""

    def fit(self, X, y):  # noqa: N803 - sklearn convention
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y) if len(y) else np.arange(9)
        self.majority_label_ = int(pd.Series(y).mode().iloc[0]) if len(y) else 0
        self._idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
        return self

    def predict(self, X):  # noqa: N803 - sklearn convention
        X = np.asarray(X, dtype=float)
        out = np.full(X.shape[0], self.majority_label_, dtype=int)
        i = self._idx
        for row_idx, row in enumerate(X):
            nan_count = row[i["nan_count"]]
            pmu_missing = row[i["pmu_missing_count"]]
            v_max = row[i["VA_MAG_max"]]
            i_max = row[i["IA_MAG_max"]]
            rocof_max = row[i["ROCOF_max"]]
            freq_max = row[i["Freq_max"]]
            entropy = row[i["nu_entropy"]]
            state_line = row[i["state_line_24_23_energy"]]
            state_bus7 = row[i["state_bus7_energy"]]
            if pmu_missing > 0 and (v_max > 2_000 or i_max > 20 or rocof_max > 0.05):
                out[row_idx] = 6
            elif pmu_missing > 0 or nan_count > 20:
                out[row_idx] = 5
            elif entropy < 0.25 and v_max > 5_000:
                out[row_idx] = 7
            elif v_max > 20_000 and i_max > 80:
                out[row_idx] = 1
            elif state_line > max(1.0, state_bus7 * 1.2):
                out[row_idx] = 2
            elif abs(rocof_max) > 0.05 or abs(freq_max) > 0.03:
                out[row_idx] = 3
            elif v_max > 800 or i_max > 5:
                out[row_idx] = 4
            else:
                out[row_idx] = 0
        return out


def _lightgbm_model(**overrides):
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover - dependency is pinned
        raise RuntimeError("lightgbm is required for the benchmark model zoo") from exc
    params = {
        **LGBM_DEFAULTS,
        "verbose": -1,
        "random_state": overrides.pop("random_state", 42),
        **overrides,
    }
    return lgb.LGBMClassifier(**params)


def _model_zoo(seed: int) -> dict[str, object]:
    """Return exactly ten compact candidate classifiers."""
    return {
        "physics_rules": PhysicsRulesClassifier(),
        "gaussian_nb": GaussianNB(),
        "l1_logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                penalty="l1",
                solver="saga",
                C=0.7,
                class_weight="balanced",
                max_iter=1200,
                random_state=seed,
            ),
        ),
        "linear_svm_sgd": make_pipeline(
            RobustScaler(),
            SGDClassifier(
                loss="hinge",
                alpha=5e-4,
                class_weight="balanced",
                max_iter=2000,
                tol=1e-4,
                random_state=seed,
            ),
        ),
        "pruned_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
        ),
        "small_random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "small_extra_trees": ExtraTreesClassifier(
            n_estimators=160,
            max_depth=8,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(
                max_iter=160,
                max_leaf_nodes=15,
                learning_rate=0.05,
                l2_regularization=0.02,
                random_state=seed,
            ),
        ),
        "lightgbm_tiny": _lightgbm_model(
            n_estimators=80,
            num_leaves=15,
            learning_rate=0.06,
            min_data_in_leaf=4,
            random_state=seed,
        ),
        "lightgbm_small": _lightgbm_model(
            n_estimators=180,
            num_leaves=31,
            learning_rate=0.04,
            min_data_in_leaf=3,
            random_state=seed,
        ),
    }


def _labels_for_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> list[int]:
    return sorted(set(np.asarray(y_true, dtype=int).tolist()) | set(np.asarray(y_pred, dtype=int).tolist()))


def _unwrap_model(model: object) -> object:
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model


def _tree_param_count(model: object) -> int:
    if hasattr(model, "tree_"):
        return int(model.tree_.node_count * 2)
    if hasattr(model, "estimators_"):
        total = 0
        for est in np.ravel(model.estimators_):
            if hasattr(est, "tree_"):
                total += int(est.tree_.node_count * 2)
        return total
    return 0


def parameter_count(model: object) -> int:
    """Best-effort effective parameter count for score/size reporting."""
    core = _unwrap_model(model)
    if isinstance(core, PhysicsRulesClassifier):
        return 0
    if hasattr(core, "coef_"):
        coef = np.asarray(core.coef_)
        intercept = np.asarray(getattr(core, "intercept_", []))
        return int(np.count_nonzero(coef) + np.count_nonzero(intercept))
    if isinstance(core, GaussianNB) and hasattr(core, "theta_"):
        return int(np.size(core.theta_) + np.size(core.var_) + np.size(core.class_prior_))
    if hasattr(core, "booster_"):
        try:
            dump = core.booster_.dump_model()
            leaves = sum(int(t["num_leaves"]) for t in dump.get("tree_info", []))
            return int(max(1, leaves * 2))
        except Exception:
            return int(getattr(core, "n_estimators_", 1) * getattr(core, "num_leaves", 1) * 2)
    tree_count = _tree_param_count(core)
    if tree_count:
        return tree_count
    if isinstance(core, HistGradientBoostingClassifier) and hasattr(core, "_predictors"):
        total = 0
        for predictors in core._predictors:
            for pred in predictors:
                total += int(len(pred.nodes) * 2)
        return total
    return max(1, len(pickle.dumps(model)) // 128)


def serialized_size_bytes(model: object) -> int:
    return int(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)))


def _split_synthetic(
    X_syn: np.ndarray,
    y_syn: np.ndarray,
    *,
    seed: int,
    holdout_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y_syn) == 0:
        return X_syn, y_syn, X_syn, y_syn
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y_syn))
    n_hold = max(1, int(round(len(y_syn) * holdout_frac))) if len(y_syn) > 5 else 0
    hold_idx = order[:n_hold]
    train_idx = order[n_hold:]
    return X_syn[train_idx], y_syn[train_idx], X_syn[hold_idx], y_syn[hold_idx]


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = _labels_for_metrics(y_true, y_pred)
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def _score_model(model, X, y) -> tuple[float, float, float]:
    if len(y) == 0:
        return 0.0, 0.0, 0.0
    y_pred = model.predict(X)
    labels = _labels_for_metrics(y, y_pred)
    return (
        float(accuracy_score(y, y_pred)),
        float(f1_score(y, y_pred, labels=labels, average="macro", zero_division=0)),
        float(f1_score(y, y_pred, labels=labels, average="weighted", zero_division=0)),
    )


def run_benchmark(
    *,
    data_dir: Path,
    raw_path: Path,
    synthetic_dir: Path | None,
    feature_cache: Path | None,
    out_json: Path,
    out_csv: Path,
    model_dir: Path,
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
    X_train_real, y_train_real = _collect_transition_events(
        df_train,
        J_cols,
        fps,
        window_sec=3.0,
        ignore_labels=set(),
        add_normal_samples=True,
        state_estimator=state_estimator,
    )
    X_eval_real, y_eval_real = _collect_transition_events(
        df,
        J_cols,
        fps,
        window_sec=3.0,
        ignore_labels=set(),
        add_normal_samples=True,
        state_estimator=state_estimator,
    )

    X_syn = np.empty((0, X_train_real.shape[1] if len(X_train_real) else len(FEATURE_NAMES)), dtype=float)
    y_syn = np.empty(0, dtype=int)
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
            feature_cache_path=feature_cache,
        )

    X_syn_train, y_syn_train, X_syn_eval, y_syn_eval = _split_synthetic(
        X_syn,
        y_syn,
        seed=seed,
    )
    X_train = np.concatenate([X_train_real, X_syn_train], axis=0) if len(X_syn_train) else X_train_real
    y_train = np.concatenate([y_train_real, y_syn_train], axis=0) if len(y_syn_train) else y_train_real

    rows: list[dict] = []
    fitted_models: dict[str, object] = {}
    for name, model in _model_zoo(seed).items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        visible_accuracy, visible_macro, visible_weighted = _score_model(model, X_eval_real, y_eval_real)
        visible_predict_sec = time.perf_counter() - t1
        syn_accuracy, syn_macro, syn_weighted = _score_model(model, X_syn_eval, y_syn_eval)

        n_params = parameter_count(model)
        size = serialized_size_bytes(model)
        penalty = SELECTION_LAMBDA * float(np.log10(n_params + 1))
        selection_score = 0.7 * visible_macro + 0.3 * syn_macro - penalty
        row = {
            "model": name,
            "train_windows": int(len(y_train)),
            "real_train_windows": int(len(y_train_real)),
            "synthetic_train_windows": int(len(y_syn_train)),
            "visible_eval_windows": int(len(y_eval_real)),
            "synthetic_holdout_windows": int(len(y_syn_eval)),
            "train_labels": {str(int(k)): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
            "accuracy": visible_accuracy,
            "macro_f1": visible_macro,
            "weighted_f1": visible_weighted,
            "visible_real_macro_f1": visible_macro,
            "visible_real_weighted_f1": visible_weighted,
            "synthetic_holdout_accuracy": syn_accuracy,
            "synthetic_holdout_macro_f1": syn_macro,
            "synthetic_holdout_weighted_f1": syn_weighted,
            "parameter_count": int(n_params),
            "serialized_size_bytes": int(size),
            "model_size_kb": float(size / 1024.0),
            "penalty": float(penalty),
            "penalized_score": float(visible_macro - penalty),
            "selection_score": float(selection_score),
            "fit_sec": float(fit_sec),
            "predict_sec": float(visible_predict_sec),
        }
        rows.append(row)
        fitted_models[name] = model

    rows.sort(key=lambda row: row["selection_score"], reverse=True)
    best = rows[0]
    model_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = model_dir / "best_event_classifier.pkl"
    registry_path = model_dir / "model_registry.json"
    with best_model_path.open("wb") as handle:
        pickle.dump(
            {
                "model": fitted_models[best["model"]],
                "selected_model": best["model"],
                "feature_names": FEATURE_NAMES,
                "selection_rule": "0.7*visible_real_macro_f1 + 0.3*synthetic_holdout_macro_f1 - 0.05*log10(parameter_count+1)",
                "benchmark_row": best,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    registry = {
        "selected_model": best["model"],
        "best_model_path": str(best_model_path),
        "selection_rule": "0.7*visible_real_macro_f1 + 0.3*synthetic_holdout_macro_f1 - 0.05*log10(parameter_count+1)",
        "n_models": len(rows),
        "rows": rows,
    }
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--raw", type=Path, default=Path("data/metadata/IEEE 39 Bus Power System.raw"))
    parser.add_argument("--synthetic", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--feature-cache", type=Path, default=Path("data/synthetic/features_v1_5000.npz"))
    parser.add_argument("--out-json", type=Path, default=Path("report/model_benchmark.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("report/model_benchmark.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = run_benchmark(
        data_dir=args.data,
        raw_path=args.raw,
        synthetic_dir=args.synthetic,
        feature_cache=args.feature_cache,
        out_json=args.out_json,
        out_csv=args.out_csv,
        model_dir=args.model_dir,
        seed=args.seed,
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
