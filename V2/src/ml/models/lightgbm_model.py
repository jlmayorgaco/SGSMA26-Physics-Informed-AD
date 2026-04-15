"""LightGBM model family with a safe import fallback."""

from __future__ import annotations


try:  # pragma: no cover - optional dependency.
    from lightgbm import LGBMClassifier

    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - fallback handled by registry.
    LGBMClassifier = None
    HAS_LIGHTGBM = False


def make_lightgbm_classifier(seed: int, n_classes: int = 3):
    if not HAS_LIGHTGBM:
        raise RuntimeError("lightgbm is not installed")
    objective = "binary" if int(n_classes) == 2 else "multiclass"
    return LGBMClassifier(
        objective=objective,
        n_estimators=150,
        learning_rate=0.055,
        num_leaves=15,
        max_depth=5,
        min_child_samples=2,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=int(seed),
        n_jobs=-1,
        verbose=-1,
    )
