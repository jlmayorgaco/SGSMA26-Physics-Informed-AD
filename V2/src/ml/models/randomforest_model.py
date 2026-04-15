"""Random forest model family."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def make_randomforest_classifier(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=220,
        max_depth=14,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=int(seed),
        n_jobs=-1,
    )
