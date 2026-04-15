"""ExtraTrees model family."""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier


def make_extratrees_classifier(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=180,
        max_depth=12,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=-1,
    )


def make_extratrees_bus_state_classifier(seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=140,
        max_depth=14,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=int(seed) + 101,
        n_jobs=-1,
    )
