"""Histogram gradient boosting model family."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier


def make_histgb_classifier(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.06,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=int(seed),
    )
