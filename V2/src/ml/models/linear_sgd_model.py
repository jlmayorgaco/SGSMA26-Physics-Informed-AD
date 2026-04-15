"""Fast linear probabilistic baseline."""

from __future__ import annotations

from sklearn.linear_model import SGDClassifier


def make_linear_sgd_classifier(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        alpha=1e-4,
        l1_ratio=0.15,
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=int(seed),
        n_jobs=-1,
    )
