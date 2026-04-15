"""Small neural-network baseline using sklearn MLP."""

from __future__ import annotations

from sklearn.neural_network import MLPClassifier


def make_mlp_classifier(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(80, 24),
        activation="relu",
        alpha=5e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=120,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=int(seed),
    )
