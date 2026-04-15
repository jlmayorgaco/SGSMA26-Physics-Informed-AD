"""Topology-aware graph message-passing model for the V2 PMU pipeline."""

from __future__ import annotations

from .feature_augmented import FeatureAugmentedClassifier
from .feature_layers import GraphTopologyFeatureLayer


class GraphTopologyClassifier(FeatureAugmentedClassifier):
    """Use IEEE-39 graph diffusion features before a compact tree classifier."""

    def __init__(self, seed: int = 0, base_model: str = "extratrees") -> None:
        super().__init__(
            feature_layer=GraphTopologyFeatureLayer(propagation_steps=3, include_original=True),
            seed=seed,
            base_model=base_model,
            n_estimators=220,
            max_depth=16,
            min_samples_leaf=2,
        )


def make_graph_topology_classifier(seed: int) -> GraphTopologyClassifier:
    return GraphTopologyClassifier(seed=seed)
