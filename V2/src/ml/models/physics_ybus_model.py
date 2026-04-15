"""Physics/Ybus feature-layer model for the V2 PMU pipeline."""

from __future__ import annotations

from .feature_augmented import FeatureAugmentedClassifier
from .feature_layers import PhysicsYbusFeatureLayer


class PhysicsYbusClassifier(FeatureAugmentedClassifier):
    """Estimate hidden voltages with Ybus/KCL cues before classification."""

    def __init__(self, seed: int = 0, base_model: str = "extratrees") -> None:
        super().__init__(
            feature_layer=PhysicsYbusFeatureLayer(include_original=True),
            seed=seed,
            base_model=base_model,
            n_estimators=220,
            max_depth=16,
            min_samples_leaf=2,
        )


def make_physics_ybus_classifier(seed: int) -> PhysicsYbusClassifier:
    return PhysicsYbusClassifier(seed=seed)
