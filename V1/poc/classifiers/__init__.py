"""Classifier registry for the POC ablation."""

from poc.classifiers.classifier_c1_rules import RuleBasedClassifier
from poc.classifiers.classifier_c2_lgbm import LightGBMClassifierPOC
from poc.classifiers.classifier_c3_tcn import TemporalConvNetClassifier
from poc.classifiers.classifier_c4_transformer import TransformerEncoderClassifier

CLASSIFIERS = {
    "C1_RULES": RuleBasedClassifier,
    "C2_LGBM": LightGBMClassifierPOC,
    "C3_TCN": TemporalConvNetClassifier,
    "C4_TRANSFORMER": TransformerEncoderClassifier,
}

__all__ = [
    "RuleBasedClassifier",
    "LightGBMClassifierPOC",
    "TemporalConvNetClassifier",
    "TransformerEncoderClassifier",
    "CLASSIFIERS",
]

