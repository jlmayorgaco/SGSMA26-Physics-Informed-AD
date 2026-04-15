"""Central registry for V2 model families."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from .extratrees_model import make_extratrees_bus_state_classifier, make_extratrees_classifier
from .graph_topology_model import make_graph_topology_classifier
from .histgb_model import make_histgb_classifier
from .lightgbm_model import HAS_LIGHTGBM, make_lightgbm_classifier
from .linear_sgd_model import make_linear_sgd_classifier
from .mlp_model import make_mlp_classifier
from .physics_ybus_model import make_physics_ybus_classifier
from .randomforest_model import make_randomforest_classifier


SUPPORTED_MODEL_NAMES = [
    "lightgbm",
    "histgb",
    "extratrees",
    "randomforest",
    "linear_sgd",
    "mlp",
    "graph_topology",
    "physics_ybus",
]

FEATURE_LAYER_MODELS = {"graph_topology", "physics_ybus"}
TREE_BUS_STATE_HEAD_MODELS = {"linear_sgd", "mlp"}


def resolve_model_name(model_name: str) -> str:
    name = str(model_name).strip().lower()
    if name == "lightgbm" and not HAS_LIGHTGBM:
        return "histgb"
    return name


def build_event_estimator(model_name: str, n_classes: int, seed: int) -> Pipeline:
    """Return an estimator for event or location classification."""

    resolved = resolve_model_name(model_name)
    if n_classes < 2:
        return Pipeline([("classifier", DummyClassifier(strategy="most_frequent"))])
    if resolved in FEATURE_LAYER_MODELS:
        return Pipeline([("classifier", _build_classifier(resolved, seed, n_classes))])
    return Pipeline(
        [
            ("imputer", _make_imputer()),
            ("classifier", _build_classifier(resolved, seed, n_classes)),
        ]
    )


def build_bus_state_estimator(model_name: str, seed: int) -> Pipeline:
    """Return an estimator for the 39-bus multi-output state target."""

    resolved = resolve_model_name(model_name)
    if resolved in FEATURE_LAYER_MODELS:
        return Pipeline([("classifier", _build_classifier(resolved, seed + 101, 9))])
    if resolved in {"randomforest", "extratrees"} | TREE_BUS_STATE_HEAD_MODELS:
        return Pipeline([("imputer", _make_imputer()), ("classifier", _build_bus_state_tree(resolved, seed))])
    classifier = MultiOutputClassifier(_build_classifier(resolved, seed + 101, 9), n_jobs=-1)
    return Pipeline([("imputer", _make_imputer()), ("classifier", classifier)])


def _make_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


def _build_classifier(model_name: str, seed: int, n_classes: int = 3):
    if model_name == "lightgbm":
        return make_lightgbm_classifier(seed, n_classes)
    if model_name == "histgb":
        return make_histgb_classifier(seed)
    if model_name == "extratrees":
        return make_extratrees_classifier(seed)
    if model_name == "randomforest":
        return make_randomforest_classifier(seed)
    if model_name == "linear_sgd":
        return make_linear_sgd_classifier(seed)
    if model_name == "mlp":
        return make_mlp_classifier(seed)
    if model_name == "graph_topology":
        return make_graph_topology_classifier(seed)
    if model_name == "physics_ybus":
        return make_physics_ybus_classifier(seed)
    raise ValueError(f"Unsupported model: {model_name}. Supported: {SUPPORTED_MODEL_NAMES}")


def _build_bus_state_tree(model_name: str, seed: int):
    if model_name == "randomforest":
        return make_randomforest_classifier(seed + 101)
    return make_extratrees_bus_state_classifier(seed)
