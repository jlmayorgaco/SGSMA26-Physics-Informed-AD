from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data_factory.train_hierarchical_pipeline_v2 import _predict_hierarchical, _predict_locations
from src.models.localizer import HybridLocalizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class SGSMAFinalModel:
    """Final detector, classifier, and localizer bundle.

    The model expects already extracted V2 base features. Dynamic features are
    optional at call time, but the selected localizer uses them when available.
    """

    def __init__(self, model_dir: Path = DEFAULT_MODEL_DIR) -> None:
        self.model_dir = Path(model_dir)
        self.config = _read_json(self.model_dir / "final_model_config.json")
        self.hierarchical_config = _read_json(self.model_dir / "detector" / "hierarchical_model_config.json")
        self.base_feature_cols = _read_json(self.model_dir / "feature_columns.json")
        self.physical_model = joblib.load(self.model_dir / "classifiers" / "physical_event_classifier.joblib")
        self.bad_data_model = joblib.load(self.model_dir / "detector" / "bad_data_detector.joblib")
        self.missing_model = joblib.load(self.model_dir / "detector" / "missing_composition_detector.joblib")
        localizer_bundle = joblib.load(self.model_dir / "localizer" / "final_dynamic_localizers.joblib")
        self.localizers: dict[str, Any] = localizer_bundle["localizers"]
        self.localizer_feature_cols: list[str] = list(localizer_bundle["feature_cols"])
        self.variant = str(localizer_bundle["variant"])
        hybrid_path = self.model_dir / "localizer" / "final_hybrid_localizer.joblib"
        if hybrid_path.exists():
            self.hybrid_localizer: HybridLocalizer = joblib.load(hybrid_path)
        elif "hybrid_localizer" in localizer_bundle:
            self.hybrid_localizer = localizer_bundle["hybrid_localizer"]
        else:
            self.hybrid_localizer = HybridLocalizer.from_localizers(self.localizers)

    def predict_from_features(
        self,
        base_features: pd.DataFrame,
        dynamic_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        base = base_features.reset_index(drop=True).copy()
        x_base = base.reindex(columns=self.base_feature_cols, fill_value=np.nan)
        pred_event, pred_physical, bad_score, physical_confidence = _predict_hierarchical(
            x_base,
            base,
            self.physical_model,
            self.bad_data_model,
            self.missing_model,
            float(self.hierarchical_config["bad_data_probability"]),
            float(self.hierarchical_config["missing_composition_probability"]),
            float(self.hierarchical_config.get("missing_physical_confidence", 0.0)),
        )
        if dynamic_features is None:
            dynamic = pd.DataFrame(index=base.index)
        else:
            dynamic = dynamic_features.reset_index(drop=True).copy()
        x_localizer = pd.concat(
            [
                base.reindex(columns=self.base_feature_cols, fill_value=np.nan),
                dynamic,
            ],
            axis=1,
        ).reindex(columns=self.localizer_feature_cols, fill_value=np.nan)
        if bool(self.config.get("use_hybrid_top1", False)):
            pred_location = self.hybrid_localizer.predict(x_localizer, pred_event, pred_physical)
        else:
            pred_location = _predict_locations(x_localizer, pred_event, pred_physical, self.localizers)
        return pd.DataFrame(
            {
                "pred_event": pred_event.astype(int),
                "pred_abnormal": (pred_event != 0).astype(int),
                "pred_physical_event": pred_physical.astype(int),
                "pred_location": pred_location,
                "bad_data_score": bad_score,
                "physical_confidence": physical_confidence,
            }
        )

    def predict_location_topk(
        self,
        base_features: pd.DataFrame,
        dynamic_features: pd.DataFrame | None = None,
        k: int = 3,
    ) -> list[list[dict[str, Any]]]:
        base = base_features.reset_index(drop=True).copy()
        x_base = base.reindex(columns=self.base_feature_cols, fill_value=np.nan)
        pred_event, pred_physical, _, _ = _predict_hierarchical(
            x_base,
            base,
            self.physical_model,
            self.bad_data_model,
            self.missing_model,
            float(self.hierarchical_config["bad_data_probability"]),
            float(self.hierarchical_config["missing_composition_probability"]),
            float(self.hierarchical_config.get("missing_physical_confidence", 0.0)),
        )
        if dynamic_features is None:
            dynamic = pd.DataFrame(index=base.index)
        else:
            dynamic = dynamic_features.reset_index(drop=True).copy()
        x_localizer = pd.concat(
            [base.reindex(columns=self.base_feature_cols, fill_value=np.nan), dynamic],
            axis=1,
        ).reindex(columns=self.localizer_feature_cols, fill_value=np.nan)
        return self.hybrid_localizer.predict_topk(x_localizer, pred_event, pred_physical, k=k)


def load_final_model(model_dir: Path = DEFAULT_MODEL_DIR) -> SGSMAFinalModel:
    return SGSMAFinalModel(model_dir)
