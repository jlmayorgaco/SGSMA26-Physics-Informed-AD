from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physics_feature_extractor import PhysicsFeatureExtractor, load_bus_frames, missing_data_summary


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "unified_event_detector" / "unified_event_detector.pkl"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if np.isfinite(f) else None
    return value


class UnifiedEventDetector:
    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        with self.model_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.model = payload["model"]
        self.feature_columns = list(payload["feature_columns"])
        self.label_meaning = {int(k): v for k, v in payload["label_meaning"].items()}
        self.extractor = PhysicsFeatureExtractor()

    def predict_frames(self, bus_frames: dict[str, pd.DataFrame], sample_id: str = "") -> dict[str, Any]:
        row = self.extractor.summarize_frames(bus_frames, sample_id=sample_id)
        X = pd.DataFrame([row])
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_columns]
        ml_pred = int(self.model.predict(X)[0])
        probabilities: dict[str, float] = {}
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            classes = [int(value) for value in self.model.named_steps["model"].classes_]
            probabilities = {str(cls): float(prob) for cls, prob in zip(classes, probs)}
        missing = missing_data_summary(bus_frames)
        pred = 6 if bool(missing["missing_data"]) and ml_pred in {1, 2, 3, 4} else ml_pred
        return {
            "sample_id": sample_id,
            "pred_event": pred,
            "pred_label": f"event{pred}",
            "pred_description": self.label_meaning.get(pred, "unknown"),
            "ml_pred_event_before_physics_postprocess": ml_pred,
            "probabilities": probabilities,
            "missing_data": missing["missing_data"],
            "missing_buses": missing["missing_buses"],
            "key_features": {
                key: row.get(key)
                for key in [
                    "GLOBAL_v_min_min",
                    "GLOBAL_i_max_max",
                    "GLOBAL_di_abs_max_max",
                    "GLOBAL_freq_span_max",
                    "GLOBAL_rocof_abs_max_max",
                    "missing_fraction",
                    "missing_bus_count",
                ]
                if key in row
            },
        }

    def predict_chunk(self, chunk_dir: Path | str) -> dict[str, Any]:
        chunk_path = Path(chunk_dir)
        return self.predict_frames(load_bus_frames(chunk_path), sample_id=chunk_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified physics + ML event detector.")
    parser.add_argument("--chunk-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = UnifiedEventDetector(model_path=args.model_path)
    print(json.dumps(_json_safe(detector.predict_chunk(args.chunk_dir)), indent=2))


if __name__ == "__main__":
    main()
