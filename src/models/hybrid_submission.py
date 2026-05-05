"""Production submission router for the strongest available SGSMA models."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_factory.dynamic_feature_extractor_v3 import extract_dynamic_features_from_frames
from src.data_factory.feature_extractor_v2 import extract_window_features_v2
from src.data_factory.final_model import SGSMAFinalModel
from src.features.pmu_discovery import discover_bus_csvs
from src.models.bus_agnostic import run_bus_agnostic_prediction


OFFICIAL_COLUMNS = ["TIMESTAMP", "Bus", "Predicted_Event", "Predicted_Location"]


def _load_frames(input_dir: Path) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for bus, path in discover_bus_csvs(input_dir).items():
        frame = pd.read_csv(path).sort_values("TIMESTAMP").reset_index(drop=True)
        frames[int(bus)] = frame
    if not frames:
        raise ValueError(f"No Bus*.csv PMU files were found in {input_dir}")
    return frames


def _duration_seconds(frames: dict[int, pd.DataFrame]) -> float:
    starts = []
    ends = []
    for frame in frames.values():
        if "TIMESTAMP" not in frame.columns or frame.empty:
            continue
        ts = pd.to_numeric(frame["TIMESTAMP"], errors="coerce")
        starts.append(float(ts.min()))
        ends.append(float(ts.max()))
    if not starts or not ends:
        return 0.0
    return max(ends) - min(starts)


def _prefix_numeric(features: dict[str, Any], prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in features.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            out[f"{prefix}__{key}"] = float(value)
    return out


def _build_feature_rows(frames: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    row: dict[str, float] = {}
    bus_feature_rows = []
    for bus, frame in sorted(frames.items()):
        segment = frame.copy()
        ts = pd.to_numeric(segment["TIMESTAMP"], errors="coerce")
        segment["TIMESTAMP"] = ts - float(ts.min())
        features = extract_window_features_v2(segment)
        row.update(_prefix_numeric(features, f"BUS{bus}"))
        bus_feature_rows.append(features)

    if bus_feature_rows:
        numeric = pd.DataFrame(bus_feature_rows).select_dtypes(include=[np.number])
        for col in numeric.columns:
            row[f"GLOBAL__{col}__mean"] = float(numeric[col].mean())
            row[f"GLOBAL__{col}__max"] = float(numeric[col].max())
            row[f"GLOBAL__{col}__min"] = float(numeric[col].min())
            row[f"GLOBAL__{col}__std"] = float(numeric[col].std(ddof=0))

    dynamic = extract_dynamic_features_from_frames(frames, include_blocks=("rolling", "rls_kalman", "graph_temporal"))
    return pd.DataFrame([row]), pd.DataFrame([dynamic])


def _broadcast_prediction(
    frames: dict[int, pd.DataFrame],
    event: int,
    location: str,
    output_csv: Path,
    diagnostics: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for bus, frame in sorted(frames.items()):
        timestamps = pd.to_numeric(frame["TIMESTAMP"], errors="coerce").to_numpy(dtype=float)
        rows.append(
            pd.DataFrame(
                {
                    "TIMESTAMP": timestamps,
                    "Bus": int(bus),
                    "Predicted_Event": int(event),
                    "Predicted_Location": str(location),
                }
            )
        )
    predictions = pd.concat(rows, ignore_index=True).sort_values(["TIMESTAMP", "Bus"], kind="mergesort")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions[OFFICIAL_COLUMNS].to_csv(output_csv, index=False)
    diagnostics["summary"] = {
        "n_rows": int(len(predictions)),
        "n_timestamps": int(predictions["TIMESTAMP"].nunique()),
        "event_counts": {str(k): int(v) for k, v in predictions["Predicted_Event"].value_counts().sort_index().items()},
        "location_counts_top10": {
            str(k): int(v) for k, v in predictions["Predicted_Location"].value_counts().head(10).items()
        },
    }
    diagnostics["schema"] = {"output_columns": OFFICIAL_COLUMNS}
    (output_csv.parent / "prediction_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return predictions, diagnostics


def run_hybrid_submission_prediction(
    input_dir: Path,
    output_csv: Path | None = None,
    topology_dir: Path | None = None,
    physics_model_dir: Path | None = None,
    ml_model_dir: Path | None = None,
    ml_max_duration_s: float = 120.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    input_dir = Path(input_dir)
    output_csv = Path(output_csv) if output_csv is not None else input_dir / "predictions.csv"
    ml_model_dir = Path(ml_model_dir) if ml_model_dir is not None else Path(__file__).resolve().parents[2] / "models"

    t0 = time.perf_counter()
    frames = _load_frames(input_dir)
    duration_s = _duration_seconds(frames)
    has_ml_bundle = (ml_model_dir / "final_model_config.json").exists()

    if has_ml_bundle and duration_s <= float(ml_max_duration_s):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            model = SGSMAFinalModel(ml_model_dir)
            base_features, dynamic_features = _build_feature_rows(frames)
            pred = model.predict_from_features(base_features, dynamic_features).iloc[0]
            topk = model.predict_location_topk(base_features, dynamic_features, k=3)[0]
        diagnostics = {
            "input_dir": str(input_dir.resolve()),
            "output_csv": str(output_csv.resolve()),
            "buses_detected": sorted(int(bus) for bus in frames),
            "model": "sgms_extra_trees_hybrid_v2",
            "route": "ml_chunk",
            "duration_s": float(duration_s),
            "inference_seconds": float(time.perf_counter() - t0),
            "top3_location": topk,
        }
        return _broadcast_prediction(frames, int(pred["pred_event"]), str(pred["pred_location"]), output_csv, diagnostics)

    predictions, diagnostics = run_bus_agnostic_prediction(
        input_dir=input_dir,
        output_csv=output_csv,
        topology_dir=topology_dir if topology_dir is not None else Path(__file__).resolve().parents[2] / "data" / "topology" / "ieee39",
        model_dir=physics_model_dir,
    )
    diagnostics["route"] = "physics_long_raw"
    diagnostics["duration_s"] = float(duration_s)
    diagnostics["inference_seconds"] = float(time.perf_counter() - t0)
    (output_csv.parent / "prediction_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return predictions, diagnostics
