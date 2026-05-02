from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from enevt0_d3tector import DEFAULT_CHUNK_ROOT, Event0RangeDetector, _bus_csv_paths


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "models" / "event3_generation_change_detector_config.json"
DEFAULT_REPORT = ROOT / "models" / "event3_generation_change_detector_report.json"
DEFAULT_PREDICTIONS = ROOT / "models" / "event3_generation_change_detector_predictions.csv"
DEFAULT_PARAM_SEARCH = ROOT / "models" / "event3_generation_change_detector_param_search.csv"

BUSES = ("BUS2", "BUS5", "BUS6", "BUS10", "BUS19", "BUS22", "BUS29", "BUS39")
PHASES = ("A", "B", "C")


def _event_label(chunk_name: str) -> int | None:
    match = re.search(r"_event(\d+)$", chunk_name)
    return int(match.group(1)) if match else None


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _nan_stat(values: np.ndarray, op: str, default: float = 0.0) -> float:
    clean = np.asarray(values, dtype=float)
    if clean.size == 0 or np.isnan(clean).all():
        return float(default)
    if op == "min":
        return float(np.nanmin(clean))
    if op == "max":
        return float(np.nanmax(clean))
    if op == "median":
        return float(np.nanmedian(clean))
    raise ValueError(op)


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(y & pred))
    fn = int(np.sum(y & ~pred))
    fp = int(np.sum(~y & pred))
    tn = int(np.sum(~y & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp_event3_as_event3": tp,
        "fn_event3_as_non3": fn,
        "fp_non3_as_event3": fp,
        "tn_non3_as_non3": tn,
        "accuracy": _json_float((tp + tn) / len(y) if len(y) else 0.0),
        "precision_event3": _json_float(precision),
        "recall_event3": _json_float(recall),
        "specificity_non3": _json_float(specificity),
        "f1_event3": _json_float(f1),
    }


class Event3GenerationFeatureExtractor:
    def __init__(self) -> None:
        self.event0_detector = Event0RangeDetector()

    def _load_features(self, chunk_dir: Path) -> pd.DataFrame:
        bus_frames = {}
        for csv_path in _bus_csv_paths(chunk_dir):
            bus = csv_path.name.split("_", 1)[0].upper()
            bus_frames[bus] = pd.read_csv(csv_path)
        if not bus_frames:
            raise FileNotFoundError(f"No Bus*_Competition_Data*.csv files found in {chunk_dir}")
        return self.event0_detector.build_features(bus_frames, include_derivatives=True)

    def summarize_chunk(self, chunk_dir: Path) -> dict[str, Any]:
        x = self._load_features(chunk_dir)
        bus_rows: list[dict[str, Any]] = []
        for bus in BUSES:
            v_cols = [f"V{p}_MAG_PU_FILTERED_{bus}" for p in PHASES]
            i_cols = [f"I{p}_MAG_PU_FILTERED_{bus}" for p in PHASES]
            di_cols = [f"{col}_D1_PER_S" for col in i_cols]
            freq_col = f"Freq_HZ_DEV_FILTERED_{bus}"
            rocof_col = f"ROCOF_HZ_PER_S_FILTERED_{bus}"
            row = {
                "bus": bus,
                "min_vabc_pu": _nan_stat(x[v_cols].to_numpy(dtype=float), "min"),
                "max_vabc_pu": _nan_stat(x[v_cols].to_numpy(dtype=float), "max"),
                "max_iabc_pu": _nan_stat(x[i_cols].to_numpy(dtype=float), "max"),
                "span_iabc_pu": _nan_stat(x[i_cols].to_numpy(dtype=float), "max") - _nan_stat(x[i_cols].to_numpy(dtype=float), "min"),
                "max_abs_diabc_dt": _nan_stat(np.abs(x[di_cols].to_numpy(dtype=float)), "max"),
                "freq_span_hz": _nan_stat(x[freq_col].to_numpy(dtype=float), "max") - _nan_stat(x[freq_col].to_numpy(dtype=float), "min"),
                "freq_abs_step_hz": abs(
                    _nan_stat(x[freq_col].tail(max(1, len(x) // 5)).to_numpy(dtype=float), "median")
                    - _nan_stat(x[freq_col].head(max(1, len(x) // 5)).to_numpy(dtype=float), "median")
                ),
                "rocof_abs_max": _nan_stat(np.abs(x[rocof_col].to_numpy(dtype=float)), "max"),
            }
            bus_rows.append(row)

        best = max(
            bus_rows,
            key=lambda row: (row["freq_span_hz"], row["rocof_abs_max"], row["max_abs_diabc_dt"]),
        )
        return {
            "chunk_name": chunk_dir.name,
            "event_label": _event_label(chunk_dir.name),
            "true_event3": _event_label(chunk_dir.name) == 3,
            "duration_seconds_est": float(len(x) / 30.0),
            "n_rows": int(len(x)),
            "best_bus": best["bus"],
            "bus_summaries": bus_rows,
            "max_freq_span_hz": float(max(row["freq_span_hz"] for row in bus_rows)),
            "max_freq_abs_step_hz": float(max(row["freq_abs_step_hz"] for row in bus_rows)),
            "max_rocof_abs": float(max(row["rocof_abs_max"] for row in bus_rows)),
            "max_abs_diabc_dt": float(max(row["max_abs_diabc_dt"] for row in bus_rows)),
            "max_iabc_pu": float(max(row["max_iabc_pu"] for row in bus_rows)),
            "min_vabc_pu": float(min(row["min_vabc_pu"] for row in bus_rows)),
            "bus2_freq_span_hz": next(row["freq_span_hz"] for row in bus_rows if row["bus"] == "BUS2"),
            "bus2_rocof_abs": next(row["rocof_abs_max"] for row in bus_rows if row["bus"] == "BUS2"),
            "bus2_max_abs_diabc_dt": next(row["max_abs_diabc_dt"] for row in bus_rows if row["bus"] == "BUS2"),
        }


def _passes_rule(summary: dict[str, Any], params: dict[str, Any]) -> bool:
    duration_ok = summary["duration_seconds_est"] >= params["min_duration_seconds"]
    freq_ok = summary["max_freq_span_hz"] >= params["min_freq_span_hz"]
    rocof_ok = summary["max_rocof_abs"] >= params["min_rocof_abs"]
    bus2_ok = (
        summary["bus2_freq_span_hz"] >= params["min_bus2_freq_span_hz"]
        or summary["bus2_rocof_abs"] >= params["min_bus2_rocof_abs"]
        or summary["bus2_max_abs_diabc_dt"] >= params["min_bus2_didt"]
    )
    current_ok = summary["max_abs_diabc_dt"] >= params["min_global_didt"]
    return bool(duration_ok and freq_ok and rocof_ok and bus2_ok and current_ok)


def _parameter_grid() -> list[dict[str, Any]]:
    grid = []
    for min_duration in [5.0, 10.0, 15.0]:
        for freq_span in [0.10, 0.12, 0.13, 0.15, 0.20, 0.30]:
            for rocof in [5.0, 7.0, 10.0, 15.0, 20.0]:
                for bus2_freq in [0.05, 0.10, 0.15, 0.20]:
                    for didt in [2.0, 5.0, 10.0]:
                        grid.append(
                            {
                                "min_duration_seconds": min_duration,
                                "min_freq_span_hz": freq_span,
                                "min_rocof_abs": rocof,
                                "min_bus2_freq_span_hz": bus2_freq,
                                "min_bus2_rocof_abs": rocof,
                                "min_bus2_didt": 10.0,
                                "min_global_didt": didt,
                            }
                        )
    return grid


def tune_params(summaries: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    y = np.array([s["true_event3"] for s in summaries], dtype=bool)
    rows = []
    best_params: dict[str, Any] | None = None
    best_score = (-1.0, -1.0, -1.0, -1.0)
    for params in _parameter_grid():
        pred = np.array([_passes_rule(s, params) for s in summaries], dtype=bool)
        metric = _metrics(y, pred)
        score = (metric["f1_event3"], metric["accuracy"], metric["specificity_non3"], metric["recall_event3"])
        rows.append({**params, **metric})
        if score > best_score:
            best_score = score
            best_params = params
    return best_params or {}, pd.DataFrame(rows).sort_values(["f1_event3", "accuracy", "specificity_non3", "recall_event3"], ascending=False)


class Event3GenerationChangeDetector:
    def __init__(self, config_path: Path | str = DEFAULT_CONFIG) -> None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.params = config["params"]
        self.extractor = Event3GenerationFeatureExtractor()

    def predict_chunk(self, chunk_dir: Path | str) -> dict[str, Any]:
        summary = self.extractor.summarize_chunk(Path(chunk_dir))
        pred = _passes_rule(summary, self.params)
        return {
            "chunk_name": summary["chunk_name"],
            "pred_event3": bool(pred),
            "pred_label": "event3" if pred else "non3",
            "best_bus": summary["best_bus"],
            "max_freq_span_hz": _json_float(summary["max_freq_span_hz"]),
            "max_rocof_abs": _json_float(summary["max_rocof_abs"]),
            "bus2_freq_span_hz": _json_float(summary["bus2_freq_span_hz"]),
            "duration_seconds_est": _json_float(summary["duration_seconds_est"]),
        }


def build_and_evaluate(chunk_root: Path, config_path: Path, report_path: Path, predictions_path: Path, param_search_path: Path) -> dict[str, Any]:
    extractor = Event3GenerationFeatureExtractor()
    chunk_dirs = sorted([p for p in chunk_root.iterdir() if p.is_dir() and "_event" in p.name], key=lambda p: int(re.search(r"\d+", p.name).group(0)))
    summaries = [extractor.summarize_chunk(p) for p in chunk_dirs]
    params, search_df = tune_params(summaries)
    search_df.to_csv(param_search_path, index=False)
    y = np.array([s["true_event3"] for s in summaries], dtype=bool)
    pred = np.array([_passes_rule(s, params) for s in summaries], dtype=bool)
    metrics = _metrics(y, pred)
    rows = []
    for s, p in zip(summaries, pred):
        rows.append({k: v for k, v in s.items() if k != "bus_summaries"} | {"pred_event3": bool(p)})
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    config = {
        "schema_version": 1,
        "model_name": "event3_generation_change_rule",
        "model_type": "parameter_tuned_physics_rule",
        "params": params,
        "feature_description": {
            "frequency": "sustained/global frequency span and final-initial shift",
            "rocof": "large ROCOF peak during generation imbalance",
            "bus2": "BUS2-specific frequency/ROCOF/current derivative evidence",
            "duration": "generation change persists longer than transient fault/outage windows",
        },
        "training_metrics_raw0001": metrics,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"metrics": metrics, "params": params, "predictions": rows, "config_path": str(config_path.resolve())}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune/evaluate event3 generation-change detector.")
    parser.add_argument("--chunk-root", type=Path, default=DEFAULT_CHUNK_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--param-search", type=Path, default=DEFAULT_PARAM_SEARCH)
    parser.add_argument("--chunk-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_dir is not None:
        detector = Event3GenerationChangeDetector(config_path=args.config)
        print(json.dumps(detector.predict_chunk(args.chunk_dir), indent=2))
        return
    report = build_and_evaluate(args.chunk_root, args.config, args.report, args.predictions, args.param_search)
    print(f"Wrote {args.config.resolve()}")
    print(f"Wrote {args.report.resolve()}")
    print(f"Wrote {args.predictions.resolve()}")
    print(f"Wrote {args.param_search.resolve()}")
    print(json.dumps(report["metrics"], indent=2))
    print(json.dumps(report["params"], indent=2))


if __name__ == "__main__":
    main()
