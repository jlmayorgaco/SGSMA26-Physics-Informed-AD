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
DEFAULT_CONFIG = ROOT / "models" / "event4_load_change_detector_config.json"
DEFAULT_REPORT = ROOT / "models" / "event4_load_change_detector_report.json"
DEFAULT_PREDICTIONS = ROOT / "models" / "event4_load_change_detector_predictions.csv"
DEFAULT_PARAM_SEARCH = ROOT / "models" / "event4_load_change_detector_param_search.csv"

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
        "tp_event4_as_event4": tp,
        "fn_event4_as_non4": fn,
        "fp_non4_as_event4": fp,
        "tn_non4_as_non4": tn,
        "accuracy": _json_float((tp + tn) / len(y) if len(y) else 0.0),
        "precision_event4": _json_float(precision),
        "recall_event4": _json_float(recall),
        "specificity_non4": _json_float(specificity),
        "f1_event4": _json_float(f1),
    }


class Event4LoadFeatureExtractor:
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
        bus_rows = []
        for bus in BUSES:
            v_cols = [f"V{p}_MAG_PU_FILTERED_{bus}" for p in PHASES]
            i_cols = [f"I{p}_MAG_PU_FILTERED_{bus}" for p in PHASES]
            di_cols = [f"{col}_D1_PER_S" for col in i_cols]
            freq_col = f"Freq_HZ_DEV_FILTERED_{bus}"
            rocof_col = f"ROCOF_HZ_PER_S_FILTERED_{bus}"
            vals_v = x[v_cols].to_numpy(dtype=float)
            vals_i = x[i_cols].to_numpy(dtype=float)
            row = {
                "bus": bus,
                "min_vabc_pu": _nan_stat(vals_v, "min"),
                "max_vabc_pu": _nan_stat(vals_v, "max"),
                "span_vabc_pu": _nan_stat(vals_v, "max") - _nan_stat(vals_v, "min"),
                "max_iabc_pu": _nan_stat(vals_i, "max"),
                "span_iabc_pu": _nan_stat(vals_i, "max") - _nan_stat(vals_i, "min"),
                "max_abs_diabc_dt": _nan_stat(np.abs(x[di_cols].to_numpy(dtype=float)), "max"),
                "freq_span_hz": _nan_stat(x[freq_col].to_numpy(dtype=float), "max") - _nan_stat(x[freq_col].to_numpy(dtype=float), "min"),
                "rocof_abs_max": _nan_stat(np.abs(x[rocof_col].to_numpy(dtype=float)), "max"),
            }
            bus_rows.append(row)
        best = max(bus_rows, key=lambda row: (row["span_iabc_pu"], row["max_abs_diabc_dt"]))
        return {
            "chunk_name": chunk_dir.name,
            "event_label": _event_label(chunk_dir.name),
            "true_event4": _event_label(chunk_dir.name) == 4,
            "duration_seconds_est": float(len(x) / 30.0),
            "n_rows": int(len(x)),
            "best_bus": best["bus"],
            "bus_summaries": bus_rows,
            "max_current_span_pu": float(max(row["span_iabc_pu"] for row in bus_rows)),
            "max_iabc_pu": float(max(row["max_iabc_pu"] for row in bus_rows)),
            "max_abs_diabc_dt": float(max(row["max_abs_diabc_dt"] for row in bus_rows)),
            "max_freq_span_hz": float(max(row["freq_span_hz"] for row in bus_rows)),
            "max_rocof_abs": float(max(row["rocof_abs_max"] for row in bus_rows)),
            "min_vabc_pu": float(min(row["min_vabc_pu"] for row in bus_rows)),
            "max_voltage_span_pu": float(max(row["span_vabc_pu"] for row in bus_rows)),
        }


def _passes_rule(summary: dict[str, Any], params: dict[str, Any]) -> bool:
    return bool(
        summary["duration_seconds_est"] >= params["min_duration_seconds"]
        and summary["max_current_span_pu"] >= params["min_current_span_pu"]
        and summary["max_iabc_pu"] >= params["min_i_peak_pu"]
        and summary["max_abs_diabc_dt"] >= params["min_didt_pu_per_s"]
        and summary["max_abs_diabc_dt"] <= params["max_didt_pu_per_s"]
        and summary["min_vabc_pu"] >= params["min_voltage_floor_pu"]
        and summary["max_voltage_span_pu"] <= params["max_voltage_span_pu"]
        and summary["max_freq_span_hz"] <= params["max_freq_span_hz"]
        and summary["max_rocof_abs"] >= params["min_rocof_abs"]
        and summary["max_rocof_abs"] <= params["max_rocof_abs"]
    )


def _parameter_grid() -> list[dict[str, Any]]:
    grid = []
    for duration in [30.0, 60.0, 120.0, 180.0]:
        for current_span in [0.15, 0.20, 0.30, 0.50]:
            for i_peak in [4.0, 5.0, 6.0]:
                for didt in [0.5, 1.0, 2.0, 4.0, 6.0]:
                    for max_didt in [8.0, 10.0, 15.0, 25.0]:
                        for v_floor in [0.95, 0.97, 0.98]:
                            for v_span in [0.03, 0.05, 0.07, 0.10]:
                                for freq_span in [0.10, 0.13, 0.16, 0.20]:
                                    for min_rocof in [0.5, 1.0, 2.0, 4.0]:
                                        for max_rocof in [5.0, 8.0, 10.0, 15.0]:
                                            if min_rocof > max_rocof:
                                                continue
                                            grid.append(
                                                {
                                                    "min_duration_seconds": duration,
                                                    "min_current_span_pu": current_span,
                                                    "min_i_peak_pu": i_peak,
                                                    "min_didt_pu_per_s": didt,
                                                    "max_didt_pu_per_s": max_didt,
                                                    "min_voltage_floor_pu": v_floor,
                                                    "max_voltage_span_pu": v_span,
                                                    "max_freq_span_hz": freq_span,
                                                    "min_rocof_abs": min_rocof,
                                                    "max_rocof_abs": max_rocof,
                                                }
                                            )
    return grid


def tune_params(summaries: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    y = np.array([s["true_event4"] for s in summaries], dtype=bool)
    rows = []
    best_params: dict[str, Any] | None = None
    best_score = (-1.0, -1.0, -1.0, -1.0)
    for params in _parameter_grid():
        pred = np.array([_passes_rule(s, params) for s in summaries], dtype=bool)
        metric = _metrics(y, pred)
        score = (metric["f1_event4"], metric["accuracy"], metric["specificity_non4"], metric["recall_event4"])
        rows.append({**params, **metric})
        if score > best_score:
            best_score = score
            best_params = params
    return best_params or {}, pd.DataFrame(rows).sort_values(["f1_event4", "accuracy", "specificity_non4", "recall_event4"], ascending=False)


class Event4LoadChangeDetector:
    def __init__(self, config_path: Path | str = DEFAULT_CONFIG) -> None:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.params = config["params"]
        self.extractor = Event4LoadFeatureExtractor()

    def predict_chunk(self, chunk_dir: Path | str) -> dict[str, Any]:
        summary = self.extractor.summarize_chunk(Path(chunk_dir))
        pred = _passes_rule(summary, self.params)
        return {
            "chunk_name": summary["chunk_name"],
            "pred_event4": bool(pred),
            "pred_label": "event4" if pred else "non4",
            "best_bus": summary["best_bus"],
            "max_current_span_pu": _json_float(summary["max_current_span_pu"]),
            "max_freq_span_hz": _json_float(summary["max_freq_span_hz"]),
            "max_rocof_abs": _json_float(summary["max_rocof_abs"]),
            "min_vabc_pu": _json_float(summary["min_vabc_pu"]),
            "duration_seconds_est": _json_float(summary["duration_seconds_est"]),
        }


def build_and_evaluate(chunk_root: Path, config_path: Path, report_path: Path, predictions_path: Path, param_search_path: Path) -> dict[str, Any]:
    extractor = Event4LoadFeatureExtractor()
    chunk_dirs = sorted([p for p in chunk_root.iterdir() if p.is_dir() and "_event" in p.name], key=lambda p: int(re.search(r"\d+", p.name).group(0)))
    summaries = [extractor.summarize_chunk(p) for p in chunk_dirs]
    params, search_df = tune_params(summaries)
    search_df.to_csv(param_search_path, index=False)
    y = np.array([s["true_event4"] for s in summaries], dtype=bool)
    pred = np.array([_passes_rule(s, params) for s in summaries], dtype=bool)
    metrics = _metrics(y, pred)
    rows = []
    for s, p in zip(summaries, pred):
        rows.append({k: v for k, v in s.items() if k != "bus_summaries"} | {"pred_event4": bool(p)})
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    config = {
        "schema_version": 1,
        "model_name": "event4_load_change_rule",
        "model_type": "parameter_tuned_physics_rule",
        "params": params,
        "feature_description": {
            "current": "moderate sustained current step over a long window",
            "voltage_guard": "voltage remains close to normal; no deep sag",
            "frequency_guard": "frequency and ROCOF are moderate, unlike generation-change event3",
        },
        "training_metrics_raw0001": metrics,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"metrics": metrics, "params": params, "predictions": rows, "config_path": str(config_path.resolve())}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune/evaluate event4 load-change detector.")
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
        detector = Event4LoadChangeDetector(config_path=args.config)
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
