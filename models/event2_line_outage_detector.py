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
DEFAULT_CONFIG = ROOT / "models" / "event2_line_outage_detector_config.json"
DEFAULT_REPORT = ROOT / "models" / "event2_line_outage_detector_report.json"
DEFAULT_PREDICTIONS = ROOT / "models" / "event2_line_outage_detector_predictions.csv"
DEFAULT_PARAM_SEARCH = ROOT / "models" / "event2_line_outage_detector_param_search.csv"

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


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(y & pred))
    fn = int(np.sum(y & ~pred))
    fp = int(np.sum(~y & pred))
    tn = int(np.sum(~y & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    return {
        "tp_event2_as_event2": tp,
        "fn_event2_as_non2": fn,
        "fp_non2_as_event2": fp,
        "tn_non2_as_non2": tn,
        "accuracy": _json_float(accuracy),
        "precision_event2": _json_float(precision),
        "recall_event2": _json_float(recall),
        "specificity_non2": _json_float(specificity),
        "f1_event2": _json_float(f1),
    }


class Event2LineOutageFeatureExtractor:
    def __init__(self) -> None:
        self.event0_detector = Event0RangeDetector()

    def build_detector_features(self, bus_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        filtered = self.event0_detector.build_features(bus_frames, include_derivatives=True)
        original_window = self.event0_detector.rolling_window
        try:
            self.event0_detector.rolling_window = 1
            unfiltered = self.event0_detector.build_features(bus_frames, include_derivatives=False)
        finally:
            self.event0_detector.rolling_window = original_window
        for column in unfiltered.columns:
            if "_MAG_PU_FILTERED_" in column:
                filtered[column] = unfiltered[column]
        return filtered

    def _load_chunk_features(self, chunk_dir: Path) -> pd.DataFrame:
        bus_frames = {}
        for csv_path in _bus_csv_paths(chunk_dir):
            bus = csv_path.name.split("_", 1)[0].upper()
            bus_frames[bus] = pd.read_csv(csv_path)
        if not bus_frames:
            raise FileNotFoundError(f"No Bus*_Competition_Data*.csv files found in {chunk_dir}")
        return self.build_detector_features(bus_frames)

    def summarize_chunk(self, chunk_dir: Path) -> dict[str, Any]:
        features = self._load_chunk_features(chunk_dir)
        bus_rows: list[dict[str, Any]] = []
        for bus in BUSES:
            v_cols = [f"V{phase}_MAG_PU_FILTERED_{bus}" for phase in PHASES]
            i_cols = [f"I{phase}_MAG_PU_FILTERED_{bus}" for phase in PHASES]
            dv_cols = [f"{col}_D1_PER_S" for col in v_cols]
            di_cols = [f"{col}_D1_PER_S" for col in i_cols]

            v_min = features[v_cols].min(axis=0)
            v_max = features[v_cols].max(axis=0)
            i_min = features[i_cols].min(axis=0)
            i_max = features[i_cols].max(axis=0)
            dv_abs_max = features[dv_cols].abs().max(axis=0)
            di_abs_max = features[di_cols].abs().max(axis=0)

            row = {
                "bus": bus,
                "min_vabc_pu": float(v_min.min()),
                "max_vabc_pu": float(v_max.max()),
                "span_vabc_pu": float(v_max.max() - v_min.min()),
                "max_iabc_pu": float(i_max.max()),
                "min_iabc_pu": float(i_min.min()),
                "span_iabc_pu": float(i_max.max() - i_min.min()),
                "max_abs_dvabc_dt": float(dv_abs_max.max()),
                "max_abs_diabc_dt": float(di_abs_max.max()),
                "mean_abs_diabc_dt": float(di_abs_max.mean()),
                "i_phase_spread_at_max": float(i_max.max() - i_max.min()),
                "v_phase_spread_at_min": float(v_min.max() - v_min.min()),
            }
            for phase, value in zip(PHASES, v_min):
                row[f"v{phase}_min_pu"] = float(value)
            for phase, value in zip(PHASES, i_max):
                row[f"i{phase}_max_pu"] = float(value)
            for phase, value in zip(PHASES, di_abs_max):
                row[f"di{phase}_abs_max"] = float(value)
            bus_rows.append(row)

        best = max(
            bus_rows,
            key=lambda row: (
                row["max_iabc_pu"],
                row["max_abs_diabc_dt"],
                -abs(row["min_vabc_pu"] - 1.0),
            ),
        )
        return {
            "chunk_name": chunk_dir.name,
            "event_label": _event_label(chunk_dir.name),
            "true_event2": _event_label(chunk_dir.name) == 2,
            "best_bus": best["bus"],
            "bus_summaries": bus_rows,
            **{f"best_{key}": value for key, value in best.items() if key != "bus"},
        }


def _passes_rule(summary: dict[str, Any], params: dict[str, Any]) -> bool:
    matching_buses = 0
    for bus in summary["bus_summaries"]:
        high_current_phases = sum(bus[f"i{phase}_max_pu"] >= params["i_peak_threshold_pu"] for phase in PHASES)
        high_didt_phases = sum(bus[f"di{phase}_abs_max"] >= params["didt_threshold_pu_per_s"] for phase in PHASES)
        voltage_not_fault_like = bus["min_vabc_pu"] >= params["min_voltage_floor_pu"]
        voltage_disturbance_small = bus["span_vabc_pu"] <= params["max_voltage_span_pu"]
        current_step_ok = bus["span_iabc_pu"] >= params["min_current_span_pu"]
        current_symmetry_ok = bus["i_phase_spread_at_max"] <= params["max_current_phase_spread_pu"]
        if (
            high_current_phases >= params["min_current_surge_phases"]
            and high_didt_phases >= params["min_didt_phases"]
            and voltage_not_fault_like
            and voltage_disturbance_small
            and current_step_ok
            and current_symmetry_ok
        ):
            matching_buses += 1
    return matching_buses >= params["min_matching_buses"]


def _parameter_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for i_peak in [6.0, 7.0, 8.0, 9.0, 10.0]:
        for didt in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
            for v_floor in [0.85, 0.90, 0.95, 0.98]:
                for v_span in [0.05, 0.08, 0.10, 0.15]:
                    for i_span in [1.0, 2.0, 3.0, 4.0]:
                        grid.append(
                            {
                                "i_peak_threshold_pu": i_peak,
                                "didt_threshold_pu_per_s": didt,
                                "min_voltage_floor_pu": v_floor,
                                "max_voltage_span_pu": v_span,
                                "min_current_span_pu": i_span,
                                "min_current_surge_phases": 3,
                                "min_didt_phases": 3,
                                "max_current_phase_spread_pu": 0.75,
                                "min_matching_buses": 1,
                            }
                        )
    return grid


def tune_params(summaries: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    y = np.array([summary["true_event2"] for summary in summaries], dtype=bool)
    rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_score = (-1.0, -1.0, -1.0, -1.0)
    for params in _parameter_grid():
        pred = np.array([_passes_rule(summary, params) for summary in summaries], dtype=bool)
        metric = _metrics(y, pred)
        score = (
            metric["f1_event2"],
            metric["accuracy"],
            metric["specificity_non2"],
            metric["recall_event2"],
        )
        rows.append({**params, **metric})
        if score > best_score:
            best_score = score
            best_params = params
    return best_params or {}, pd.DataFrame(rows).sort_values(
        ["f1_event2", "accuracy", "specificity_non2", "recall_event2"],
        ascending=False,
    )


class Event2LineOutageDetector:
    def __init__(self, config_path: Path | str = DEFAULT_CONFIG) -> None:
        self.config_path = Path(config_path)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.params = config["params"]
        self.extractor = Event2LineOutageFeatureExtractor()

    def predict_chunk(self, chunk_dir: Path | str) -> dict[str, Any]:
        summary = self.extractor.summarize_chunk(Path(chunk_dir))
        pred = _passes_rule(summary, self.params)
        return {
            "chunk_name": summary["chunk_name"],
            "pred_event2": bool(pred),
            "pred_label": "event2" if pred else "non2",
            "best_bus": summary["best_bus"],
            "best_min_vabc_pu": _json_float(summary["best_min_vabc_pu"]),
            "best_span_vabc_pu": _json_float(summary["best_span_vabc_pu"]),
            "best_max_iabc_pu": _json_float(summary["best_max_iabc_pu"]),
            "best_span_iabc_pu": _json_float(summary["best_span_iabc_pu"]),
            "best_max_abs_diabc_dt": _json_float(summary["best_max_abs_diabc_dt"]),
        }


def build_and_evaluate(chunk_root: Path, config_path: Path, report_path: Path, predictions_path: Path, param_search_path: Path) -> dict[str, Any]:
    extractor = Event2LineOutageFeatureExtractor()
    chunk_dirs = sorted(
        [path for path in chunk_root.iterdir() if path.is_dir() and "_event" in path.name],
        key=lambda path: int(re.search(r"\d+", path.name).group(0)),
    )
    summaries = [extractor.summarize_chunk(chunk_dir) for chunk_dir in chunk_dirs]
    best_params, search_df = tune_params(summaries)
    search_df.to_csv(param_search_path, index=False)
    y = np.array([summary["true_event2"] for summary in summaries], dtype=bool)
    pred = np.array([_passes_rule(summary, best_params) for summary in summaries], dtype=bool)
    metrics = _metrics(y, pred)

    prediction_rows = []
    for summary, pred_value in zip(summaries, pred):
        prediction_rows.append(
            {
                "chunk_name": summary["chunk_name"],
                "event_label": summary["event_label"],
                "true_event2": summary["true_event2"],
                "pred_event2": bool(pred_value),
                "best_bus": summary["best_bus"],
                "best_min_vabc_pu": summary["best_min_vabc_pu"],
                "best_span_vabc_pu": summary["best_span_vabc_pu"],
                "best_max_iabc_pu": summary["best_max_iabc_pu"],
                "best_span_iabc_pu": summary["best_span_iabc_pu"],
                "best_max_abs_diabc_dt": summary["best_max_abs_diabc_dt"],
                "best_i_phase_spread_at_max": summary["best_i_phase_spread_at_max"],
            }
        )
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    config = {
        "schema_version": 1,
        "model_name": "event2_line_outage_rule",
        "model_type": "parameter_tuned_physics_rule",
        "params": best_params,
        "feature_description": {
            "current": "three-phase current peak and current step in p.u.",
            "derivative": "three-phase current derivative after event0 normalization + rolling mean 5",
            "voltage_guard": "line outage has no deep three-phase voltage sag; voltage span remains small",
        },
        "training_metrics_raw0001": metrics,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {
        "config_path": str(config_path.resolve()),
        "predictions_path": str(predictions_path.resolve()),
        "param_search_path": str(param_search_path.resolve()),
        "metrics": metrics,
        "params": best_params,
        "predictions": prediction_rows,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune/evaluate event2 line outage detector.")
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
        detector = Event2LineOutageDetector(config_path=args.config)
        print(json.dumps(detector.predict_chunk(args.chunk_dir), indent=2))
        return
    report = build_and_evaluate(
        chunk_root=args.chunk_root,
        config_path=args.config,
        report_path=args.report,
        predictions_path=args.predictions,
        param_search_path=args.param_search,
    )
    print(f"Wrote {args.config.resolve()}")
    print(f"Wrote {args.report.resolve()}")
    print(f"Wrote {args.predictions.resolve()}")
    print(f"Wrote {args.param_search.resolve()}")
    print(json.dumps(report["metrics"], indent=2))
    print(json.dumps(report["params"], indent=2))


if __name__ == "__main__":
    main()
