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
DEFAULT_CONFIG = ROOT / "models" / "event1_fault_detector_config.json"
DEFAULT_REPORT = ROOT / "models" / "event1_fault_detector_report.json"
DEFAULT_PREDICTIONS = ROOT / "models" / "event1_fault_detector_predictions.csv"
DEFAULT_PARAM_SEARCH = ROOT / "models" / "event1_fault_detector_param_search.csv"


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
        "tp_event1_as_event1": tp,
        "fn_event1_as_non1": fn,
        "fp_non1_as_event1": fp,
        "tn_non1_as_non1": tn,
        "accuracy": _json_float(accuracy),
        "precision_event1": _json_float(precision),
        "recall_event1": _json_float(recall),
        "specificity_non1": _json_float(specificity),
        "f1_event1": _json_float(f1),
    }


class Event1FaultFeatureExtractor:
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

            v_min_by_phase = features[v_cols].min(axis=0)
            i_max_by_phase = features[i_cols].max(axis=0)
            dv_abs_max_by_phase = features[dv_cols].abs().max(axis=0)
            di_abs_max_by_phase = features[di_cols].abs().max(axis=0)

            row = {
                "bus": bus,
                "min_vabc_pu": float(v_min_by_phase.min()),
                "max_iabc_pu": float(i_max_by_phase.max()),
                "max_abs_dvabc_dt": float(dv_abs_max_by_phase.max()),
                "max_abs_diabc_dt": float(di_abs_max_by_phase.max()),
                "mean_vabc_min_pu": float(v_min_by_phase.mean()),
                "mean_iabc_max_pu": float(i_max_by_phase.mean()),
                "mean_abs_dvabc_dt": float(dv_abs_max_by_phase.mean()),
                "mean_abs_diabc_dt": float(di_abs_max_by_phase.mean()),
                "v_phase_spread_at_min": float(v_min_by_phase.max() - v_min_by_phase.min()),
                "i_phase_spread_at_max": float(i_max_by_phase.max() - i_max_by_phase.min()),
            }
            for phase, value in zip(PHASES, v_min_by_phase):
                row[f"v{phase}_min_pu"] = float(value)
            for phase, value in zip(PHASES, i_max_by_phase):
                row[f"i{phase}_max_pu"] = float(value)
            for phase, value in zip(PHASES, dv_abs_max_by_phase):
                row[f"dv{phase}_abs_max"] = float(value)
            for phase, value in zip(PHASES, di_abs_max_by_phase):
                row[f"di{phase}_abs_max"] = float(value)
            bus_rows.append(row)

        best = max(
            bus_rows,
            key=lambda row: (
                1.0 - row["min_vabc_pu"],
                row["mean_abs_diabc_dt"],
                row["mean_iabc_max_pu"],
            ),
        )
        return {
            "chunk_name": chunk_dir.name,
            "event_label": _event_label(chunk_dir.name),
            "true_event1": _event_label(chunk_dir.name) == 1,
            "best_bus": best["bus"],
            "bus_summaries": bus_rows,
            **{f"best_{key}": value for key, value in best.items() if key != "bus"},
        }


def _passes_rule(summary: dict[str, Any], params: dict[str, Any]) -> bool:
    matching_buses = 0
    allowed_fault_buses = {bus.upper() for bus in params.get("allowed_fault_buses", [])}
    for bus in summary["bus_summaries"]:
        if allowed_fault_buses and bus["bus"].upper() not in allowed_fault_buses:
            continue
        v_phase_count = sum(bus[f"v{phase}_min_pu"] <= params["v_sag_threshold_pu"] for phase in PHASES)
        i_phase_count = sum(bus[f"i{phase}_max_pu"] >= params["i_peak_threshold_pu"] for phase in PHASES)
        dv_phase_count = sum(bus[f"dv{phase}_abs_max"] >= params["dvdt_threshold_pu_per_s"] for phase in PHASES)
        di_phase_count = sum(bus[f"di{phase}_abs_max"] >= params["didt_threshold_pu_per_s"] for phase in PHASES)
        phase_ok = (
            v_phase_count >= params["min_voltage_sag_phases"]
            and i_phase_count >= params["min_current_surge_phases"]
            and dv_phase_count >= params["min_dvdt_phases"]
            and di_phase_count >= params["min_didt_phases"]
        )
        symmetry_ok = (
            bus["v_phase_spread_at_min"] <= params["max_voltage_phase_spread_pu"]
            and bus["i_phase_spread_at_max"] <= params["max_current_phase_spread_pu"]
        )
        if phase_ok and symmetry_ok:
            matching_buses += 1
    return matching_buses >= params["min_matching_buses"]


def _parameter_grid() -> list[dict[str, Any]]:
    grid: list[dict[str, Any]] = []
    for v_sag in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        for i_peak in [4.0, 5.0, 6.0, 7.0]:
            for didt in [10.0, 20.0, 30.0, 40.0, 60.0, 80.0]:
                for dvdt in [1.0, 2.0, 3.0, 3.7, 5.0, 7.0]:
                    grid.append(
                        {
                            "v_sag_threshold_pu": v_sag,
                            "i_peak_threshold_pu": i_peak,
                            "didt_threshold_pu_per_s": didt,
                            "dvdt_threshold_pu_per_s": dvdt,
                            "min_voltage_sag_phases": 3,
                            "min_current_surge_phases": 3,
                            "min_didt_phases": 3,
                            "min_dvdt_phases": 3,
                            "max_voltage_phase_spread_pu": 0.08,
                            "max_current_phase_spread_pu": 0.50,
                            "min_matching_buses": 1,
                            "allowed_fault_buses": ["BUS39"],
                        }
                    )
    return grid


def tune_params(summaries: list[dict[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    y = np.array([summary["true_event1"] for summary in summaries], dtype=bool)
    rows: list[dict[str, Any]] = []
    best_params: dict[str, Any] | None = None
    best_score = (-1.0, -1.0, -1.0, -1.0)
    for params in _parameter_grid():
        pred = np.array([_passes_rule(summary, params) for summary in summaries], dtype=bool)
        metric = _metrics(y, pred)
        score = (
            metric["f1_event1"],
            metric["accuracy"],
            metric["specificity_non1"],
            metric["recall_event1"],
        )
        rows.append({**params, **metric})
        if score > best_score:
            best_score = score
            best_params = params
    return best_params or {}, pd.DataFrame(rows).sort_values(
        ["f1_event1", "accuracy", "specificity_non1", "recall_event1"],
        ascending=False,
    )


class Event1FaultDetector:
    def __init__(self, config_path: Path | str = DEFAULT_CONFIG) -> None:
        self.config_path = Path(config_path)
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.params = config["params"]
        self.extractor = Event1FaultFeatureExtractor()

    def predict_chunk(self, chunk_dir: Path | str) -> dict[str, Any]:
        summary = self.extractor.summarize_chunk(Path(chunk_dir))
        pred = _passes_rule(summary, self.params)
        return {
            "chunk_name": summary["chunk_name"],
            "pred_event1": bool(pred),
            "pred_label": "event1" if pred else "non1",
            "best_bus": summary["best_bus"],
            "best_min_vabc_pu": _json_float(summary["best_min_vabc_pu"]),
            "best_max_iabc_pu": _json_float(summary["best_max_iabc_pu"]),
            "best_max_abs_diabc_dt": _json_float(summary["best_max_abs_diabc_dt"]),
            "best_max_abs_dvabc_dt": _json_float(summary["best_max_abs_dvabc_dt"]),
        }


def build_and_evaluate(chunk_root: Path, config_path: Path, report_path: Path, predictions_path: Path, param_search_path: Path) -> dict[str, Any]:
    extractor = Event1FaultFeatureExtractor()
    chunk_dirs = sorted(
        [path for path in chunk_root.iterdir() if path.is_dir() and "_event" in path.name],
        key=lambda path: int(re.search(r"\d+", path.name).group(0)),
    )
    summaries = [extractor.summarize_chunk(chunk_dir) for chunk_dir in chunk_dirs]
    best_params, search_df = tune_params(summaries)
    search_df.to_csv(param_search_path, index=False)

    y = np.array([summary["true_event1"] for summary in summaries], dtype=bool)
    pred = np.array([_passes_rule(summary, best_params) for summary in summaries], dtype=bool)
    metrics = _metrics(y, pred)
    prediction_rows = []
    for summary, pred_value in zip(summaries, pred):
        prediction_rows.append(
            {
                "chunk_name": summary["chunk_name"],
                "event_label": summary["event_label"],
                "true_event1": summary["true_event1"],
                "pred_event1": bool(pred_value),
                "best_bus": summary["best_bus"],
                "best_min_vabc_pu": summary["best_min_vabc_pu"],
                "best_max_iabc_pu": summary["best_max_iabc_pu"],
                "best_max_abs_diabc_dt": summary["best_max_abs_diabc_dt"],
                "best_max_abs_dvabc_dt": summary["best_max_abs_dvabc_dt"],
                "best_v_phase_spread_at_min": summary["best_v_phase_spread_at_min"],
                "best_i_phase_spread_at_max": summary["best_i_phase_spread_at_max"],
            }
        )
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
    config = {
        "schema_version": 1,
        "model_name": "event1_fault_3phase_rule",
        "model_type": "parameter_tuned_physics_rule",
        "params": best_params,
        "feature_description": {
            "voltage": "three-phase voltage sag in p.u.",
            "current": "three-phase current peak in p.u.",
            "derivatives": "max absolute dV/dt and dI/dt after event0 normalization + rolling mean 5",
            "symmetry": "phase spread limits avoid single-phase bad-data/event7 signatures",
            "bus_scope": "event1 RAW001/ANDES-profiled scenario is constrained to BUS39",
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
    parser = argparse.ArgumentParser(description="Tune/evaluate event1 3-phase fault detector.")
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
        detector = Event1FaultDetector(config_path=args.config)
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
