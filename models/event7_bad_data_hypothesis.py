from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from enevt0_d3tector import DEFAULT_CHUNK_ROOT, Event0RangeDetector
from event5_detector import Event5NanMaskDetector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "models" / "event7_bad_data_hypothesis_report.json"
DEFAULT_FEATURE_CSV = ROOT / "models" / "event7_bad_data_feature_summary.csv"
DEFAULT_VARIANT_CSV = ROOT / "models" / "event7_bad_data_variant_metrics.csv"
DEFAULT_PREDICTIONS_CSV = ROOT / "models" / "event7_bad_data_variant_predictions.csv"


def _event_label(chunk_name: str) -> int | None:
    match = re.search(r"_event(\d+)$", chunk_name)
    return int(match.group(1)) if match else None


def _feature_parts(feature: str) -> tuple[str, str]:
    if "_FILTERED_" not in feature:
        return "UNKNOWN", feature
    left, bus = feature.rsplit("_FILTERED_", 1)
    if left.endswith("_PU"):
        signal = left[:-3]
    elif left.endswith("_RAD_WRAPPED"):
        signal = left[: -len("_RAD_WRAPPED")]
    elif left.endswith("_HZ_DEV"):
        signal = "Freq"
    elif left.endswith("_HZ_PER_S"):
        signal = "ROCOF"
    else:
        signal = left
    return bus.upper(), signal


def _phase_family(signal: str) -> tuple[str, str, str]:
    if "_" not in signal:
        return signal, signal, signal
    phase, measurement = signal.split("_", 1)
    family = "voltage" if phase.startswith("V") else "current" if phase.startswith("I") else phase
    return phase, measurement, family


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


class Event7BadDataHypothesisTester:
    def __init__(self) -> None:
        self.event0_detector = Event0RangeDetector()
        self.event5_detector = Event5NanMaskDetector()

    def feature_outside_summary(self, chunk_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        bus_frames = {}
        for csv_path in sorted(chunk_dir.glob("Bus*_Competition_Data_nanmask.csv")):
            bus = csv_path.name.split("_", 1)[0].upper()
            bus_frames[bus] = pd.read_csv(csv_path)
        features = self.event0_detector.build_features(bus_frames)

        feature_rows: list[dict[str, Any]] = []
        sample_any = np.zeros(len(features), dtype=bool)
        bus_any: dict[str, np.ndarray] = {}
        signal_any: dict[str, np.ndarray] = {}
        phase_channel_any: dict[str, np.ndarray] = {}

        for feature, limit in self.event0_detector.limits.items():
            values = pd.to_numeric(features[feature], errors="coerce")
            lower = float(limit["lower"])
            upper = float(limit["upper"])
            outside = ((values < lower) | (values > upper) | values.isna()).to_numpy(dtype=bool)
            sample_any |= outside
            bus, signal = _feature_parts(feature)
            phase, measurement, family = _phase_family(signal)
            phase_channel = f"{bus}:{family}:{phase}:{measurement}"
            bus_any[bus] = outside if bus not in bus_any else (bus_any[bus] | outside)
            signal_any[signal] = outside if signal not in signal_any else (signal_any[signal] | outside)
            phase_channel_any[phase_channel] = outside if phase_channel not in phase_channel_any else (phase_channel_any[phase_channel] | outside)
            feature_rows.append(
                {
                    "chunk_name": chunk_dir.name,
                    "event_label": _event_label(chunk_dir.name),
                    "feature": feature,
                    "bus": bus,
                    "signal": signal,
                    "phase": phase,
                    "measurement": measurement,
                    "family": family,
                    "phase_channel": phase_channel,
                    "outside_rate": float(np.mean(outside)),
                    "outside_count": int(np.sum(outside)),
                    "n_samples": int(len(outside)),
                }
            )

        top = sorted(feature_rows, key=lambda row: row["outside_rate"], reverse=True)
        nonzero = [row for row in top if row["outside_rate"] > 0]
        top_rate = nonzero[0]["outside_rate"] if nonzero else 0.0
        second_rate = nonzero[1]["outside_rate"] if len(nonzero) > 1 else 0.0
        top_bus = nonzero[0]["bus"] if nonzero else None
        top_signal = nonzero[0]["signal"] if nonzero else None

        bus_rates = {bus: float(np.mean(mask)) for bus, mask in bus_any.items()}
        signal_rates = {signal: float(np.mean(mask)) for signal, mask in signal_any.items()}
        phase_channel_rates = {channel: float(np.mean(mask)) for channel, mask in phase_channel_any.items()}
        active_bus_count_001 = int(sum(rate >= 0.001 for rate in bus_rates.values()))
        active_bus_count_01 = int(sum(rate >= 0.01 for rate in bus_rates.values()))
        active_feature_count_001 = int(sum(row["outside_rate"] >= 0.001 for row in feature_rows))
        active_feature_count_01 = int(sum(row["outside_rate"] >= 0.01 for row in feature_rows))
        active_signal_count_001 = int(sum(rate >= 0.001 for rate in signal_rates.values()))
        active_signal_count_01 = int(sum(rate >= 0.01 for rate in signal_rates.values()))
        active_phase_channel_count_001 = int(sum(rate >= 0.001 for rate in phase_channel_rates.values()))
        active_phase_channel_count_01 = int(sum(rate >= 0.01 for rate in phase_channel_rates.values()))
        top_phase_channel, top_phase_channel_rate = (None, 0.0)
        nonzero_phase_channels = sorted(phase_channel_rates.items(), key=lambda item: item[1], reverse=True)
        if nonzero_phase_channels and nonzero_phase_channels[0][1] > 0:
            top_phase_channel, top_phase_channel_rate = nonzero_phase_channels[0]

        event5_result = self.event5_detector.predict_chunk(chunk_dir)
        summary = {
            "chunk_name": chunk_dir.name,
            "event_label": _event_label(chunk_dir.name),
            "true_event7": _event_label(chunk_dir.name) == 7,
            "event5_nanmask_pred": bool(event5_result["pred_event5"]),
            "event5_nanmask_buses": "|".join(event5_result["event5_buses"]),
            "outside_rate_any_feature": float(np.mean(sample_any)),
            "top_feature": nonzero[0]["feature"] if nonzero else None,
            "top_bus": top_bus,
            "top_signal": top_signal,
            "top_outside_rate": _json_float(top_rate),
            "second_outside_rate": _json_float(second_rate),
            "top_to_second_ratio": _json_float(top_rate / second_rate) if second_rate > 0 else None,
            "active_bus_count_001": active_bus_count_001,
            "active_bus_count_01": active_bus_count_01,
            "active_signal_count_001": active_signal_count_001,
            "active_signal_count_01": active_signal_count_01,
            "active_feature_count_001": active_feature_count_001,
            "active_feature_count_01": active_feature_count_01,
            "active_phase_channel_count_001": active_phase_channel_count_001,
            "active_phase_channel_count_01": active_phase_channel_count_01,
            "top_phase_channel": top_phase_channel,
            "top_phase_channel_rate": _json_float(top_phase_channel_rate),
            "bus_rates": bus_rates,
            "signal_rates": signal_rates,
            "phase_channel_rates": phase_channel_rates,
        }
        return summary, feature_rows


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
        "tp_event7_as_event7": tp,
        "fn_event7_as_non7": fn,
        "fp_non7_as_event7": fp,
        "tn_non7_as_non7": tn,
        "accuracy": _json_float(accuracy),
        "precision_event7": _json_float(precision),
        "recall_event7": _json_float(recall),
        "specificity_non7": _json_float(specificity),
        "f1_event7": _json_float(f1),
    }


def _apply_variant(df: pd.DataFrame, variant: dict[str, Any]) -> np.ndarray:
    pred = df["outside_rate_any_feature"].to_numpy(dtype=float) >= variant["min_any_outside_rate"]
    pred &= df["top_outside_rate"].fillna(0).to_numpy(dtype=float) >= variant["min_top_rate"]
    pred &= df["active_bus_count_01"].to_numpy(dtype=int) <= variant["max_active_buses_01"]
    pred &= df["active_feature_count_01"].to_numpy(dtype=int) <= variant["max_active_features_01"]
    pred &= df["active_signal_count_01"].to_numpy(dtype=int) <= variant["max_active_signals_01"]
    if "max_active_phase_channels_01" in variant:
        pred &= df["active_phase_channel_count_01"].to_numpy(dtype=int) <= variant["max_active_phase_channels_01"]
    if variant.get("exclude_event5_nanmask", False):
        pred &= ~df["event5_nanmask_pred"].to_numpy(dtype=bool)
    if variant.get("require_single_top_bus", False):
        pred &= df["active_bus_count_001"].to_numpy(dtype=int) <= 1
    return pred


def evaluate_variants(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants: list[dict[str, Any]] = []
    for min_top_rate in [0.001, 0.01, 0.05, 0.10, 0.25, 0.50]:
        for max_buses in [1, 2, 3, 8]:
            for max_features in [1, 2, 4, 8, 16, 112]:
                for exclude_event5 in [False, True]:
                    variants.append(
                        {
                            "variant": f"top{min_top_rate}_bus{max_buses}_feat{max_features}_ex5{int(exclude_event5)}",
                            "min_any_outside_rate": 0.0,
                            "min_top_rate": min_top_rate,
                            "max_active_buses_01": max_buses,
                            "max_active_features_01": max_features,
                            "max_active_signals_01": 14,
                            "max_active_phase_channels_01": 24,
                            "exclude_event5_nanmask": exclude_event5,
                            "require_single_top_bus": False,
                        }
                    )
    variants.extend(
        [
            {
                "variant": "strict_one_bus_one_feature_exclude_event5",
                "min_any_outside_rate": 0.001,
                "min_top_rate": 0.001,
                "max_active_buses_01": 1,
                "max_active_features_01": 1,
                "max_active_signals_01": 1,
                "max_active_phase_channels_01": 1,
                "exclude_event5_nanmask": True,
                "require_single_top_bus": True,
            },
            {
                "variant": "one_bus_few_features_exclude_event5",
                "min_any_outside_rate": 0.001,
                "min_top_rate": 0.001,
                "max_active_buses_01": 1,
                "max_active_features_01": 4,
                "max_active_signals_01": 4,
                "max_active_phase_channels_01": 4,
                "exclude_event5_nanmask": True,
                "require_single_top_bus": True,
            },
            {
                "variant": "single_phase_channel_exclude_event5",
                "min_any_outside_rate": 0.001,
                "min_top_rate": 0.001,
                "max_active_buses_01": 1,
                "max_active_features_01": 2,
                "max_active_signals_01": 2,
                "max_active_phase_channels_01": 1,
                "exclude_event5_nanmask": True,
                "require_single_top_bus": True,
            },
        ]
    )
    y = summary_df["true_event7"].to_numpy(dtype=bool)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for variant in variants:
        pred = _apply_variant(summary_df, variant)
        metric_rows.append({**variant, **_metrics(y, pred)})
        for chunk_name, true_event7, pred_event7 in zip(summary_df["chunk_name"], y, pred):
            prediction_rows.append(
                {
                    "variant": variant["variant"],
                    "chunk_name": chunk_name,
                    "true_event7": bool(true_event7),
                    "pred_event7": bool(pred_event7),
                }
            )
    metrics_df = pd.DataFrame(metric_rows).sort_values(
        ["f1_event7", "accuracy", "specificity_non7", "recall_event7"],
        ascending=False,
    )
    predictions_df = pd.DataFrame(prediction_rows)
    return metrics_df, predictions_df


def build_report(chunk_root: Path, output: Path, feature_csv: Path, variant_csv: Path, predictions_csv: Path) -> dict[str, Any]:
    tester = Event7BadDataHypothesisTester()
    summaries: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    for chunk_dir in sorted([p for p in chunk_root.iterdir() if p.is_dir() and "_event" in p.name], key=lambda p: int(re.search(r"\d+", p.name).group(0))):
        summary, rows = tester.feature_outside_summary(chunk_dir)
        summaries.append(summary)
        feature_rows.extend(rows)

    summary_df = pd.DataFrame(summaries)
    pd.DataFrame(feature_rows).to_csv(feature_csv, index=False)
    metrics_df, predictions_df = evaluate_variants(summary_df)
    metrics_df.to_csv(variant_csv, index=False)
    predictions_df.to_csv(predictions_csv, index=False)
    best_variant = metrics_df.iloc[0].to_dict()
    best_predictions = predictions_df[predictions_df["variant"] == best_variant["variant"]]
    errors = best_predictions[best_predictions["true_event7"] != best_predictions["pred_event7"]]

    report = {
        "schema_version": 1,
        "hypothesis": (
            "Event7 bad data should look like event0 except for a very small number "
            "of out-of-range features concentrated in one bus/signal."
        ),
        "chunk_root": str(chunk_root.resolve()),
        "best_variant": best_variant,
        "best_variant_errors": errors.to_dict(orient="records"),
        "chunk_summaries": summary_df.drop(columns=["bus_rates", "signal_rates", "phase_channel_rates"]).to_dict(orient="records"),
        "artifacts": {
            "feature_csv": str(feature_csv.resolve()),
            "variant_metrics_csv": str(variant_csv.resolve()),
            "variant_predictions_csv": str(predictions_csv.resolve()),
        },
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test event7 one-bus/one-signal bad-data hypothesis.")
    parser.add_argument("--chunk-root", type=Path, default=DEFAULT_CHUNK_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--variant-csv", type=Path, default=DEFAULT_VARIANT_CSV)
    parser.add_argument("--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        chunk_root=args.chunk_root,
        output=args.output,
        feature_csv=args.feature_csv,
        variant_csv=args.variant_csv,
        predictions_csv=args.predictions_csv,
    )
    print(f"Wrote {args.output.resolve()}")
    print(f"Wrote {args.feature_csv.resolve()}")
    print(f"Wrote {args.variant_csv.resolve()}")
    print(f"Wrote {args.predictions_csv.resolve()}")
    print(json.dumps(report["best_variant"], indent=2))
    if report["best_variant_errors"]:
        print("Errors:")
        print(json.dumps(report["best_variant_errors"], indent=2))


if __name__ == "__main__":
    main()
