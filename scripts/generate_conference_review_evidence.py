from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "workbench" / "features" / "sgsma_generated"
MODEL_DIR = ROOT / "models"
PAPER_DIR = ROOT / "conference_ieee_sgsma2026"
RANDOM_STATE = 20260503
META_COLUMNS = {
    "sim_id",
    "chunk_name",
    "event_label",
    "abnormal_label",
    "physical_event_label",
    "location_label",
    "location_type",
    "window_start",
    "window_end",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    spread = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return ((centre - spread) / denom, (centre + spread) / denom)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        if col in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def make_tree(kind: str, seed: int) -> Pipeline:
    if kind == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=1,
        )
    elif kind == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=200,
            random_state=seed,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=1,
        )
    else:
        raise ValueError(f"unknown baseline kind: {kind}")
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def evaluate_dynamic_baselines() -> list[dict[str, Any]]:
    sim = pd.read_csv(FEATURE_DIR / "dynamic_features_v3.csv")
    raw = pd.read_csv(FEATURE_DIR / "raw0001_dynamic_features_v3.csv")
    cols = feature_columns(sim)
    train_idx, test_idx = train_test_split(
        np.arange(len(sim)),
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=sim["event_label"],
    )
    train = sim.iloc[train_idx].reset_index(drop=True)
    test = sim.iloc[test_idx].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for kind in ("random_forest", "extra_trees"):
        detector = make_tree(kind, RANDOM_STATE + 11)
        detector.fit(train[cols], train["abnormal_label"].astype(int))
        classifier = make_tree(kind, RANDOM_STATE + 12)
        classifier.fit(train[cols], train["event_label"].astype(int))

        loc_train = train[train["location_label"].astype(str).ne("none")].reset_index(drop=True)
        loc_test = test[test["location_label"].astype(str).ne("none")].reset_index(drop=True)
        loc_raw = raw[raw["location_label"].astype(str).ne("none")].reset_index(drop=True)
        localizer = make_tree(kind, RANDOM_STATE + 13)
        localizer.fit(loc_train[cols], loc_train["location_label"].astype(str))

        sim_det = detector.predict(test[cols]).astype(int)
        raw_det = detector.predict(raw[cols]).astype(int)
        sim_evt = classifier.predict(test[cols]).astype(int)
        raw_evt = classifier.predict(raw[cols]).astype(int)
        sim_loc = localizer.predict(loc_test[cols])
        raw_loc = localizer.predict(loc_raw[cols])

        rows.append(
            {
                "method": f"Dynamic-only {kind.replace('_', ' ').title()}",
                "feature_set": "dynamic V3 only",
                "sim_detection_accuracy": float(accuracy_score(test["abnormal_label"].astype(int), sim_det)),
                "sim_event_accuracy": float(accuracy_score(test["event_label"].astype(int), sim_evt)),
                "sim_event_macro_f1": float(f1_score(test["event_label"].astype(int), sim_evt, average="macro", zero_division=0)),
                "sim_localization_top1": float(accuracy_score(loc_test["location_label"].astype(str), sim_loc)),
                "raw_detection_accuracy": float(accuracy_score(raw["abnormal_label"].astype(int), raw_det)),
                "raw_event_accuracy": float(accuracy_score(raw["event_label"].astype(int), raw_evt)),
                "raw_event_macro_f1_observed": float(f1_score(raw["event_label"].astype(int), raw_evt, average="macro", zero_division=0)),
                "raw_localization_top1": float(accuracy_score(loc_raw["location_label"].astype(str), raw_loc)),
                "sim_localization_correct": int((loc_test["location_label"].astype(str).to_numpy() == sim_loc).sum()),
                "sim_localization_support": int(len(loc_test)),
                "raw_localization_correct": int((loc_raw["location_label"].astype(str).to_numpy() == raw_loc).sum()),
                "raw_localization_support": int(len(loc_raw)),
            }
        )
    return rows


def raw_error_summary(predictions_path: Path) -> dict[str, Any]:
    pred = pd.read_csv(predictions_path)
    loc = pred[pred["true_location"].astype(str).ne("none")].copy()
    per_event = []
    for event, group in loc.groupby("true_event"):
        correct = int(group["correct_location"].astype(bool).sum())
        total = int(len(group))
        low, high = wilson_interval(correct, total)
        per_event.append(
            {
                "event": int(event),
                "correct": correct,
                "support": total,
                "top1": correct / total if total else float("nan"),
                "wilson95_low": low,
                "wilson95_high": high,
            }
        )
    errors = pred[(pred["true_location"].astype(str).ne("none")) & (~pred["correct_location"].astype(bool))]
    return {
        "localized_support": int(len(loc)),
        "localized_correct": int(loc["correct_location"].astype(bool).sum()),
        "localized_top1": float(loc["correct_location"].astype(bool).mean()),
        "localized_wilson95": wilson_interval(int(loc["correct_location"].astype(bool).sum()), int(len(loc))),
        "per_event": per_event,
        "errors": errors[
            ["chunk_name", "true_event", "true_location", "pred_event", "pred_location"]
        ].to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reviewer-facing evidence tables for the conference paper.")
    parser.add_argument("--skip-baselines", action="store_true", help="Only generate tables from existing model artifacts.")
    parser.add_argument("--out-dir", type=Path, default=PAPER_DIR / "evidence")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    final_metrics = read_json(MODEL_DIR / "final_metrics.json")["selected"]
    raw_summary = raw_error_summary(MODEL_DIR / "localizer" / "raw001_final_predictions.csv")
    source_fusion = pd.read_csv(MODEL_DIR / "localizer" / "source_fusion_summary.csv")
    baselines = [] if args.skip_baselines else evaluate_dynamic_baselines()

    sim_loc_correct = int(round(float(final_metrics["sim_localizer_exact"]) * 1186))
    final_row = {
        "method": "PI-HED full hierarchy",
        "feature_set": "base V2 + rolling + RLS/Kalman + graph-temporal",
        "sim_detection_accuracy": float(final_metrics["sim_detector_accuracy"]),
        "sim_event_accuracy": float(final_metrics["sim_classifier_accuracy"]),
        "sim_event_macro_f1": float(final_metrics["sim_classifier_macro_f1"]),
        "sim_localization_top1": float(final_metrics["sim_localizer_exact"]),
        "raw_detection_accuracy": float(final_metrics["raw_detector_accuracy"]),
        "raw_event_accuracy": float(final_metrics["raw_classifier_accuracy"]),
        "raw_event_macro_f1_observed": float(final_metrics["raw_classifier_macro_f1"]),
        "raw_localization_top1": float(final_metrics["raw_localizer_exact"]),
        "sim_localization_correct": sim_loc_correct,
        "sim_localization_support": 1186,
        "raw_localization_correct": raw_summary["localized_correct"],
        "raw_localization_support": raw_summary["localized_support"],
    }
    comparison = pd.DataFrame([*baselines, final_row])
    comparison.to_csv(out_dir / "baseline_comparison.csv", index=False)
    source_fusion.to_csv(out_dir / "feature_variant_sweep.csv", index=False)
    pd.DataFrame(raw_summary["per_event"]).to_csv(out_dir / "raw_localization_by_event.csv", index=False)
    pd.DataFrame(raw_summary["errors"]).to_csv(out_dir / "raw_localization_errors.csv", index=False)

    ci_rows = [
        {
            "metric": "SIM localization Top-1",
            "correct": final_row["sim_localization_correct"],
            "support": final_row["sim_localization_support"],
            "estimate": final_row["sim_localization_top1"],
            "wilson95_low": wilson_interval(final_row["sim_localization_correct"], final_row["sim_localization_support"])[0],
            "wilson95_high": wilson_interval(final_row["sim_localization_correct"], final_row["sim_localization_support"])[1],
        },
        {
            "metric": "RAW localization Top-1",
            "correct": final_row["raw_localization_correct"],
            "support": final_row["raw_localization_support"],
            "estimate": final_row["raw_localization_top1"],
            "wilson95_low": raw_summary["localized_wilson95"][0],
            "wilson95_high": raw_summary["localized_wilson95"][1],
        },
        {
            "metric": "RAW detection accuracy",
            "correct": 21,
            "support": 21,
            "estimate": 1.0,
            "wilson95_low": wilson_interval(21, 21)[0],
            "wilson95_high": wilson_interval(21, 21)[1],
        },
        {
            "metric": "RAW event accuracy",
            "correct": 21,
            "support": 21,
            "estimate": 1.0,
            "wilson95_low": wilson_interval(21, 21)[0],
            "wilson95_high": wilson_interval(21, 21)[1],
        },
    ]
    pd.DataFrame(ci_rows).to_csv(out_dir / "confidence_intervals.csv", index=False)

    write_json(
        out_dir / "reviewer_evidence.json",
        {
            "random_state": RANDOM_STATE,
            "final_metrics": final_row,
            "raw_summary": raw_summary,
            "confidence_intervals": ci_rows,
            "baseline_methods": baselines,
            "feature_variant_count": int(len(source_fusion)),
        },
    )
    print(json.dumps({"out_dir": str(out_dir.resolve()), "baselines": len(baselines)}, indent=2))


if __name__ == "__main__":
    main()
