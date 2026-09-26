from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split

from src.data_factory.feature_extractor_v2 import extract_window_features_v2
from src.data_factory.train_hierarchical_pipeline import (
    CYBER_TYPES,
    DEFAULT_SIM_DIR,
    PHYSICAL_TYPES,
    _event_label,
    _json_safe,
    _location_label,
    _location_type,
    _metrics,
    _physical_event_label,
    _positive_proba,
    _prefix_features,
    _tree,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "models" / "sgsma_2026_factory_v2_transfer_v2"
UNSTABLE_FEATURE_TOKENS = ("freq_angle_mismatch",)
FAULT_VOLTAGE_SPAN_MIN = 50_000.0
FAULT_VOLTAGE_DERIVATIVE_MIN = 500_000.0
LINE_CURRENT_SPAN_MIN = 500.0
LINE_CURRENT_DERIVATIVE_MIN = 3_000.0
LINE_VOLTAGE_SPAN_MAX = 50_000.0
BAD_SINGLE_SIGNAL_SCORE_MIN = 200_000.0
BAD_SINGLE_SIGNAL_RATIO_MIN = 3.0
LINE_GATE_MAX_PHYSICAL_CONFIDENCE = 0.75
FAULT_GUARD_MAX_PHYSICAL_CONFIDENCE = 0.60


def _build_one_row(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sim_id = str(manifest["sim_id"])
    scenario = json.loads(Path(manifest["scenario_json"]).read_text(encoding="utf-8"))
    events = list(scenario.get("events", []))
    location = _location_label(events)
    row: dict[str, Any] = {
        "sim_id": sim_id,
        "event_label": _event_label(events),
        "abnormal_label": int(bool(events)),
        "physical_event_label": _physical_event_label(events),
        "location_label": location,
        "location_type": _location_type(location),
    }
    pmu_dir = Path(manifest["pmu_dir"])
    bus_feature_rows = []
    for csv_path in sorted(pmu_dir.glob("Bus*_Competition_Data_sim.csv")):
        frame = pd.read_csv(csv_path)
        bus = csv_path.name.split("_", 1)[0].upper()
        features = extract_window_features_v2(frame)
        row.update(_prefix_features(features, bus))
        bus_feature_rows.append(features)
    if bus_feature_rows:
        numeric = pd.DataFrame(bus_feature_rows).select_dtypes(include=[np.number])
        for col in numeric.columns:
            row[f"GLOBAL__{col}__mean"] = float(numeric[col].mean())
            row[f"GLOBAL__{col}__max"] = float(numeric[col].max())
            row[f"GLOBAL__{col}__min"] = float(numeric[col].min())
            row[f"GLOBAL__{col}__std"] = float(numeric[col].std(ddof=0))
    return row


def build_dataset(sim_dir: Path, out_csv: Path | None = None, force: bool = False, workers: int = 1) -> pd.DataFrame:
    if out_csv is not None and out_csv.exists() and not force:
        return pd.read_csv(out_csv)
    manifest_paths = sorted(sim_dir.glob("SIM*/manifest.json"))
    rows: list[dict[str, Any]] = []
    if int(workers) <= 1:
        for idx, manifest_path in enumerate(manifest_paths, start=1):
            rows.append(_build_one_row(manifest_path))
            if idx == 1 or idx == len(manifest_paths) or idx % 250 == 0:
                print(f"v2 features {idx}/{len(manifest_paths)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = [pool.submit(_build_one_row, path) for path in manifest_paths]
            for idx, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if idx == 1 or idx == len(manifest_paths) or idx % 250 == 0:
                    print(f"v2 features {idx}/{len(manifest_paths)}", flush=True)
    data = pd.DataFrame(rows).sort_values("sim_id").reset_index(drop=True)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(out_csv, index=False)
    return data


def _feature_matrix(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    ignored = {"sim_id", "event_label", "abnormal_label", "physical_event_label", "location_label", "location_type"}
    feature_cols = []
    for col in data.columns:
        if col in ignored:
            continue
        if "__label_" in col or col.endswith("__Event__min") or col.endswith("__Event__max"):
            continue
        if any(token in col for token in UNSTABLE_FEATURE_TOKENS):
            continue
        if pd.api.types.is_numeric_dtype(data[col]):
            feature_cols.append(col)
    return data[feature_cols], feature_cols


def _has_missing(data: pd.DataFrame) -> np.ndarray:
    present_cols = []
    for col in data.columns:
        if col.startswith("BUS") and any(
            col.endswith(f"__data_present_fraction__{segment}") for segment in ("early", "mid", "late")
        ):
            present_cols.append(col)
        elif (
            col.startswith("GLOBAL__")
            and any(f"__data_present_fraction__{segment}__" in col for segment in ("early", "mid", "late"))
            and col.endswith("__min")
        ):
            present_cols.append(col)
    nan_cols = [col for col in data.columns if col.endswith("__nan_fraction_max") or col == "GLOBAL__nan_fraction_max__max"]
    gap_cols = []
    for col in data.columns:
        if col.startswith("BUS") and col.endswith("__max_timestamp_gap_ratio"):
            gap_cols.append(col)
        elif col == "GLOBAL__max_timestamp_gap_ratio__max":
            gap_cols.append(col)
    if present_cols:
        min_present = data[present_cols].min(axis=1).to_numpy(dtype=float)
    else:
        min_present = np.ones(len(data), dtype=float)
    if nan_cols:
        max_nan = data[nan_cols].max(axis=1).to_numpy(dtype=float)
    else:
        max_nan = np.zeros(len(data), dtype=float)
    if gap_cols:
        max_gap = data[gap_cols].max(axis=1).to_numpy(dtype=float)
    else:
        max_gap = np.ones(len(data), dtype=float)
    return (min_present < 0.999) | (max_nan > 0.20) | (max_gap > 1.5)


def _class_probability(model: Any, x: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=float)
    proba = model.predict_proba(x)
    classes = list(model.named_steps["model"].classes_)
    out = np.zeros(len(x), dtype=float)
    for idx, label in enumerate(labels):
        if int(label) in classes:
            out[idx] = float(proba[idx, classes.index(int(label))])
    return out


def _max_abs_matching(data: pd.DataFrame, pattern: str) -> np.ndarray:
    regex = re.compile(pattern)
    cols = [col for col in data.columns if regex.search(col)]
    if not cols:
        return np.zeros(len(data), dtype=float)
    return data[cols].abs().max(axis=1).fillna(0.0).to_numpy(dtype=float)


def _fault_like(data: pd.DataFrame) -> np.ndarray:
    voltage_span = _max_abs_matching(data, r"V[A-C]_MAG__full__span$")
    voltage_derivative = _max_abs_matching(data, r"V[A-C]_MAG__max_abs_derivative$")
    return (voltage_span >= FAULT_VOLTAGE_SPAN_MIN) | (voltage_derivative >= FAULT_VOLTAGE_DERIVATIVE_MIN)


def _line_outage_like(data: pd.DataFrame) -> np.ndarray:
    current_span = _max_abs_matching(data, r"I[A-C]_MAG__full__span$")
    current_derivative = _max_abs_matching(data, r"I[A-C]_MAG__max_abs_derivative$")
    voltage_span = _max_abs_matching(data, r"V[A-C]_MAG__full__span$")
    return (
        (current_span >= LINE_CURRENT_SPAN_MIN)
        & (current_derivative >= LINE_CURRENT_DERIVATIVE_MIN)
        & (voltage_span <= LINE_VOLTAGE_SPAN_MAX)
    )


def _single_signal_bad_data_like(data: pd.DataFrame) -> np.ndarray:
    score = data.get("GLOBAL__single_signal_score_max__max", pd.Series(np.zeros(len(data), dtype=float))).to_numpy(dtype=float)
    ratio = data.get("GLOBAL__single_signal_score_ratio__max", pd.Series(np.zeros(len(data), dtype=float))).to_numpy(dtype=float)
    return (score >= BAD_SINGLE_SIGNAL_SCORE_MIN) & (ratio >= BAD_SINGLE_SIGNAL_RATIO_MIN)


def _best_binary_threshold(y_true: np.ndarray, score: np.ndarray, min_recall: float = 0.85) -> float:
    candidates = np.unique(np.quantile(score[np.isfinite(score)], np.linspace(0.02, 0.98, 97))) if np.isfinite(score).any() else np.array([0.5])
    best_threshold = float(candidates[0])
    best_key = (-1.0, -1.0)
    for threshold in candidates:
        pred = score >= threshold
        recall = recall_score(y_true, pred.astype(int), zero_division=0)
        f1 = f1_score(y_true, pred.astype(int), zero_division=0)
        if recall >= min_recall:
            key = (f1, -float(threshold))
        else:
            key = (0.25 * f1, recall)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _best_composition_thresholds(
    y_true: np.ndarray,
    missing_score: np.ndarray,
    physical_pred: np.ndarray,
    physical_confidence: np.ndarray,
) -> tuple[float, float]:
    score_candidates = np.unique(np.quantile(missing_score[np.isfinite(missing_score)], np.linspace(0.05, 0.98, 60)))
    if len(score_candidates) == 0:
        score_candidates = np.array([0.5])
    conf_candidates = np.array([0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80])
    best = (-1.0, 0.0, 0.5, 0.0)
    for score_threshold in score_candidates:
        for conf_threshold in conf_candidates:
            pred = (missing_score >= score_threshold) & (physical_pred != 0) & (physical_confidence >= conf_threshold)
            macro = f1_score(y_true, pred.astype(int), zero_division=0)
            recall = recall_score(y_true, pred.astype(int), zero_division=0)
            key = (macro, recall, -float(score_threshold), -float(conf_threshold))
            if key > best:
                best = key
    return float(best[2] * -1.0), float(best[3] * -1.0)


class PhysicsRuleEngine:
    def __init__(
        self,
        bad_data_threshold: float,
        missing_threshold: float,
        missing_physical_confidence: float,
    ) -> None:
        self.bad_data_threshold = float(bad_data_threshold)
        self.missing_threshold = float(missing_threshold)
        self.missing_physical_confidence = float(missing_physical_confidence)

    def predict(
        self,
        x: pd.DataFrame,
        data: pd.DataFrame,
        physical_model: Any,
        bad_data_model: Any,
        missing_model: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        physical_pred = physical_model.predict(x).astype(int)
        physical_confidence = _class_probability(physical_model, x, physical_pred)
        missing = _has_missing(data)
        bad_score = _positive_proba(bad_data_model, x)
        line_rule = _line_outage_like(data)
        fault_rule = _fault_like(data)
        single_signal_rule = _single_signal_bad_data_like(data)
        pred = physical_pred.copy()
        self.apply_line_outage_gate(pred, physical_pred, physical_confidence, line_rule, missing)
        self.apply_fault_guard(pred, physical_pred, physical_confidence, line_rule, fault_rule, missing)
        self.apply_bad_data_rules(pred, physical_pred, bad_score, single_signal_rule, missing)
        self.apply_missing_composition(pred, x, missing, physical_pred, physical_confidence, missing_model)
        return pred.astype(int), physical_pred.astype(int), bad_score, physical_confidence

    def apply_line_outage_gate(
        self,
        pred: np.ndarray,
        physical_pred: np.ndarray,
        physical_confidence: np.ndarray,
        line_rule: np.ndarray,
        missing: np.ndarray,
    ) -> None:
        pred[
            (physical_pred == 1)
            & line_rule
            & (~missing)
            & (physical_confidence <= LINE_GATE_MAX_PHYSICAL_CONFIDENCE)
        ] = 2

    def apply_fault_guard(
        self,
        pred: np.ndarray,
        physical_pred: np.ndarray,
        physical_confidence: np.ndarray,
        line_rule: np.ndarray,
        fault_rule: np.ndarray,
        missing: np.ndarray,
    ) -> None:
        pred[
            (physical_pred == 1)
            & (~line_rule)
            & (~fault_rule)
            & (~missing)
            & (physical_confidence <= FAULT_GUARD_MAX_PHYSICAL_CONFIDENCE)
        ] = 0

    def apply_bad_data_rules(
        self,
        pred: np.ndarray,
        physical_pred: np.ndarray,
        bad_score: np.ndarray,
        single_signal_rule: np.ndarray,
        missing: np.ndarray,
    ) -> None:
        pred[(~missing) & (bad_score >= self.bad_data_threshold) & (physical_pred != 8)] = 7
        pred[(~missing) & (physical_pred == 0) & single_signal_rule] = 7

    def apply_missing_composition(
        self,
        pred: np.ndarray,
        x: pd.DataFrame,
        missing: np.ndarray,
        physical_pred: np.ndarray,
        physical_confidence: np.ndarray,
        missing_model: Any,
    ) -> None:
        if not missing.any():
            return
        missing_score = _positive_proba(missing_model, x.loc[missing])
        composed = (
            (missing_score >= self.missing_threshold)
            & (physical_pred[missing] != 0)
            & (physical_confidence[missing] >= self.missing_physical_confidence)
        )
        pred[missing] = np.where(composed, 6, 5)


def _predict_hierarchical(
    x: pd.DataFrame,
    data: pd.DataFrame,
    physical_model: Any,
    bad_data_model: Any,
    missing_model: Any,
    bad_data_threshold: float,
    missing_threshold: float,
    missing_physical_confidence: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    engine = PhysicsRuleEngine(bad_data_threshold, missing_threshold, missing_physical_confidence)
    return engine.predict(x, data, physical_model, bad_data_model, missing_model)


def _fit_localizers(x_train: pd.DataFrame, data_train: pd.DataFrame, random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {}
    stable_seed = {"BUS": 101, "LINE": 202, "PMU": 303}
    for loc_type in ("BUS", "LINE", "PMU"):
        mask = data_train["location_type"].eq(loc_type)
        if mask.sum() < 2 or data_train.loc[mask, "location_label"].nunique() < 2:
            continue
        model = _tree(random_state + stable_seed[loc_type], n_estimators=650)
        model.fit(x_train.loc[mask], data_train.loc[mask, "location_label"])
        models[loc_type] = model
    for physical_label in sorted(data_train["physical_event_label"].unique().tolist()):
        if int(physical_label) == 0:
            continue
        loc_type = "LINE" if int(physical_label) == 2 else "BUS"
        mask = data_train["location_type"].eq(loc_type) & data_train["physical_event_label"].eq(int(physical_label))
        if mask.sum() < 2 or data_train.loc[mask, "location_label"].nunique() < 2:
            continue
        key = f"{loc_type}:event{int(physical_label)}"
        model = _tree(random_state + 2000 + int(physical_label), n_estimators=750)
        model.fit(x_train.loc[mask], data_train.loc[mask, "location_label"])
        models[key] = model
    return models


def _predict_locations(x: pd.DataFrame, pred_event: np.ndarray, physical_pred: np.ndarray, models: dict[str, Any]) -> np.ndarray:
    out = np.full(len(x), "none", dtype=object)
    loc_type = np.full(len(x), "BUS", dtype=object)
    loc_type[np.isin(physical_pred, [2]) | np.isin(pred_event, [2])] = "LINE"
    loc_type[np.isin(pred_event, [5, 7])] = "PMU"
    loc_type[pred_event == 0] = "NONE"
    for physical_label in sorted(set(int(value) for value in physical_pred)):
        if physical_label == 0:
            continue
        current_type = "LINE" if physical_label == 2 else "BUS"
        key = f"{current_type}:event{physical_label}"
        model = models.get(key)
        if model is None:
            continue
        mask = (loc_type == current_type) & (physical_pred == physical_label)
        if mask.any():
            out[mask] = model.predict(x.loc[mask])
    for current_type in ("BUS", "LINE", "PMU"):
        model = models.get(current_type)
        if model is None:
            continue
        mask = (loc_type == current_type) & pd.Series(out).eq("none").to_numpy()
        if mask.any():
            out[mask] = model.predict(x.loc[mask])
    return out


def _localizer_breakdowns(y_loc: pd.Series, pred_loc: np.ndarray, y_event: pd.Series) -> dict[str, Any]:
    mask = y_loc.ne("none")
    true = y_loc.loc[mask].reset_index(drop=True)
    pred = pd.Series(pred_loc[mask], dtype=object).reset_index(drop=True)
    events = y_event.loc[mask].reset_index(drop=True)
    type_labels = ["BUS", "LINE", "PMU"]
    true_type = true.map(_location_type)
    pred_type = pred.map(_location_type)
    by_event = {}
    for event in sorted(events.unique().tolist()):
        event_mask = events.eq(event)
        by_event[str(int(event))] = {
            "support": int(event_mask.sum()),
            "accuracy": float((true[event_mask].to_numpy() == pred[event_mask].to_numpy()).mean()) if event_mask.any() else 0.0,
        }
    return {
        "exact": _metrics(true, pred, labels=sorted(true.unique().tolist())),
        "type_confusion_matrix": confusion_matrix(true_type, pred_type, labels=type_labels).tolist(),
        "type_labels": type_labels,
        "by_event": by_event,
    }


def _feature_importance(model: Any, feature_cols: list[str], limit: int = 40) -> list[dict[str, Any]]:
    estimator = model.named_steps.get("model")
    importance = getattr(estimator, "feature_importances_", None)
    if importance is None:
        return []
    order = np.argsort(importance)[::-1][:limit]
    return [{"feature": feature_cols[int(idx)], "importance": float(importance[int(idx)])} for idx in order]


def train_hierarchical_v2(
    sim_dir: Path = DEFAULT_SIM_DIR,
    out_dir: Path = DEFAULT_OUT,
    random_state: int = 20260503,
    force_features: bool = False,
    feature_workers: int = 1,
    input_feature_csv: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "factory_features_v2.csv"
    if input_feature_csv is not None:
        data = pd.read_csv(input_feature_csv)
        dataset_path = input_feature_csv
    else:
        data = build_dataset(sim_dir, out_csv=dataset_path, force=force_features, workers=feature_workers)
    x, feature_cols = _feature_matrix(data)

    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.30,
        random_state=random_state,
        stratify=data["event_label"],
    )
    x_train, x_test = x.iloc[train_idx].reset_index(drop=True), x.iloc[test_idx].reset_index(drop=True)
    data_train = data.iloc[train_idx].reset_index(drop=True)
    data_test = data.iloc[test_idx].reset_index(drop=True)

    physical_model = _tree(random_state + 1, n_estimators=750)
    physical_model.fit(x_train, data_train["physical_event_label"])

    fit_idx, cal_idx = train_test_split(
        np.arange(len(data_train)),
        test_size=0.25,
        random_state=random_state + 100,
        stratify=data_train["event_label"],
    )
    fit_data = data_train.iloc[fit_idx].reset_index(drop=True)
    cal_data = data_train.iloc[cal_idx].reset_index(drop=True)
    x_fit = x_train.iloc[fit_idx].reset_index(drop=True)
    x_cal = x_train.iloc[cal_idx].reset_index(drop=True)

    bad_fit_mask = ~_has_missing(fit_data)
    bad_data_model = _tree(random_state + 2, n_estimators=650)
    bad_data_model.fit(x_fit.loc[bad_fit_mask], fit_data.loc[bad_fit_mask, "event_label"].eq(7).astype(int))
    bad_cal_mask = ~_has_missing(cal_data)
    bad_threshold = _best_binary_threshold(
        cal_data.loc[bad_cal_mask, "event_label"].eq(7).astype(int).to_numpy(),
        _positive_proba(bad_data_model, x_cal.loc[bad_cal_mask]),
        min_recall=0.75,
    )

    missing_fit_mask = _has_missing(fit_data)
    missing_model = _tree(random_state + 3, n_estimators=650)
    missing_model.fit(x_fit.loc[missing_fit_mask], fit_data.loc[missing_fit_mask, "event_label"].eq(6).astype(int))
    missing_cal_mask = _has_missing(cal_data)
    cal_missing_score = _positive_proba(missing_model, x_cal.loc[missing_cal_mask])
    cal_physical_pred = physical_model.predict(x_cal.loc[missing_cal_mask]).astype(int)
    cal_physical_conf = _class_probability(physical_model, x_cal.loc[missing_cal_mask], cal_physical_pred)
    missing_threshold, missing_physical_conf = _best_composition_thresholds(
        cal_data.loc[missing_cal_mask, "event_label"].eq(6).astype(int).to_numpy(),
        cal_missing_score,
        cal_physical_pred,
        cal_physical_conf,
    )

    localizers = _fit_localizers(x_train, data_train, random_state + 10)

    pred_event, pred_physical, bad_score, physical_confidence = _predict_hierarchical(
        x_test,
        data_test,
        physical_model,
        bad_data_model,
        missing_model,
        bad_threshold,
        missing_threshold,
        missing_physical_conf,
    )
    pred_abnormal = (pred_event != 0).astype(int)
    pred_location = _predict_locations(x_test, pred_event, pred_physical, localizers)

    event_labels = sorted(data["event_label"].unique().tolist())
    report = {
        "dataset": {
            "sim_dir": str(sim_dir.resolve()),
            "features_csv": str(dataset_path.resolve()),
            "n_samples": int(len(data)),
            "n_features": int(len(feature_cols)),
            "train_samples": int(len(train_idx)),
            "test_samples": int(len(test_idx)),
            "event_counts": {str(k): int(v) for k, v in data["event_label"].value_counts().sort_index().items()},
            "location_classes": int(data.loc[data["location_label"].ne("none"), "location_label"].nunique()),
            "feature_backend": "stable_v2_no_freq_angle_mismatch",
            "unstable_features_excluded": list(UNSTABLE_FEATURE_TOKENS),
        },
        "thresholds": {
            "bad_data_probability": float(bad_threshold),
            "missing_composition_probability": float(missing_threshold),
            "missing_physical_confidence": float(missing_physical_conf),
        },
        "detector": {
            **_metrics(data_test["abnormal_label"], pred_abnormal, labels=[0, 1]),
            "abnormal_recall": float(recall_score(data_test["abnormal_label"], pred_abnormal, pos_label=1)),
        },
        "classifier": _metrics(data_test["event_label"], pred_event, labels=event_labels),
        "physical_classifier": _metrics(
            data_test["physical_event_label"],
            pred_physical,
            labels=sorted(data["physical_event_label"].unique().tolist()),
        ),
        "localizer": _localizer_breakdowns(data_test["location_label"], pred_location, data_test["event_label"]),
        "feature_importance": {
            "physical_event_classifier": _feature_importance(physical_model, feature_cols),
            "bad_data_detector": _feature_importance(bad_data_model, feature_cols),
            "missing_composition_detector": _feature_importance(missing_model, feature_cols),
        },
    }

    joblib.dump(physical_model, out_dir / "physical_event_classifier.joblib")
    joblib.dump(bad_data_model, out_dir / "bad_data_detector.joblib")
    joblib.dump(missing_model, out_dir / "missing_composition_detector.joblib")
    joblib.dump(localizers, out_dir / "typed_localizers.joblib")
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (out_dir / "hierarchical_model_config.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "backend": "hierarchical_physics_informed_v2",
                    "bad_data_probability": bad_threshold,
                    "missing_composition_probability": missing_threshold,
                    "missing_physical_confidence": missing_physical_conf,
                    "excluded_feature_tokens": list(UNSTABLE_FEATURE_TOKENS),
                    "physical_gate_params": {
                        "fault_voltage_span_min": FAULT_VOLTAGE_SPAN_MIN,
                        "fault_voltage_derivative_min": FAULT_VOLTAGE_DERIVATIVE_MIN,
                        "line_current_span_min": LINE_CURRENT_SPAN_MIN,
                        "line_current_derivative_min": LINE_CURRENT_DERIVATIVE_MIN,
                        "line_voltage_span_max": LINE_VOLTAGE_SPAN_MAX,
                        "bad_single_signal_score_min": BAD_SINGLE_SIGNAL_SCORE_MIN,
                        "bad_single_signal_ratio_min": BAD_SINGLE_SIGNAL_RATIO_MIN,
                        "line_gate_max_physical_confidence": LINE_GATE_MAX_PHYSICAL_CONFIDENCE,
                        "fault_guard_max_physical_confidence": FAULT_GUARD_MAX_PHYSICAL_CONFIDENCE,
                    },
                    "event_logic": [
                        "physical line-outage gate: large current step + no deep voltage sag => event2",
                        "physical fault guard: event1 requires deep voltage sag or high voltage derivative",
                        "missing + strong physical/composition evidence => event6",
                        "missing only => event5",
                        "single-signal bad data gate/model => event7",
                        "otherwise physical classifier => event0/event1/event2/event3/event4/event8",
                    ],
                    "localizer_logic": "event2 uses LINE localizer, event5/event7 use PMU localizer, remaining physical events use BUS localizer.",
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "training_report.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stable v2 hierarchical SGSMA pipeline.")
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--random-state", type=int, default=20260503)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--feature-workers", type=int, default=1)
    parser.add_argument("--input-feature-csv", type=Path)
    args = parser.parse_args()
    report = train_hierarchical_v2(
        sim_dir=args.sim_dir,
        out_dir=args.out_dir,
        random_state=args.random_state,
        force_features=bool(args.force_features),
        feature_workers=max(1, int(args.feature_workers)),
        input_feature_csv=args.input_feature_csv,
    )
    print(json.dumps(_json_safe(report), indent=2))


if __name__ == "__main__":
    main()
