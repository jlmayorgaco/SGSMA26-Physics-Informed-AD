from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data_factory.feature_extractor import extract_window_features
from src.utils.io import json_safe as _shared_json_safe


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIM_DIR = ROOT / "data" / "simulated" / "sgsma_2026_factory_v2_rawprofile_5000_pmu_hier"
DEFAULT_OUT = ROOT / "models" / "sgsma_2026_factory_v2_hierarchical"
PHYSICAL_TYPES = {"event1", "event2", "event3", "event4", "event8"}
CYBER_TYPES = {"event5", "event7"}


def _json_safe(value: Any) -> Any:
    return _shared_json_safe(value)


def _event_label(events: list[dict[str, Any]]) -> int:
    if not events:
        return 0
    event_types = [str(event["type"]) for event in events]
    if "event5" in event_types and any(item in event_types for item in ("event1", "event2", "event3", "event4", "event8")):
        return 6
    return int(event_types[0].replace("event", ""))


def _physical_event_label(events: list[dict[str, Any]]) -> int:
    for event in events:
        event_type = str(event.get("type"))
        if event_type in PHYSICAL_TYPES:
            return int(event_type.replace("event", ""))
    return 0


def _location_label(events: list[dict[str, Any]]) -> str:
    physical = [event for event in events if str(event.get("type")) in PHYSICAL_TYPES]
    if physical:
        event = physical[0]
        if event.get("line_from") is not None and event.get("line_to") is not None:
            a, b = int(event["line_from"]), int(event["line_to"])
            return f"LINE{min(a, b)}-{max(a, b)}"
        if event.get("bus_target") is not None:
            return f"BUS{int(event['bus_target'])}"
    cyber = [event for event in events if str(event.get("type")) in CYBER_TYPES]
    if cyber:
        return f"PMU{int(cyber[0].get('pmu_target') or -1)}"
    return "none"


def _location_type(label: str) -> str:
    if label.startswith("BUS"):
        return "BUS"
    if label.startswith("LINE"):
        return "LINE"
    if label.startswith("PMU"):
        return "PMU"
    return "NONE"


def _prefix_features(features: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}__{key}": value for key, value in features.items() if isinstance(value, int | float | np.integer | np.floating)}


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
        features = extract_window_features(frame)
        row.update(_prefix_features(features, bus))
        bus_feature_rows.append(features)
    if bus_feature_rows:
        numeric = pd.DataFrame(bus_feature_rows).select_dtypes(include=[np.number])
        for col in numeric.columns:
            row[f"GLOBAL__{col}__mean"] = float(numeric[col].mean())
            row[f"GLOBAL__{col}__max"] = float(numeric[col].max())
            row[f"GLOBAL__{col}__min"] = float(numeric[col].min())
    return row


def build_dataset(sim_dir: Path, out_csv: Path | None = None, force: bool = False, workers: int = 1) -> pd.DataFrame:
    if out_csv is not None and out_csv.exists() and not force:
        return pd.read_csv(out_csv)
    manifest_paths = sorted(sim_dir.glob("SIM*/manifest.json"))
    rows: list[dict[str, Any]] = []
    if int(workers) <= 1:
        for idx, manifest_path in enumerate(manifest_paths, start=1):
            rows.append(_build_one_row(manifest_path))
            if idx == 1 or idx == len(manifest_paths) or idx % 500 == 0:
                print(f"features {idx}/{len(manifest_paths)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = [pool.submit(_build_one_row, path) for path in manifest_paths]
            for idx, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if idx == 1 or idx == len(manifest_paths) or idx % 500 == 0:
                    print(f"features {idx}/{len(manifest_paths)}", flush=True)
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
        if pd.api.types.is_numeric_dtype(data[col]):
            feature_cols.append(col)
    return data[feature_cols], feature_cols


def _tree(random_state: int, n_estimators: int = 450) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, labels: list[Any] | None = None) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist() if labels is not None else confusion_matrix(y_true, y_pred).tolist(),
        "labels": labels if labels is not None else sorted(pd.Series(y_true).unique().tolist()),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def _has_missing(data: pd.DataFrame) -> np.ndarray:
    present_cols = [col for col in data.columns if col.endswith("__data_present_fraction") or col == "GLOBAL__data_present_fraction__min"]
    nan_cols = [col for col in data.columns if col.endswith("__nan_fraction_max") or col == "GLOBAL__nan_fraction_max__max"]
    if present_cols:
        min_present = data[present_cols].min(axis=1).to_numpy(dtype=float)
    else:
        min_present = np.ones(len(data), dtype=float)
    if nan_cols:
        max_nan = data[nan_cols].max(axis=1).to_numpy(dtype=float)
    else:
        max_nan = np.zeros(len(data), dtype=float)
    return (min_present < 0.999) | (max_nan > 0.001)


def _tune_threshold(y_true: np.ndarray, score: np.ndarray, min_recall: float = 0.90) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2.0 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    candidates = np.where(recall[:-1] >= float(min_recall))[0]
    best = int(candidates[np.nanargmax(f1[candidates])]) if len(candidates) else int(np.nanargmax(f1))
    return float(thresholds[best])


def _positive_proba(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    classes = list(model.named_steps["model"].classes_)
    if 1 not in classes:
        return np.zeros(len(x), dtype=float)
    return proba[:, classes.index(1)]


def _predict_hierarchical(
    x: pd.DataFrame,
    data: pd.DataFrame,
    physical_model: Pipeline,
    bad_data_model: Pipeline,
    missing_model: Pipeline,
    bad_data_threshold: float,
    missing_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    physical_pred = physical_model.predict(x).astype(int)
    missing = _has_missing(data)
    bad_score = _positive_proba(bad_data_model, x)
    bad_pred = bad_score >= bad_data_threshold
    pred = physical_pred.copy()
    pred[(~missing) & bad_pred & (physical_pred != 8)] = 7
    if missing.any():
        missing_score = _positive_proba(missing_model, x.loc[missing])
        composed = missing_score >= missing_threshold
        pred[missing] = np.where(composed, 6, 5)
    return pred.astype(int), physical_pred.astype(int), bad_score


def _fit_localizers(x_train: pd.DataFrame, data_train: pd.DataFrame, random_state: int) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}
    for loc_type in ("BUS", "LINE", "PMU"):
        mask = data_train["location_type"].eq(loc_type)
        if mask.sum() < 2:
            continue
        model = _tree(random_state + hash(loc_type) % 1000, n_estimators=500)
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
        model = _tree(random_state + 2000 + int(physical_label), n_estimators=650)
        model.fit(x_train.loc[mask], data_train.loc[mask, "location_label"])
        models[key] = model
    return models


def _predict_locations(x: pd.DataFrame, pred_event: np.ndarray, physical_pred: np.ndarray, models: dict[str, Pipeline]) -> np.ndarray:
    out = np.full(len(x), "none", dtype=object)
    loc_type = np.full(len(x), "BUS", dtype=object)
    loc_type[np.isin(physical_pred, [2])] = "LINE"
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
    by_type = {}
    for loc_type in type_labels:
        type_mask = true_type.eq(loc_type)
        by_type[loc_type] = {
            "support": int(type_mask.sum()),
            "accuracy": float((true[type_mask].to_numpy() == pred[type_mask].to_numpy()).mean()) if type_mask.any() else 0.0,
        }
    return {
        "exact": _metrics(true, pred, labels=sorted(true.unique().tolist())),
        "type_confusion_matrix": confusion_matrix(true_type, pred_type, labels=type_labels).tolist(),
        "type_labels": type_labels,
        "by_event": by_event,
        "by_type": by_type,
    }


def train_hierarchical(
    sim_dir: Path = DEFAULT_SIM_DIR,
    out_dir: Path = DEFAULT_OUT,
    random_state: int = 20260503,
    force_features: bool = False,
    feature_workers: int = 1,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "factory_features.csv"
    data = build_dataset(sim_dir, out_csv=dataset_path, force=force_features, workers=feature_workers)
    x, feature_cols = _feature_matrix(data)

    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.30,
        random_state=random_state,
        stratify=data["event_label"],
    )
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    data_train, data_test = data.iloc[train_idx].reset_index(drop=True), data.iloc[test_idx].reset_index(drop=True)
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)

    physical_model = _tree(random_state + 1, n_estimators=500)
    physical_model.fit(x_train, data_train["physical_event_label"])

    fit_idx, cal_idx = train_test_split(
        np.arange(len(data_train)),
        test_size=0.25,
        random_state=random_state + 100,
        stratify=data_train["event_label"],
    )

    bad_mask_fit = ~_has_missing(data_train.iloc[fit_idx].reset_index(drop=True))
    x_bad_fit = x_train.iloc[fit_idx].reset_index(drop=True).loc[bad_mask_fit]
    y_bad_fit = data_train.iloc[fit_idx].reset_index(drop=True).loc[bad_mask_fit, "event_label"].eq(7).astype(int)
    bad_cal_data = data_train.iloc[cal_idx].reset_index(drop=True)
    x_bad_cal_all = x_train.iloc[cal_idx].reset_index(drop=True)
    bad_mask_cal = ~_has_missing(bad_cal_data)
    bad_threshold_model = _tree(random_state + 20, n_estimators=450)
    bad_threshold_model.fit(x_bad_fit, y_bad_fit)
    y_bad_cal = bad_cal_data.loc[bad_mask_cal, "event_label"].eq(7).astype(int)
    bad_threshold = _tune_threshold(
        y_bad_cal.to_numpy(),
        _positive_proba(bad_threshold_model, x_bad_cal_all.loc[bad_mask_cal]),
        min_recall=0.80,
    )

    bad_mask_train = ~_has_missing(data_train)
    bad_data_model = _tree(random_state + 2, n_estimators=450)
    y_bad_train = data_train.loc[bad_mask_train, "event_label"].eq(7).astype(int)
    bad_data_model.fit(x_train.loc[bad_mask_train], y_bad_train)

    missing_fit_data = data_train.iloc[fit_idx].reset_index(drop=True)
    x_missing_fit_all = x_train.iloc[fit_idx].reset_index(drop=True)
    missing_mask_fit = _has_missing(missing_fit_data)
    missing_cal_data = data_train.iloc[cal_idx].reset_index(drop=True)
    x_missing_cal_all = x_train.iloc[cal_idx].reset_index(drop=True)
    missing_mask_cal = _has_missing(missing_cal_data)
    missing_threshold_model = _tree(random_state + 30, n_estimators=450)
    missing_threshold_model.fit(
        x_missing_fit_all.loc[missing_mask_fit],
        missing_fit_data.loc[missing_mask_fit, "event_label"].eq(6).astype(int),
    )
    y_missing_cal = missing_cal_data.loc[missing_mask_cal, "event_label"].eq(6).astype(int)
    missing_threshold = _tune_threshold(
        y_missing_cal.to_numpy(),
        _positive_proba(missing_threshold_model, x_missing_cal_all.loc[missing_mask_cal]),
        min_recall=0.95,
    )

    missing_mask_train = _has_missing(data_train)
    missing_model = _tree(random_state + 3, n_estimators=450)
    y_missing_train = data_train.loc[missing_mask_train, "event_label"].eq(6).astype(int)
    missing_model.fit(x_train.loc[missing_mask_train], y_missing_train)

    localizers = _fit_localizers(x_train, data_train, random_state + 10)

    pred_event, pred_physical, bad_score = _predict_hierarchical(
        x_test,
        data_test,
        physical_model,
        bad_data_model,
        missing_model,
        bad_threshold,
        missing_threshold,
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
            "leakage_guard": "Columns containing __label_ are excluded from all feature matrices.",
        },
        "thresholds": {
            "bad_data_probability": bad_threshold,
            "missing_composition_probability": missing_threshold,
        },
        "detector": {
            **_metrics(data_test["abnormal_label"], pred_abnormal, labels=[0, 1]),
            "abnormal_recall": float(recall_score(data_test["abnormal_label"], pred_abnormal, pos_label=1)),
        },
        "classifier": _metrics(data_test["event_label"], pred_event, labels=event_labels),
        "physical_classifier": _metrics(data_test["physical_event_label"], pred_physical, labels=sorted(data["physical_event_label"].unique().tolist())),
        "localizer": _localizer_breakdowns(data_test["location_label"], pred_location, data_test["event_label"]),
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
                    "backend": "hierarchical_physics_informed_v1",
                    "bad_data_probability": bad_threshold,
                    "missing_composition_probability": missing_threshold,
                    "event_logic": [
                        "missing + physical/composition => event6",
                        "missing only => event5",
                        "single-signal bad data => event7",
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
    parser = argparse.ArgumentParser(description="Train hierarchical SGSMA detector/classifier/localizer.")
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--random-state", type=int, default=20260503)
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--feature-workers", type=int, default=1)
    args = parser.parse_args()
    report = train_hierarchical(
        args.sim_dir,
        args.out_dir,
        args.random_state,
        force_features=bool(args.force_features),
        feature_workers=max(1, int(args.feature_workers)),
    )
    print(json.dumps(_json_safe(report), indent=2))


if __name__ == "__main__":
    main()
