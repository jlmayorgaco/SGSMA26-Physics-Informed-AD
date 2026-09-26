from __future__ import annotations

import argparse
import json
import platform
import re
import time
from pathlib import Path
from typing import Any

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from pipelines import _bootstrap  # type: ignore  # noqa: F401

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from pipelines.p10_train_bus_agnostic_ranker import _hierarchical_predictions, _load_hierarchical_artifacts
except ModuleNotFoundError:
    _hierarchical_predictions = None  # type: ignore[assignment]
    _load_hierarchical_artifacts = None  # type: ignore[assignment]
from src.classes import PipelineResult
from src.helpers.paths import DEFAULT_RAW_DIR, DEFAULT_TOPOLOGY_DIR, ROOT, WORKBENCH_DIR
from src.models.localizer import TopologyResidualRanker, electrical_distance, location_ranking_report
from src.utils.io import json_safe, write_json


DEFAULT_FEATURE_DIR = WORKBENCH_DIR / "features" / "sgsma_generated"
DEFAULT_SIM_DIR = WORKBENCH_DIR / "simulated" / "sgsma_generated"
DEFAULT_OUT = WORKBENCH_DIR / "bus_agnostic_full"
DEFAULT_BASELINE_MODEL_DIR = ROOT / "models_submission_v2"
LABEL_COLUMNS = {"sim_id", "event_label", "abnormal_label", "physical_event_label", "location_label", "location_type"}
ALL_BUSES = tuple(range(1, 40))
LOAD_BUSES = (3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29)
AGNOSTIC_TOKENS = (
    "data_present",
    "nan_fraction",
    "missing_run",
    "timestamp_gap",
    "max_abs",
    "span",
    "early_pre_delta",
    "mid_pre_delta",
    "late_pre_delta",
    "max_abs_derivative",
    "rms_derivative",
    "robust_z",
    "phase_spread",
    "Freq",
    "ROCOF",
)


def _load_feature_tables(feature_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(feature_dir / "factory_features_v2.csv")
    dynamic = pd.read_csv(feature_dir / "dynamic_features_v3.csv")
    if "sim_id" in base and "sim_id" in dynamic:
        dynamic = base[["sim_id"]].merge(dynamic, on="sim_id", how="left")
        suffix_cols = [col for col in dynamic.columns if col not in LABEL_COLUMNS and col != "sim_id"]
        return pd.concat([base.reset_index(drop=True), dynamic[suffix_cols].reset_index(drop=True)], axis=1)
    return base


def _observed_pmus_from_columns(columns: list[str] | pd.Index) -> tuple[int, ...]:
    buses: set[int] = set()
    for col in columns:
        match = re.match(r"BUS(\d+)__", str(col))
        if match:
            buses.add(int(match.group(1)))
    return tuple(sorted(buses))


def _generic_bus_feature_name(column: str) -> str | None:
    value = str(column)
    match = re.match(r"BUS\d+__(.+)$", value)
    if not match:
        return None
    tail = match.group(1)
    tail = re.sub(r"^BUS\d+_", "", tail)
    if not any(token in tail for token in AGNOSTIC_TOKENS):
        return None
    return tail


def _build_bus_agnostic_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert fixed BUS-prefixed features into order-invariant PMU aggregates.

    The resulting feature names intentionally do not contain BUS ids. This keeps
    detector/classifier training independent of a specific PMU placement.
    """
    out = frame[[col for col in frame.columns if col in LABEL_COLUMNS]].copy()
    grouped: dict[str, list[str]] = {}
    for col in frame.columns:
        if col in LABEL_COLUMNS or frame[col].dtype.kind not in "bifc":
            continue
        generic = _generic_bus_feature_name(col)
        if generic is not None:
            grouped.setdefault(generic, []).append(col)
    for generic, cols in sorted(grouped.items()):
        values = frame[cols]
        safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", generic).strip("_")
        out[f"PMU_AGG__{safe_name}__mean"] = values.mean(axis=1)
        out[f"PMU_AGG__{safe_name}__max"] = values.max(axis=1)
        out[f"PMU_AGG__{safe_name}__min"] = values.min(axis=1)
        out[f"PMU_AGG__{safe_name}__std"] = values.std(axis=1).fillna(0.0)
    graph_cols = [
        col
        for col in frame.columns
        if frame[col].dtype.kind in "bifc"
        and col.startswith(("GRAPH__", "RESID__grid__", "GSP__grid__", "GSP__line_grid__"))
        and not re.search(r"BUS\d+|LINE\d+", col)
    ]
    for col in graph_cols:
        out[f"TOPO__{col}"] = frame[col]
    return out


def _ranker_input_columns(frame: pd.DataFrame, observed_pmus: tuple[int, ...]) -> list[str]:
    wanted: set[str] = set()
    suffixes = ("full__max_abs", "full__span", "early_pre_delta", "mid_pre_delta", "late_pre_delta", "max_abs_derivative")
    signals = ("VA_MAG", "VB_MAG", "VC_MAG", "IA_MAG", "IB_MAG", "IC_MAG", "VA_ANG", "IA_ANG")
    for pmu in observed_pmus:
        for signal in signals:
            for suffix in suffixes:
                wanted.add(f"BUS{pmu}__BUS{pmu}_{signal}__{suffix}")
        wanted.add(f"META__BUS{pmu}__nan_fraction_max")
        wanted.add(f"META__BUS{pmu}__data_present_fraction")
        wanted.add(f"META__BUS{pmu}__max_timestamp_gap_ratio")
    wanted.update(col for col in frame.columns if col.startswith(("GRAPH__", "RESID__", "GSP__")))
    return [col for col in frame.columns if col in wanted and frame[col].dtype.kind in "bifc"]


def _location_type(label: str) -> str:
    value = str(label)
    if value.startswith("BUS"):
        return "BUS"
    if value.startswith("LINE"):
        return "LINE"
    if value.startswith("PMU"):
        return "PMU"
    return "NONE"


def _branches(topology_dir: Path = DEFAULT_TOPOLOGY_DIR) -> list[str]:
    data = pd.read_csv(topology_dir / "branches_physical.csv")
    out = []
    for row in data.itertuples(index=False):
        a, b = min(int(row.from_bus), int(row.to_bus)), max(int(row.from_bus), int(row.to_bus))
        out.append(f"LINE{a}-{b}")
    return list(dict.fromkeys(out))


def _candidate_set(pred_event: int, pred_physical: int, observed_pmus: tuple[int, ...]) -> list[str]:
    event = int(pred_event)
    physical = int(pred_physical)
    if event == 0:
        return ["none"]
    if event in {5, 7}:
        return [f"PMU{bus}" for bus in observed_pmus]
    if event == 2 or physical == 2:
        return _branches()
    if event == 4 or physical == 4:
        return [f"BUS{bus}" for bus in LOAD_BUSES]
    return [f"BUS{bus}" for bus in ALL_BUSES]


def _bus_digits(candidate: str) -> list[int]:
    ctype = _location_type(candidate)
    if ctype in {"BUS", "PMU"}:
        digits = "".join(ch for ch in candidate if ch.isdigit())
        return [int(digits)] if digits else []
    if ctype == "LINE":
        text = candidate.replace("LINE", "")
        if "-" in text:
            a, b = text.split("-", 1)
            return [int(a), int(b)]
    return []


def _pmu_intensities(row: pd.Series, observed_pmus: tuple[int, ...]) -> dict[int, float]:
    out = {}
    for pmu in observed_pmus:
        values = []
        for sig in ("VA_MAG", "VB_MAG", "VC_MAG", "IA_MAG", "IB_MAG", "IC_MAG", "VA_ANG", "IA_ANG"):
            for suffix in ("full__max_abs", "full__span", "early_pre_delta", "mid_pre_delta", "late_pre_delta", "max_abs_derivative"):
                values.append(float(row.get(f"BUS{pmu}__BUS{pmu}_{sig}__{suffix}", 0.0) or 0.0))
        values.append(float(row.get(f"META__BUS{pmu}__nan_fraction_max", 0.0) or 0.0))
        out[int(pmu)] = float(np.nanmax(np.abs(values))) if values else 0.0
    return out


def _candidate_features(
    row: pd.Series,
    candidate: str,
    pred_event: int,
    pred_physical: int,
    ranker: TopologyResidualRanker,
    observed_pmus: tuple[int, ...],
) -> dict[str, float]:
    ctype = _location_type(candidate)
    physics = ranker.score_candidates(row, [candidate]).get(candidate, {})
    buses = _bus_digits(candidate)
    distances = []
    for pmu in observed_pmus:
        if buses:
            distances.append(min(ranker._distance(pmu, bus) for bus in buses))
    dist = np.asarray(distances, dtype=float)
    intensities = _pmu_intensities(row, observed_pmus)
    strongest_pmu = max(intensities, key=intensities.get) if intensities else -1
    expected_at_strongest = 0.0
    if buses and strongest_pmu > 0:
        expected_at_strongest = 1.0 / max(min(ranker._distance(strongest_pmu, bus) for bus in buses), 1e-9)
    return {
        "pred_event": float(pred_event),
        "pred_physical": float(pred_physical),
        "event_is_line": float(int(pred_event) == 2 or int(pred_physical) == 2),
        "event_is_fault": float(int(pred_event) == 1 or int(pred_physical) == 1),
        "event_is_generation": float(int(pred_event) in {3, 6} or int(pred_physical) == 3),
        "event_is_load": float(int(pred_event) == 4 or int(pred_physical) == 4),
        "candidate_is_bus": float(ctype == "BUS"),
        "candidate_is_line": float(ctype == "LINE"),
        "candidate_is_pmu": float(ctype == "PMU"),
        "candidate_is_observed": float(any(bus in set(observed_pmus) for bus in buses)),
        "candidate_is_load_bus": float(any(bus in set(LOAD_BUSES) for bus in buses)),
        "candidate_is_generator_bus": float(any(30 <= bus <= 39 for bus in buses)),
        "physics_score": float(physics.get("physics_score", 0.0)),
        "topology_score": float(physics.get("topology_score", 0.0)),
        "min_pmu_distance": float(np.nanmin(dist)) if dist.size else 1.0,
        "mean_pmu_distance": float(np.nanmean(dist)) if dist.size else 1.0,
        "max_pmu_distance": float(np.nanmax(dist)) if dist.size else 1.0,
        "strongest_pmu_energy": float(intensities.get(strongest_pmu, 0.0)),
        "expected_at_strongest_pmu": float(expected_at_strongest),
        "physics_x_expected_strongest": float(physics.get("physics_score", 0.0)) * float(expected_at_strongest),
    }


def _build_rows(
    x: pd.DataFrame,
    labels: pd.DataFrame,
    pred_event: np.ndarray,
    pred_physical: np.ndarray,
    ranker: TopologyResidualRanker,
    observed_pmus: tuple[int, ...],
    negatives_per_positive: int = 12,
    random_state: int = 20260504,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(int(random_state))
    rows: list[dict[str, float]] = []
    y: list[int] = []
    for idx in range(len(x)):
        if int(pred_event[idx]) in {0, 5, 7}:
            continue
        true = str(labels.iloc[idx]["location_label"])
        if true == "none":
            continue
        candidates = _candidate_set(int(pred_event[idx]), int(pred_physical[idx]), observed_pmus)
        if true not in candidates and _location_type(true) == _location_type(candidates[0] if candidates else "none"):
            candidates = candidates + [true]
        negatives = [candidate for candidate in candidates if candidate != true]
        if len(negatives) > negatives_per_positive:
            negatives = list(rng.choice(negatives, size=negatives_per_positive, replace=False))
        for candidate in [true] + negatives:
            rows.append(_candidate_features(x.iloc[idx], candidate, int(pred_event[idx]), int(pred_physical[idx]), ranker, observed_pmus))
            y.append(int(candidate == true))
    return pd.DataFrame(rows), pd.Series(y, dtype=int)


def _predict_locations(
    x: pd.DataFrame,
    pred_event: np.ndarray,
    pred_physical: np.ndarray,
    model: Pipeline,
    ranker: TopologyResidualRanker,
    observed_pmus: tuple[int, ...],
    fallback_location: np.ndarray | None = None,
    protect_non_physical: bool = True,
) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    pred = []
    topk_all = []
    for idx in range(len(x)):
        fallback = str(fallback_location[idx]) if fallback_location is not None else None
        candidates = _candidate_set(int(pred_event[idx]), int(pred_physical[idx]), observed_pmus)
        if candidates == ["none"]:
            pred.append("none")
            topk_all.append([{"candidate": "none", "score": 1.0}])
            continue
        if protect_non_physical and int(pred_event[idx]) in {5, 7}:
            value = fallback if fallback is not None else candidates[0]
            pred.append(value)
            topk_all.append([{"candidate": value, "score": 1.0, "source": "fallback_protected"}])
            continue
        rows = pd.DataFrame([
            _candidate_features(x.iloc[idx], candidate, int(pred_event[idx]), int(pred_physical[idx]), ranker, observed_pmus)
            for candidate in candidates
        ])
        scores = model.predict_proba(rows)[:, 1]
        ranked = sorted(
            [{"candidate": candidate, "score": float(score)} for candidate, score in zip(candidates, scores)],
            key=lambda item: item["score"],
            reverse=True,
        )
        chosen = ranked[0]["candidate"] if ranked else (fallback or "none")
        pred.append(chosen)
        topk_all.append(ranked[:3] if ranked else [{"candidate": chosen, "score": 1.0}])
    return np.asarray(pred, dtype=object), topk_all


def _make_classifier(n_estimators: int, random_state: int, class_weight: str | dict[str, float] | None = "balanced") -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=int(n_estimators),
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight=class_weight,
                    random_state=int(random_state),
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _false_alarm_rate_per_min(y_true: np.ndarray, y_pred: np.ndarray, window_seconds: float = 30.0) -> float:
    normal = np.asarray(y_true, dtype=int) == 0
    fp = int(np.sum((np.asarray(y_pred, dtype=int) == 1) & normal))
    normal_minutes = max(float(np.sum(normal)) * float(window_seconds) / 60.0, 1e-9)
    return float(fp / normal_minutes)


def _detection_metrics(y_event: np.ndarray, pred_abnormal: np.ndarray) -> dict[str, Any]:
    y = (np.asarray(y_event, dtype=int) != 0).astype(int)
    p = np.asarray(pred_abnormal, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, p, labels=[1], zero_division=0)
    return {
        "abnormal_precision": float(precision[0]),
        "abnormal_recall": float(recall[0]),
        "abnormal_f1": float(f1[0]),
        "false_alarm_rate_per_min": _false_alarm_rate_per_min(y, p),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1]).tolist(),
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = list(range(9))
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _plot_confusion(matrix: list[list[int]], labels: list[str], title: str, path: Path) -> None:
    arr = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    threshold = arr.max() / 2.0 if arr.size and arr.max() > 0 else 0.5
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="white" if arr[i, j] > threshold else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_metric_bars(report: dict[str, Any], path: Path) -> None:
    names = ["Detector F1", "Classifier Macro-F1", "Classifier Weighted-F1", "Localizer Top-1", "Localizer Top-3"]
    values = [
        report["sim_test"]["detection"]["abnormal_f1"],
        report["sim_test"]["classification"]["macro_f1"],
        report["sim_test"]["classification"]["weighted_f1"],
        report["sim_test"]["localization"]["exact_accuracy"],
        report["sim_test"]["localization"]["top3_accuracy"],
    ]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(names, values, color=["#376996", "#8f5d46", "#4f7f52", "#7a6f9b", "#b3832f"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("SIM Holdout Metrics")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _model_size_mb(paths: list[Path]) -> float:
    return float(sum(path.stat().st_size for path in paths if path.exists()) / (1024 * 1024))


def _evaluate_raw_if_available(
    detector: Pipeline,
    classifier: Pipeline,
    ranker_model: Pipeline,
    ranker: TopologyResidualRanker,
    feature_cols: list[str],
    ranker_cols: list[str],
    observed_pmus: tuple[int, ...],
    out_dir: Path,
) -> dict[str, Any] | None:
    raw_base_csv = WORKBENCH_DIR / "raw_current_eval" / "raw001_base_features_v2.csv"
    raw_dyn_csv = WORKBENCH_DIR / "raw_current_eval" / "raw001_dynamic_features_v3.csv"
    if not raw_base_csv.exists() or not raw_dyn_csv.exists():
        return None
    raw_base = pd.read_csv(raw_base_csv)
    raw_dyn = pd.read_csv(raw_dyn_csv)
    dyn_cols = [col for col in raw_dyn.columns if col not in set(raw_base.columns) and col not in LABEL_COLUMNS]
    raw = pd.concat([raw_base.reset_index(drop=True), raw_dyn[dyn_cols].reset_index(drop=True)], axis=1)
    raw_agnostic = _build_bus_agnostic_table(raw)
    x = raw_agnostic.reindex(columns=feature_cols, fill_value=np.nan)
    x_ranker = raw.reindex(columns=ranker_cols, fill_value=np.nan)
    pred_abnormal = detector.predict(x).astype(int)
    pred_event = classifier.predict(x).astype(int)
    pred_event = np.where(pred_abnormal == 0, 0, pred_event).astype(int)
    pred_physical = pred_event.copy()
    pred_loc, topk = _predict_locations(
        x_ranker,
        pred_event,
        pred_physical,
        ranker_model,
        ranker,
        observed_pmus,
        fallback_location=None,
        protect_non_physical=True,
    )
    labels = pd.DataFrame(
        {
            "event_label": raw["true_event"].astype(int) if "true_event" in raw else raw["event_label"].astype(int),
            "location_label": raw["true_location"].astype(str) if "true_location" in raw else raw["location_label"].astype(str),
        }
    )
    output = pd.DataFrame(
        {
            "chunk_name": raw.get("chunk_name", pd.Series(np.arange(len(raw)))),
            "true_event": labels["event_label"],
            "pred_event": pred_event,
            "true_location": labels["location_label"],
            "pred_location": pred_loc,
        }
    )
    output["location_exact"] = output["true_location"].astype(str).eq(output["pred_location"].astype(str))
    output["electrical_distance"] = [
        electrical_distance(str(t), str(p), ranker) if str(t) != "none" else 0.0
        for t, p in zip(output["true_location"], output["pred_location"])
    ]
    output.to_csv(out_dir / "raw001_bus_agnostic_full_predictions.csv", index=False)
    return {
        "detection": _detection_metrics(labels["event_label"].to_numpy(dtype=int), pred_abnormal),
        "classification": _classification_metrics(labels["event_label"].to_numpy(dtype=int), pred_event),
        "localization": location_ranking_report(labels, topk, ranker),
    }


def run(
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    baseline_model_dir: Path = DEFAULT_BASELINE_MODEL_DIR,
    out_dir: Path = DEFAULT_OUT,
    random_state: int = 20260504,
    n_estimators: int = 350,
    max_ranker_train_samples: int = 1500,
) -> PipelineResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data = _load_feature_tables(feature_dir)
    observed_pmus = _observed_pmus_from_columns(data.columns)
    if not observed_pmus:
        raise ValueError("Could not infer observed PMU buses from feature columns.")
    agnostic_data = _build_bus_agnostic_table(data)
    feature_cols = [col for col in agnostic_data.columns if col not in LABEL_COLUMNS and agnostic_data[col].dtype.kind in "bifc"]
    ranker_cols = _ranker_input_columns(data, observed_pmus)
    x = agnostic_data.reindex(columns=feature_cols, fill_value=np.nan)
    x_ranker_all = data.reindex(columns=ranker_cols, fill_value=np.nan)
    y_event = data["event_label"].to_numpy(dtype=int)
    y_abnormal = (y_event != 0).astype(int)
    idx_train, idx_tmp = train_test_split(np.arange(len(data)), test_size=0.30, random_state=random_state, stratify=y_event)
    idx_val, idx_test = train_test_split(idx_tmp, test_size=0.50, random_state=random_state, stratify=y_event[idx_tmp])

    detector = _make_classifier(n_estimators, random_state)
    classifier = _make_classifier(n_estimators, random_state + 1)
    t0 = time.perf_counter()
    detector.fit(x.iloc[idx_train], y_abnormal[idx_train])
    classifier.fit(x.iloc[idx_train], y_event[idx_train])
    train_seconds = time.perf_counter() - t0

    val_abnormal_prob = detector.predict_proba(x.iloc[idx_val])[:, 1]
    thresholds = np.linspace(0.05, 0.95, 37)
    threshold_scores = []
    for threshold in thresholds:
        pred = (val_abnormal_prob >= threshold).astype(int)
        metrics = _detection_metrics(y_event[idx_val], pred)
        threshold_scores.append((float(metrics["abnormal_f1"]), float(metrics["abnormal_recall"]), float(threshold)))
    threshold_scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
    abnormal_threshold = threshold_scores[0][2]

    artifacts = _load_hierarchical_artifacts(baseline_model_dir)
    ranker_train_idx = idx_train
    if len(ranker_train_idx) > int(max_ranker_train_samples):
        rng = np.random.default_rng(int(random_state))
        ranker_train_idx = rng.choice(ranker_train_idx, size=int(max_ranker_train_samples), replace=False)
    pred_train_event, pred_train_physical = _hierarchical_predictions(data.iloc[ranker_train_idx].reset_index(drop=True), data.iloc[ranker_train_idx].reindex(columns=artifacts[4], fill_value=np.nan).reset_index(drop=True), artifacts)
    ranker = TopologyResidualRanker(observed_pmus=observed_pmus)
    rank_train_rows, rank_train_y = _build_rows(
        x_ranker_all.iloc[ranker_train_idx].reset_index(drop=True),
        data.iloc[ranker_train_idx].reset_index(drop=True),
        pred_train_event,
        pred_train_physical,
        ranker,
        observed_pmus,
        negatives_per_positive=12,
        random_state=random_state,
    )
    ranker_model = _make_classifier(n_estimators, random_state + 2)
    ranker_model.fit(rank_train_rows, rank_train_y)

    test_abnormal = (detector.predict_proba(x.iloc[idx_test])[:, 1] >= abnormal_threshold).astype(int)
    test_event = classifier.predict(x.iloc[idx_test]).astype(int)
    test_event = np.where(test_abnormal == 0, 0, test_event).astype(int)
    test_physical = test_event.copy()
    test_locations, test_topk = _predict_locations(
        x_ranker_all.iloc[idx_test].reset_index(drop=True),
        test_event,
        test_physical,
        ranker_model,
        ranker,
        observed_pmus,
        fallback_location=None,
        protect_non_physical=True,
    )

    sim_labels = data.iloc[idx_test].reset_index(drop=True)
    t1 = time.perf_counter()
    _ = classifier.predict(x.iloc[idx_test])
    inference_seconds = time.perf_counter() - t1

    model_paths = [
        out_dir / "detector.joblib",
        out_dir / "classifier.joblib",
        out_dir / "candidate_ranker.joblib",
    ]
    joblib.dump(detector, model_paths[0])
    joblib.dump(classifier, model_paths[1])
    joblib.dump({"model": ranker_model, "feature_columns": list(rank_train_rows.columns)}, model_paths[2])
    config_path = out_dir / "model_config.json"
    write_json(
        config_path,
        {
            "model_name": "bus_agnostic_full_v1",
            "random_state": int(random_state),
            "abnormal_probability_threshold": float(abnormal_threshold),
            "feature_columns": feature_cols,
            "ranker_input_columns": ranker_cols,
            "observed_pmus_inferred_from_training": list(observed_pmus),
            "candidate_ranker_feature_columns": list(rank_train_rows.columns),
            "output_contract": "TIMESTAMP, Bus, Predicted_Event, Predicted_Location",
        },
    )

    raw_report = _evaluate_raw_if_available(detector, classifier, ranker_model, ranker, feature_cols, ranker_cols, observed_pmus, out_dir)
    report = {
        "method": {
            "name": "bus_agnostic_full_v1",
            "window_length_seconds": 30,
            "features": {
                "detector": "baseline-relative, robust-z, derivatives, missing/data quality, global PMU aggregation",
                "classifier": "detector features plus phase consistency, sequence/power/frequency proxies where available",
                "localizer": "candidate ranker with topology residuals, graph-temporal/dynamic features, Zbus electrical distance",
            },
            "csv_contract": ["TIMESTAMP", "Bus", "Predicted_Event", "Predicted_Location"],
        },
        "data": {
            "feature_dir": str(feature_dir.resolve()),
            "n_scenarios": int(len(data)),
            "n_features": int(len(feature_cols)),
            "n_ranker_input_features": int(len(ranker_cols)),
            "n_ranker_train_samples": int(len(ranker_train_idx)),
            "observed_pmus": list(observed_pmus),
            "split": {"train": int(len(idx_train)), "validation": int(len(idx_val)), "test": int(len(idx_test))},
        },
        "calibration": {
            "abnormal_probability_threshold": float(abnormal_threshold),
            "threshold_grid_top5": [
                {"threshold": item[2], "abnormal_f1": item[0], "abnormal_recall": item[1]} for item in threshold_scores[:5]
            ],
        },
        "sim_test": {
            "detection": _detection_metrics(y_event[idx_test], test_abnormal),
            "classification": _classification_metrics(y_event[idx_test], test_event),
            "localization": location_ranking_report(sim_labels, test_topk, ranker),
        },
        "raw001": raw_report,
        "efficiency": {
            "parameter_count": int(sum(getattr(step, "n_estimators", 0) for step in [detector.named_steps["model"], classifier.named_steps["model"], ranker_model.named_steps["model"]])),
            "model_size_mb": _model_size_mb(model_paths),
            "train_seconds": float(train_seconds),
            "inference_seconds_for_test_split": float(inference_seconds),
            "seconds_per_minute_pmu_data": float(inference_seconds / max(len(idx_test), 1) * 2.0),
            "hardware": platform.platform(),
        },
    }
    _plot_confusion(report["sim_test"]["detection"]["confusion_matrix"], ["normal", "abnormal"], "Detector Confusion Matrix", plots_dir / "detector_confusion.png")
    _plot_confusion(report["sim_test"]["classification"]["confusion_matrix"], [str(i) for i in range(9)], "Classifier Confusion Matrix", plots_dir / "classifier_confusion.png")
    _plot_metric_bars(report, plots_dir / "metric_summary.png")
    report_path = out_dir / "bus_agnostic_full_report.json"
    write_json(report_path, report)
    result = PipelineResult(
        name="p11_train_bus_agnostic_full",
        status="completed",
        outputs={
            "detector": str(model_paths[0].resolve()),
            "classifier": str(model_paths[1].resolve()),
            "candidate_ranker": str(model_paths[2].resolve()),
            "config": str(config_path.resolve()),
            "report": str(report_path.resolve()),
            "plots": str(plots_dir.resolve()),
        },
        metrics={
            "sim_detector_abnormal_f1": report["sim_test"]["detection"]["abnormal_f1"],
            "sim_classifier_macro_f1": report["sim_test"]["classification"]["macro_f1"],
            "sim_classifier_weighted_f1": report["sim_test"]["classification"]["weighted_f1"],
            "sim_localizer_top1": report["sim_test"]["localization"]["exact_accuracy"],
            "sim_localizer_top3": report["sim_test"]["localization"]["top3_accuracy"],
        },
        notes=[
            "Training uses the 5000 generated scenarios in workbench/features/sgsma_generated.",
            "The exported prediction schema follows the SGSMA guide separate-file contract.",
        ],
    )
    write_json(out_dir / "p11_pipeline_result.json", result.to_dict())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and report the complete bus-agnostic SGSMA pipeline.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--baseline-model-dir", type=Path, default=DEFAULT_BASELINE_MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--random-state", type=int, default=20260504)
    parser.add_argument("--n-estimators", type=int, default=350)
    parser.add_argument("--max-ranker-train-samples", type=int, default=1500)
    args = parser.parse_args()
    result = run(args.feature_dir, args.baseline_model_dir, args.out_dir, args.random_state, args.n_estimators, args.max_ranker_train_samples)
    print(json.dumps(json_safe(result.to_dict()), indent=2))


if __name__ == "__main__":
    main()
