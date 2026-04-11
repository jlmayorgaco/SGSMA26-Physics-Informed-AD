"""Run the 4x4 estimator/classifier ablation."""

from __future__ import annotations

import logging
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from poc.classifiers import CLASSIFIERS
from poc.classifiers.classifier_c2_lgbm import LightGBMClassifierPOC
from poc.detection.chi2_detector import Chi2InnovationDetector
from poc.estimators import ESTIMATORS
from poc.evaluation.metrics import (
    classification_report_numbers,
    competition_score,
    detection_metrics,
    localization_metrics,
)
from poc.evaluation.plots import save_ablation_heatmap, save_confusion_matrix
from poc.features import make_windows
from poc.localization.cosine_localizer import CosineLocalizer
from poc.schema import observations_from_merged

LOG = logging.getLogger(__name__)


@dataclass
class ScenarioRecord:
    """One synthetic event window source."""

    run_id: int
    label: int
    center: int
    df: pd.DataFrame


@dataclass
class RealEvents:
    """External validation event frames and labels."""

    frames: np.ndarray
    labels: np.ndarray
    buses: list[int]


def run_full_ablation(cfg: dict, repo_root: Path, smoke: bool = False) -> pd.DataFrame:
    """Run all 16 E_i x C_j combinations and write results artifacts."""

    results_dir = repo_root / cfg["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    _attach_run_log(results_dir)
    LOG.info("Environment: python=%s platform=%s seed=%s", sys.version.split()[0], platform.platform(), cfg["seed"])

    raw_dir = repo_root / cfg["paths"]["raw_data"]
    real_df = _load_real(raw_dir)
    baseline = _baseline_observations(real_df, float(cfg["ablation"]["baseline_seconds"]))
    real_events = _real_event_frames(real_df)
    real_obs = observations_from_merged(real_df)

    synthetic = _load_synthetic_records(repo_root, cfg, smoke=smoke)
    if len(synthetic) < 12:
        raise RuntimeError("Need at least 12 synthetic scenarios for a meaningful ablation smoke run.")
    LOG.info("Loaded %d synthetic scenarios for %s ablation", len(synthetic), "smoke" if smoke else "full")

    rows: list[dict] = []
    best_payload: dict | None = None
    localizer = CosineLocalizer()

    for est_name, est_cls in ESTIMATORS.items():
        estimator = est_cls()
        t_fit0 = time.perf_counter()
        estimator.fit(baseline)
        baseline_out = estimator.estimate(baseline)
        detector = Chi2InnovationDetector()
        detector.fit(baseline_out.normalized_score)

        syn_windows, syn_labels = _estimate_synthetic_windows(estimator, synthetic, int(cfg["ablation"]["window_size"]))
        real_t0 = time.perf_counter()
        real_out = estimator.estimate(real_obs)
        real_windows = make_windows(real_out.innovation, real_events.frames, int(cfg["ablation"]["window_size"]))
        est_fit_seconds = time.perf_counter() - t_fit0
        est_real_seconds = time.perf_counter() - real_t0
        det = detector.detect(real_out.normalized_score)
        det_metrics = detection_metrics(det.onsets, real_events.frames)
        top3_real = localizer.batch_localize(real_windows)
        loc_metrics = localization_metrics(top3_real, real_events.buses)

        split = _contiguous_split(len(syn_labels))
        normal_windows, normal_labels = _normal_training_windows(baseline_out.innovation, int(cfg["ablation"]["window_size"]))

        for clf_name, clf_cls in CLASSIFIERS.items():
            classifier = _make_classifier(clf_cls, cfg)
            x_train = np.concatenate([syn_windows[split["train"]], normal_windows], axis=0)
            y_train = np.concatenate([syn_labels[split["train"]], normal_labels], axis=0)
            x_test = syn_windows[split["test"]]
            y_test = syn_labels[split["test"]]

            train0 = time.perf_counter()
            classifier.fit(x_train, y_train)
            train_seconds = time.perf_counter() - train0

            infer0 = time.perf_counter()
            pred_test = classifier.predict(x_test)
            pred_real = classifier.predict(real_windows)
            infer_seconds = time.perf_counter() - infer0 + est_real_seconds

            syn_report = classification_report_numbers(y_test, pred_test)
            real_report = classification_report_numbers(real_events.labels, pred_real)
            params = int(estimator.count_parameters() + classifier.count_parameters())
            gap = syn_report["macro_f1"] - real_report["macro_f1"]
            score = competition_score(real_report["macro_f1"], params, float(cfg["ablation"]["lambda_params"]))

            row = {
                "estimator": est_name,
                "classifier": clf_name,
                "macro_f1_synthetic": syn_report["macro_f1"],
                "macro_f1_real": real_report["macro_f1"],
                "accuracy_synthetic": float(np.mean(pred_test == y_test)) if len(y_test) else 0.0,
                "accuracy_real": float(np.mean(pred_real == real_events.labels)) if len(real_events.labels) else 0.0,
                "gap_synthetic_minus_real": gap,
                "detection_precision": det_metrics["precision"],
                "detection_recall": det_metrics["recall"],
                "detection_f1": det_metrics["f1"],
                "localization_top1": loc_metrics["top1"],
                "localization_top3": loc_metrics["top3"],
                "total_params": params,
                "competition_score": score,
                "training_time_sec": train_seconds + est_fit_seconds,
                "inference_time_sec": infer_seconds,
            }
            rows.append(row)
            LOG.info(
                "%s x %s: real macro-F1=%.3f synthetic macro-F1=%.3f params=%d score=%.3f",
                est_name,
                clf_name,
                row["macro_f1_real"],
                row["macro_f1_synthetic"],
                params,
                score,
            )

            if best_payload is None or row["competition_score"] > best_payload["row"]["competition_score"]:
                best_payload = {
                    "row": row,
                    "per_class": real_report["per_class_f1"],
                    "cm": real_report["confusion_matrix"],
                    "pred_real": pred_real,
                    "true_real": real_events.labels,
                }

    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "ablation_table.csv", index=False)
    save_ablation_heatmap(table, results_dir / "ablation_heatmap.png")
    if best_payload is not None:
        per_class = pd.DataFrame(
            [{"label": label, "f1": f1} for label, f1 in best_payload["per_class"].items()]
        )
        per_class.to_csv(results_dir / "per_class_f1.csv", index=False)
        save_confusion_matrix(best_payload["cm"], results_dir / "confusion_matrix_best.png")
        _write_report(table, per_class, best_payload, results_dir, smoke=smoke)
    return table


def _attach_run_log(results_dir: Path) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    handler = logging.FileHandler(results_dir / f"run_{stamp}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s - %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)


def _load_real(raw_dir: Path) -> pd.DataFrame:
    from src.io.load_csv import load_all

    LOG.info("Loading real competition data from %s", raw_dir)
    return load_all(raw_dir)


def _baseline_observations(df: pd.DataFrame, seconds: float) -> np.ndarray:
    mask = df["TIMESTAMP"].to_numpy(float) <= seconds
    if "Event" in df:
        mask = mask & (df["Event"].to_numpy(int) == 0)
    obs = observations_from_merged(df.loc[mask])
    if len(obs) < 30:
        obs = observations_from_merged(df.iloc[: max(30, min(len(df), 1800))])
    return obs


def _real_event_frames(df: pd.DataFrame) -> RealEvents:
    """Map the known nine real competition events to nearest labeled transitions."""

    from src.io.label_utils import event_transition_frames

    known = [
        (600.0, 5, 29),
        (1200.0, 1, 39),
        (2400.0, 2, 24),
        (2700.0, 5, 29),
        (3000.0, 6, 29),
        (3000.0, 3, 2),
        (3300.0, 3, 2),
        (3900.0, 4, 7),
        (4200.0, 4, 7),
    ]
    transitions = event_transition_frames(df, ignore_labels={7})
    ts = df["TIMESTAMP"].to_numpy(float)
    ev = df["Event"].to_numpy(int)
    used: set[int] = set()
    frames: list[int] = []
    labels: list[int] = []
    buses: list[int] = []
    for approx, label, bus in known:
        candidates = [
            int(frame)
            for frame in transitions
            if int(frame) not in used and int(ev[int(frame)]) == label and abs(float(ts[int(frame)]) - approx) <= 360.0
        ]
        if candidates:
            frame = min(candidates, key=lambda idx: abs(float(ts[idx]) - approx))
        else:
            frame = int(np.argmin(np.abs(ts - approx)))
        used.add(frame)
        frames.append(frame)
        labels.append(label)
        buses.append(bus)
    return RealEvents(np.asarray(frames, dtype=int), np.asarray(labels, dtype=int), buses)


def _load_synthetic_records(repo_root: Path, cfg: dict, smoke: bool) -> list[ScenarioRecord]:
    """Load existing generated synthetic events and build a balanced manifest."""

    from src.augmentation.andes_sim import load_synthetic

    syn_dir = repo_root / cfg["paths"]["existing_synth"]
    max_records = int(cfg["ablation"]["max_existing_synthetic"])
    if smoke:
        max_records = min(max_records, 84)
    run_ids = []
    for path in sorted(syn_dir.glob("syn*_Bus2_Competition_Data_nanmask.csv")):
        try:
            run_ids.append(int(path.name[3:7]))
        except ValueError:
            continue

    grouped: dict[int, list[ScenarioRecord]] = {}
    for run_id in run_ids[: int(cfg["ablation"]["max_existing_synthetic"])]:
        df = load_synthetic(syn_dir, run_id)
        labels = df["Event"].to_numpy(int)
        nonzero = np.where(labels != 0)[0]
        if len(nonzero) == 0:
            continue
        label = int(labels[nonzero[len(nonzero) // 2]])
        center = int(nonzero[0])
        grouped.setdefault(label, []).append(ScenarioRecord(run_id, label, center, df))

    records = _round_robin_records(grouped)
    return records[:max_records]


def _round_robin_records(grouped: dict[int, list[ScenarioRecord]]) -> list[ScenarioRecord]:
    labels = sorted(grouped)
    cursors = {label: 0 for label in labels}
    out: list[ScenarioRecord] = []
    while True:
        added = False
        for label in labels:
            idx = cursors[label]
            if idx < len(grouped[label]):
                out.append(grouped[label][idx])
                cursors[label] += 1
                added = True
        if not added:
            break
    return out


def _estimate_synthetic_windows(estimator, records: list[ScenarioRecord], window_size: int) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    for rec in records:
        obs = observations_from_merged(rec.df)
        out = estimator.estimate(obs)
        windows.append(make_windows(out.innovation, np.asarray([rec.center]), window_size=window_size)[0])
        labels.append(rec.label)
    return np.stack(windows, axis=0), np.asarray(labels, dtype=int)


def _contiguous_split(n: int) -> dict[str, np.ndarray]:
    train_end = max(1, int(round(0.70 * n)))
    val_end = max(train_end + 1, int(round(0.85 * n)))
    val_end = min(val_end, n - 1)
    idx = np.arange(n)
    return {
        "train": idx[:train_end],
        "val": idx[train_end:val_end],
        "test": idx[val_end:],
    }


def _normal_training_windows(baseline_innovation: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(baseline_innovation)
    if n < window_size:
        return np.empty((0, window_size, baseline_innovation.shape[1])), np.empty(0, dtype=int)
    centers = np.linspace(window_size // 2, n - window_size // 2 - 1, num=min(18, max(1, n // window_size)), dtype=int)
    return make_windows(baseline_innovation, centers, window_size), np.zeros(len(centers), dtype=int)


def _make_classifier(clf_cls, cfg: dict):
    seed = int(cfg["seed"])
    if clf_cls is LightGBMClassifierPOC:
        params = cfg["classifiers"]["lgbm"]
        return clf_cls(
            num_leaves=int(params["num_leaves"]),
            n_estimators=int(params["n_estimators"]),
            min_data_in_leaf=int(params["min_data_in_leaf"]),
            learning_rate=float(params["learning_rate"]),
            seed=seed,
        )
    try:
        return clf_cls(seed=seed)
    except TypeError:
        return clf_cls()


def _write_report(
    table: pd.DataFrame,
    per_class: pd.DataFrame,
    best_payload: dict,
    results_dir: Path,
    smoke: bool,
) -> None:
    best = best_payload["row"]
    red_flags = table[table["gap_synthetic_minus_real"] > 0.15].sort_values(
        "gap_synthetic_minus_real",
        ascending=False,
    )
    weak_classes = per_class[per_class["f1"] < 0.5]["label"].astype(int).tolist()
    lines = [
        "# SGSMA 2026 POC Ablation Report",
        "",
        f"Run mode: {'smoke' if smoke else 'full'} ablation.",
        "",
        "## Winner",
        "",
        (
            f"Best compactness-adjusted row: **{best['estimator']} + {best['classifier']}** "
            f"with real Macro-F1={best['macro_f1_real']:.3f}, "
            f"synthetic Macro-F1={best['macro_f1_synthetic']:.3f}, "
            f"real accuracy={best['accuracy_real']:.3f}, "
            f"synthetic accuracy={best['accuracy_synthetic']:.3f}, "
            f"params={int(best['total_params'])}, score={best['competition_score']:.3f}."
        ),
        "",
        "## Gap Analysis",
        "",
        (
            "The requested synthetic held-out target is 0.99+ accuracy. "
            f"The best row reached {best['accuracy_synthetic']:.3f} synthetic accuracy and "
            f"{best['accuracy_real']:.3f} accuracy on the nine real competition events."
        ),
    ]
    if len(red_flags):
        lines.append(
            f"{len(red_flags)} of 16 combinations have synthetic-real Macro-F1 gap > 0.15. "
            "Treat those rows as distribution-shift red flags."
        )
    else:
        lines.append("No row exceeded the 0.15 Macro-F1 synthetic-real gap threshold.")

    lines += [
        "",
        "## Parameter Efficiency",
        "",
        "The score uses Macro-F1_real - 0.03*log10(params). Zero-parameter rules and LSE remain important baselines because every learned component must earn its parameter penalty.",
        "",
        "## Failure Cases",
        "",
        (
            "Classes with F1 below 0.5 for the winner: "
            + (", ".join(str(label) for label in weak_classes) if weak_classes else "none")
            + "."
        ),
        "",
        "## Lessons Learned",
        "",
        "The POC keeps detector and localizer shared, so classifier differences mostly reflect residual representation quality rather than different event alignment. Large synthetic-real gaps mean the synthetic generator still misses real PMU quirks and stacked cyber-physical timing.",
    ]
    (results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

