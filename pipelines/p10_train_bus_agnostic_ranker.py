from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from pipelines import _bootstrap  # type: ignore  # noqa: F401

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.classes import PipelineResult
from src.data_factory.train_dynamic_localizer_fusion_v3 import _concat_features
from src.data_factory.train_hierarchical_pipeline_v2 import _predict_hierarchical
from src.helpers.paths import DEFAULT_TOPOLOGY_DIR, WORKBENCH_DIR
from src.models.localizer import TopologyResidualRanker, TypedLocalizer, electrical_distance, location_ranking_report
from src.utils.io import json_safe, read_json, write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_FEATURE_DIR = WORKBENCH_DIR / "features" / "sgsma_generated"
DEFAULT_SUBMISSION_V2 = Path("models_submission_v2")
DEFAULT_OUT = WORKBENCH_DIR / "bus_agnostic_ranker_v3"
ALL_BUSES = tuple(range(1, 40))
OBSERVED_PMUS = (2, 5, 6, 10, 19, 22, 29, 39)
LOAD_BUSES = (3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29)
GENERATOR_TERMINAL_BUS = {
    "BUS30": "BUS2",
    "BUS31": "BUS6",
    "BUS32": "BUS10",
    "BUS33": "BUS19",
    "BUS34": "BUS20",
    "BUS35": "BUS22",
    "BUS36": "BUS23",
    "BUS37": "BUS25",
    "BUS38": "BUS29",
}


def _location_type(label: str) -> str:
    value = str(label)
    if value.startswith("BUS"):
        return "BUS"
    if value.startswith("LINE"):
        return "LINE"
    if value.startswith("PMU"):
        return "PMU"
    return "NONE"


def _load_hierarchical_artifacts(model_dir: Path) -> tuple[Any, Any, Any, dict[str, float], list[str]]:
    physical_model = joblib.load(model_dir / "classifiers" / "physical_event_classifier.joblib")
    bad_data_model = joblib.load(model_dir / "detector" / "bad_data_detector.joblib")
    missing_model = joblib.load(model_dir / "detector" / "missing_composition_detector.joblib")
    config = read_json(model_dir / "detector" / "hierarchical_model_config.json")
    thresholds = {
        "bad_data_probability": float(config["bad_data_probability"]),
        "missing_composition_probability": float(config["missing_composition_probability"]),
        "missing_physical_confidence": float(config.get("missing_physical_confidence", 0.0)),
    }
    return physical_model, bad_data_model, missing_model, thresholds, read_json(model_dir / "feature_columns.json")


def _hierarchical_predictions(data: pd.DataFrame, x_base: pd.DataFrame, artifacts: tuple[Any, Any, Any, dict[str, float], list[str]]) -> tuple[np.ndarray, np.ndarray]:
    physical_model, bad_data_model, missing_model, thresholds, _ = artifacts
    pred_event, pred_physical, _, _ = _predict_hierarchical(
        x_base,
        data,
        physical_model,
        bad_data_model,
        missing_model,
        thresholds["bad_data_probability"],
        thresholds["missing_composition_probability"],
        thresholds["missing_physical_confidence"],
    )
    return pred_event.astype(int), pred_physical.astype(int)


def _branches() -> list[str]:
    data = pd.read_csv(DEFAULT_TOPOLOGY_DIR / "branches_physical.csv")
    out = []
    for row in data.itertuples(index=False):
        a, b = min(int(row.from_bus), int(row.to_bus)), max(int(row.from_bus), int(row.to_bus))
        out.append(f"LINE{a}-{b}")
    return list(dict.fromkeys(out))


def _candidate_set(pred_event: int, pred_physical: int, observed_pmus: tuple[int, ...] = OBSERVED_PMUS) -> list[str]:
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


def _pmu_intensities(row: pd.Series) -> dict[int, float]:
    out = {}
    for pmu in OBSERVED_PMUS:
        values = []
        for sig in ("VA_MAG", "VB_MAG", "VC_MAG", "IA_MAG", "IB_MAG", "IC_MAG", "VA_ANG", "IA_ANG"):
            for suffix in ("full__max_abs", "full__span", "early_pre_delta", "mid_pre_delta", "late_pre_delta", "max_abs_derivative"):
                values.append(float(row.get(f"BUS{pmu}__BUS{pmu}_{sig}__{suffix}", 0.0) or 0.0))
        values.append(float(row.get(f"META__BUS{pmu}__nan_fraction_max", 0.0) or 0.0))
        out[pmu] = float(np.nanmax(np.abs(values))) if values else 0.0
    return out


def _candidate_features(
    row: pd.Series,
    candidate: str,
    pred_event: int,
    pred_physical: int,
    ranker: TopologyResidualRanker,
) -> dict[str, float]:
    ctype = _location_type(candidate)
    physics = ranker.score_candidates(row, [candidate]).get(candidate, {})
    buses = _bus_digits(candidate)
    distances = []
    for pmu in OBSERVED_PMUS:
        if buses:
            distances.append(min(ranker._distance(pmu, bus) for bus in buses))
    dist = np.asarray(distances, dtype=float)
    intensities = _pmu_intensities(row)
    strongest_pmu = max(intensities, key=intensities.get) if intensities else -1
    expected_at_strongest = 0.0
    if buses and strongest_pmu > 0:
        expected_at_strongest = 1.0 / max(min(ranker._distance(strongest_pmu, bus) for bus in buses), 1e-9)
    terminal = GENERATOR_TERMINAL_BUS.get(candidate, candidate)
    terminal_buses = _bus_digits(terminal)
    terminal_dist = []
    for pmu in OBSERVED_PMUS:
        if terminal_buses:
            terminal_dist.append(min(ranker._distance(pmu, bus) for bus in terminal_buses))
    terminal_arr = np.asarray(terminal_dist, dtype=float)
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
        "candidate_is_observed": float(any(bus in set(OBSERVED_PMUS) for bus in buses)),
        "candidate_is_load_bus": float(any(bus in set(LOAD_BUSES) for bus in buses)),
        "candidate_is_generator_bus": float(any(30 <= bus <= 39 for bus in buses)),
        "physics_score": float(physics.get("physics_score", 0.0)),
        "topology_score": float(physics.get("topology_score", 0.0)),
        "min_pmu_distance": float(np.nanmin(dist)) if dist.size else 1.0,
        "mean_pmu_distance": float(np.nanmean(dist)) if dist.size else 1.0,
        "max_pmu_distance": float(np.nanmax(dist)) if dist.size else 1.0,
        "terminal_min_pmu_distance": float(np.nanmin(terminal_arr)) if terminal_arr.size else 1.0,
        "strongest_pmu": float(strongest_pmu),
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
    negatives_per_positive: int = 10,
    random_state: int = 20260503,
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
        candidates = _candidate_set(int(pred_event[idx]), int(pred_physical[idx]))
        if true not in candidates and _location_type(true) == _location_type(candidates[0] if candidates else "none"):
            candidates = candidates + [true]
        negatives = [cand for cand in candidates if cand != true]
        if len(negatives) > negatives_per_positive:
            negatives = list(rng.choice(negatives, size=negatives_per_positive, replace=False))
        selected = [true] + negatives
        for candidate in selected:
            rows.append(_candidate_features(x.iloc[idx], candidate, int(pred_event[idx]), int(pred_physical[idx]), ranker))
            y.append(int(candidate == true))
    return pd.DataFrame(rows), pd.Series(y, dtype=int)


def _predict_locations(
    x: pd.DataFrame,
    pred_event: np.ndarray,
    pred_physical: np.ndarray,
    model: Pipeline,
    ranker: TopologyResidualRanker,
    min_margin: float,
    fallback_location: np.ndarray | None = None,
    protect_non_physical: bool = True,
) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    pred = []
    topk_all = []
    for idx in range(len(x)):
        fallback = str(fallback_location[idx]) if fallback_location is not None else None
        candidates = _candidate_set(int(pred_event[idx]), int(pred_physical[idx]))
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
            _candidate_features(x.iloc[idx], candidate, int(pred_event[idx]), int(pred_physical[idx]), ranker)
            for candidate in candidates
        ])
        scores = model.predict_proba(rows)[:, 1]
        ranked = sorted(
            [{"candidate": candidate, "score": float(score)} for candidate, score in zip(candidates, scores)],
            key=lambda item: item["score"],
            reverse=True,
        )
        if len(ranked) >= 2 and ranked[0]["score"] - ranked[1]["score"] < min_margin:
            chosen = fallback if fallback is not None else ranked[0]["candidate"]
        else:
            chosen = ranked[0]["candidate"]
        pred.append(chosen)
        ordered = [{"candidate": chosen, "score": 1.0 if chosen == fallback else ranked[0]["score"], "source": "chosen"}]
        for item in ranked:
            if str(item["candidate"]) != str(chosen) and len(ordered) < 3:
                ordered.append(item)
        topk_all.append(ordered)
    return np.asarray(pred, dtype=object), topk_all


def _load_features(feature_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(feature_dir / "factory_features_v2.csv")
    dyn = pd.read_csv(feature_dir / "dynamic_features_v3.csv")
    if "sim_id" in base and "sim_id" in dyn and not base["sim_id"].astype(str).equals(dyn["sim_id"].astype(str)):
        dyn = base[["sim_id"]].merge(dyn, on="sim_id", how="left")
    return base, dyn


def _report(labels: pd.DataFrame, pred_location: np.ndarray, topk: list[list[dict[str, Any]]]) -> dict[str, Any]:
    return location_ranking_report(labels, topk)


def _write_plots(report: dict[str, Any], raw_predictions_csv: Path | None, out_dir: Path) -> dict[str, str]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    rows = []
    if report.get("sim_typed_fallback"):
        rows.append(("SIM typed fallback", report["sim_typed_fallback"]))
    if report.get("sim_bus_agnostic"):
        rows.append(("SIM guarded ranker", report["sim_bus_agnostic"]))
    if report.get("raw_bus_agnostic"):
        rows.append(("RAW guarded ranker", report["raw_bus_agnostic"]))
    if rows:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        x = np.arange(len(rows))
        exact = [float(item[1].get("exact_accuracy", 0.0)) for item in rows]
        top3 = [float(item[1].get("top3_accuracy", 0.0)) for item in rows]
        width = 0.36
        ax.bar(x - width / 2, exact, width, label="Top-1 exact", color="#2563eb")
        ax.bar(x + width / 2, top3, width, label="Top-3 hit", color="#16a34a")
        ax.set_xticks(x)
        ax.set_xticklabels([item[0] for item in rows], rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_title("Bus-Agnostic Localizer Ranking Metrics")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = plot_dir / "bus_agnostic_metric_summary.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths["metric_summary"] = str(path.resolve())

    if raw_predictions_csv and raw_predictions_csv.exists():
        raw = pd.read_csv(raw_predictions_csv)
        eval_rows = raw[raw["true_location"].astype(str).ne("none")].copy()
        if not eval_rows.empty:
            colors = eval_rows["bus_agnostic_correct"].map({True: "#16a34a", False: "#dc2626"}).fillna("#dc2626")
            labels = eval_rows["chunk_name"].astype(str).str.replace("_", "\n", regex=False)
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.bar(np.arange(len(eval_rows)), eval_rows["bus_agnostic_correct"].astype(int), color=colors)
            ax.set_xticks(np.arange(len(eval_rows)))
            ax.set_xticklabels(labels, rotation=0, fontsize=8)
            ax.set_ylim(0, 1.15)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Wrong", "Correct"])
            ax.set_title("RAW001 Localizer Correctness by Event")
            ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            path = plot_dir / "raw001_correctness_by_event.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths["raw_correctness_by_event"] = str(path.resolve())

            confusion = pd.crosstab(eval_rows["true_location"], eval_rows["bus_agnostic_location"])
            fig, ax = plt.subplots(figsize=(7, 5.5))
            image = ax.imshow(confusion.to_numpy(dtype=float), cmap="Blues")
            ax.set_xticks(np.arange(confusion.shape[1]))
            ax.set_xticklabels(confusion.columns.astype(str), rotation=45, ha="right")
            ax.set_yticks(np.arange(confusion.shape[0]))
            ax.set_yticklabels(confusion.index.astype(str))
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("RAW001 Location Confusion")
            for i in range(confusion.shape[0]):
                for j in range(confusion.shape[1]):
                    value = int(confusion.iat[i, j])
                    if value:
                        ax.text(j, i, str(value), ha="center", va="center", color="black")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            path = plot_dir / "raw001_location_confusion.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths["raw_location_confusion"] = str(path.resolve())

    return paths


def run(
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    submission_v2_dir: Path = DEFAULT_SUBMISSION_V2,
    out_dir: Path = DEFAULT_OUT,
    random_state: int = 20260503,
    negatives_per_positive: int = 10,
    min_margin: float = 0.0,
) -> PipelineResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    base, dyn = _load_features(feature_dir)
    artifacts = _load_hierarchical_artifacts(submission_v2_dir)
    base_cols = artifacts[4]
    bundle = joblib.load(submission_v2_dir / "localizer" / "final_dynamic_localizers.joblib")
    typed = TypedLocalizer(bundle["localizers"])
    localizer_cols = list(bundle["feature_cols"])
    dyn_cols = [col for col in localizer_cols if col not in set(base_cols)]
    x_all = _concat_features(base, dyn, base_cols, dyn_cols).reindex(columns=localizer_cols, fill_value=np.nan)
    train_idx, test_idx = train_test_split(
        np.arange(len(base)),
        test_size=0.30,
        random_state=random_state,
        stratify=base["event_label"],
    )
    x_base_train = base.iloc[train_idx].reindex(columns=base_cols, fill_value=np.nan).reset_index(drop=True)
    x_base_test = base.iloc[test_idx].reindex(columns=base_cols, fill_value=np.nan).reset_index(drop=True)
    pred_train_event, pred_train_physical = _hierarchical_predictions(base.iloc[train_idx].reset_index(drop=True), x_base_train, artifacts)
    pred_test_event, pred_test_physical = _hierarchical_predictions(base.iloc[test_idx].reset_index(drop=True), x_base_test, artifacts)
    ranker = TopologyResidualRanker()
    train_rows, train_y = _build_rows(
        x_all.iloc[train_idx].reset_index(drop=True),
        base.iloc[train_idx].reset_index(drop=True),
        pred_train_event,
        pred_train_physical,
        ranker,
        negatives_per_positive=negatives_per_positive,
        random_state=random_state,
    )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=400,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train_rows, train_y)
    fallback_sim = typed.predict(
        x_all.iloc[test_idx].reset_index(drop=True),
        pred_test_event,
        pred_test_physical,
    )
    fallback_sim_topk = typed.predict_topk(
        x_all.iloc[test_idx].reset_index(drop=True),
        pred_test_event,
        pred_test_physical,
        k=3,
    )
    sim_pred_loc, sim_topk = _predict_locations(
        x_all.iloc[test_idx].reset_index(drop=True),
        pred_test_event,
        pred_test_physical,
        model,
        ranker,
        min_margin=min_margin,
        fallback_location=fallback_sim,
        protect_non_physical=True,
    )
    sim_labels = base.iloc[test_idx].reset_index(drop=True)
    sim_report = _report(sim_labels, sim_pred_loc, sim_topk)
    fallback_sim_report = _report(sim_labels, fallback_sim, fallback_sim_topk)
    raw_report = None
    raw_predictions_csv = None
    raw_base_csv = Path("workbench/raw_current_eval/raw001_base_features_v2.csv")
    raw_dyn_csv = Path("workbench/raw_current_eval/raw001_dynamic_features_v3.csv")
    frozen_csv = submission_v2_dir / "localizer" / "raw001_final_predictions.csv"
    if raw_base_csv.exists() and raw_dyn_csv.exists() and frozen_csv.exists():
        raw_base = pd.read_csv(raw_base_csv).set_index("chunk_name")
        raw_dyn = pd.read_csv(raw_dyn_csv).set_index("chunk_name")
        frozen = pd.read_csv(frozen_csv)
        idx = frozen["chunk_name"].tolist()
        raw_base = raw_base.loc[idx].reset_index()
        raw_dyn = raw_dyn.loc[idx].reset_index()
        raw_x = _concat_features(raw_base, raw_dyn, base_cols, dyn_cols).reindex(columns=localizer_cols, fill_value=np.nan)
        raw_pred_event = frozen["pred_event"].to_numpy(dtype=int)
        raw_pred_physical = raw_pred_event.copy()
        raw_pred_loc, raw_topk = _predict_locations(
            raw_x,
            raw_pred_event,
            raw_pred_physical,
            model,
            ranker,
            min_margin=min_margin,
            fallback_location=frozen["pred_location"].to_numpy(dtype=object),
            protect_non_physical=True,
        )
        raw_labels = pd.DataFrame(
            {
                "event_label": frozen["true_event"].astype(int),
                "abnormal_label": frozen["true_event"].astype(int).ne(0).astype(int),
                "location_label": frozen["true_location"].astype(str),
            }
        )
        raw_report = _report(raw_labels, raw_pred_loc, raw_topk)
        compact = frozen.copy()
        compact["bus_agnostic_location"] = raw_pred_loc
        compact["bus_agnostic_correct"] = compact["true_location"].astype(str).eq(compact["bus_agnostic_location"].astype(str))
        compact["bus_agnostic_distance"] = [
            electrical_distance(str(row.true_location), str(row.bus_agnostic_location))
            if str(row.true_location) != "none"
            else 0.0
            for row in compact.itertuples(index=False)
        ]
        raw_predictions_csv = out_dir / "raw001_bus_agnostic_predictions.csv"
        compact.to_csv(raw_predictions_csv, index=False)
    report = {
        "candidate_policy": "BUS1-39 for physical bus events, all physical lines for event2; event0/event5/event7 use protected typed fallback",
        "train_candidate_rows": int(len(train_rows)),
        "positive_rate": float(train_y.mean()) if len(train_y) else 0.0,
        "negatives_per_positive": int(negatives_per_positive),
        "sim_bus_agnostic": sim_report,
        "sim_typed_fallback": fallback_sim_report,
        "raw_bus_agnostic": raw_report,
    }
    plot_paths = _write_plots(report, raw_predictions_csv, out_dir)
    report["plots"] = plot_paths
    report_json = out_dir / "bus_agnostic_ranker_report.json"
    write_json(report_json, report)
    model_path = out_dir / "bus_agnostic_ranker.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": list(train_rows.columns),
            "negatives_per_positive": int(negatives_per_positive),
            "min_margin": float(min_margin),
            "candidate_policy": report["candidate_policy"],
        },
        model_path,
    )
    result = PipelineResult(
        name="p10_train_bus_agnostic_ranker",
        status="completed",
        outputs={
            "ranker_model": str(model_path.resolve()),
            "report_json": str(report_json.resolve()),
            "raw_predictions_csv": str(raw_predictions_csv.resolve()) if raw_predictions_csv else "",
            "plots_dir": str((out_dir / "plots").resolve()),
        },
        metrics={
            "sim_bus_agnostic_exact": sim_report["exact_accuracy"],
            "sim_bus_agnostic_top3": sim_report["top3_accuracy"],
            "sim_typed_fallback_exact": fallback_sim_report["exact_accuracy"],
            "sim_typed_fallback_top3": fallback_sim_report["top3_accuracy"],
            "raw_bus_agnostic_exact": raw_report["exact_accuracy"] if raw_report else None,
            "raw_bus_agnostic_top3": raw_report["top3_accuracy"] if raw_report else None,
        },
        notes=[
            "Bus/line-agnostic ranker does not depend on the typed localizer Top-K candidate set for physical events.",
            "Event0/event5/event7 are protected by the typed fallback to avoid degrading already-strong cyber/PMU localization.",
        ],
    )
    write_json(out_dir / "p10_pipeline_result.json", result.to_dict())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="P10: train universal bus/line-agnostic localization ranker.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--submission-v2-dir", type=Path, default=DEFAULT_SUBMISSION_V2)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--random-state", type=int, default=20260503)
    parser.add_argument("--negatives-per-positive", type=int, default=10)
    parser.add_argument("--min-margin", type=float, default=0.0)
    args = parser.parse_args()
    result = run(
        feature_dir=args.feature_dir,
        submission_v2_dir=args.submission_v2_dir,
        out_dir=args.out_dir,
        random_state=args.random_state,
        negatives_per_positive=args.negatives_per_positive,
        min_margin=args.min_margin,
    )
    print(json.dumps(json_safe(result.to_dict()), indent=2))


if __name__ == "__main__":
    main()
