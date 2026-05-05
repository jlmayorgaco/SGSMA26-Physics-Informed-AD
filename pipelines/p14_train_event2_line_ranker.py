from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from pipelines import _bootstrap  # type: ignore  # noqa: F401

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from src.classes import PipelineResult
from src.helpers.paths import DEFAULT_TOPOLOGY_DIR, WORKBENCH_DIR
from src.models.localizer import electrical_distance
from src.utils.io import json_safe, write_json


DEFAULT_FEATURE_DIR = WORKBENCH_DIR / "features" / "sgsma_generated"
DEFAULT_RAW_EVAL_DIR = WORKBENCH_DIR / "raw_current_eval"
DEFAULT_OUT = WORKBENCH_DIR / "event2_line_ranker"
LABEL_COLUMNS = {"sim_id", "event_label", "abnormal_label", "physical_event_label", "location_label", "location_type"}


def _load_features(feature_dir: Path) -> pd.DataFrame:
    # Event2 line ranking uses current/angle redistribution signatures that are
    # already present in the base feature table. Loading dynamic features here
    # adds substantial I/O without adding candidate-relative inputs.
    return pd.read_csv(feature_dir / "factory_features_v2.csv")


def _load_raw_features(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "raw001_base_features_v2.csv")


def _lines(topology_dir: Path) -> list[str]:
    data = pd.read_csv(topology_dir / "branches_physical.csv")
    out = []
    for row in data.itertuples(index=False):
        a, b = min(int(row.from_bus), int(row.to_bus)), max(int(row.from_bus), int(row.to_bus))
        out.append(f"LINE{a}-{b}")
    return list(dict.fromkeys(out))


def _line_metadata(topology_dir: Path) -> dict[str, dict[str, float]]:
    data = pd.read_csv(topology_dir / "branches_physical.csv")
    degree: dict[int, int] = {}
    for row in data.itertuples(index=False):
        degree[int(row.from_bus)] = degree.get(int(row.from_bus), 0) + 1
        degree[int(row.to_bus)] = degree.get(int(row.to_bus), 0) + 1
    out: dict[str, dict[str, float]] = {}
    for row in data.itertuples(index=False):
        a, b = min(int(row.from_bus), int(row.to_bus)), max(int(row.from_bus), int(row.to_bus))
        label = f"LINE{a}-{b}"
        r = float(getattr(row, "r", 0.0))
        x = float(getattr(row, "x", 0.0))
        bchg = float(getattr(row, "b", 0.0))
        tap = float(getattr(row, "tap", 1.0))
        shift = float(getattr(row, "shift_deg", 0.0))
        out[label] = {
            "line_r_abs": abs(r),
            "line_x_abs": abs(x),
            "line_b_abs": abs(bchg),
            "line_z_abs": float(math.hypot(r, x)),
            "line_x_over_r": abs(x) / max(abs(r), 1e-9),
            "line_tap_deviation": abs(tap - 1.0),
            "line_shift_abs": abs(shift),
            "line_is_transformer_like": float(abs(tap - 1.0) > 1e-6 or abs(shift) > 1e-6),
            "endpoint_degree_min": float(min(degree.get(a, 0), degree.get(b, 0))),
            "endpoint_degree_max": float(max(degree.get(a, 0), degree.get(b, 0))),
            "endpoint_degree_mean": float((degree.get(a, 0) + degree.get(b, 0)) / 2.0),
        }
    return out


def _line_buses(label: str) -> list[int]:
    match = re.search(r"LINE(\d+)-(\d+)", str(label))
    if not match:
        return []
    return [int(match.group(1)), int(match.group(2))]


def _observed_pmus(row: pd.Series) -> tuple[int, ...]:
    buses = set()
    for col in row.index:
        match = re.match(r"BUS(\d+)__", str(col))
        if match:
            buses.add(int(match.group(1)))
    return tuple(sorted(buses))


def _distances(topology_dir: Path) -> dict[tuple[int, int], float]:
    data = pd.read_csv(topology_dir / "zbus_effective_distance_full.csv")
    out: dict[tuple[int, int], float] = {}
    for row in data.itertuples(index=False):
        a, b = int(row.from_bus), int(row.to_bus)
        d = max(float(row.z_eff_abs), 1e-9)
        out[(a, b)] = d
        out[(b, a)] = d
    return out


def _value(row: pd.Series, key: str) -> float:
    try:
        value = float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _pmu_severity(row: pd.Series, observed: tuple[int, ...]) -> dict[int, float]:
    severity: dict[int, float] = {}
    for pmu in observed:
        current_terms = []
        angle_terms = []
        voltage_terms = []
        for signal in ("IA_MAG", "IB_MAG", "IC_MAG"):
            for suffix in ("full__span", "early_pre_delta", "mid_pre_delta", "late_pre_delta", "max_abs_derivative", "max_abs_robust_z"):
                current_terms.append(abs(_value(row, f"BUS{pmu}__BUS{pmu}_{signal}__{suffix}")))
        for signal in ("VA_ANG", "IA_ANG"):
            for suffix in ("full__span", "early_pre_delta", "mid_pre_delta", "late_pre_delta", "max_abs_derivative", "max_abs_robust_z"):
                angle_terms.append(abs(_value(row, f"BUS{pmu}__BUS{pmu}_{signal}__{suffix}")))
        for signal in ("VA_MAG", "VB_MAG", "VC_MAG"):
            voltage_terms.append(abs(_value(row, f"BUS{pmu}__BUS{pmu}_{signal}__full__max_abs")))
        current = float(np.nanmax(current_terms)) if current_terms else 0.0
        angle = float(np.nanmax(angle_terms)) if angle_terms else 0.0
        voltage = float(np.nanmax(voltage_terms)) if voltage_terms else 0.0
        # Line outages primarily redistribute current and angles; voltage sag is a weak supporting term.
        severity[int(pmu)] = max(current, 0.7 * angle, 0.25 * voltage)
    return severity


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr / max(float(np.linalg.norm(arr)), 1e-12)


def _entropy(values: np.ndarray) -> float:
    arr = np.maximum(np.asarray(values, dtype=float), 0.0)
    total = float(np.sum(arr))
    if total <= 1e-12:
        return 0.0
    p = arr / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def _candidate_features(
    row: pd.Series,
    candidate: str,
    distances: dict[tuple[int, int], float],
    line_meta: dict[str, dict[str, float]] | None = None,
    observed: tuple[int, ...] | None = None,
    severity_map: dict[int, float] | None = None,
) -> dict[str, float]:
    observed = observed or _observed_pmus(row)
    buses = _line_buses(candidate)
    severity_map = severity_map or _pmu_severity(row, observed)
    sev = np.asarray([severity_map.get(pmu, 0.0) for pmu in observed], dtype=float)
    sev_norm = _normalize(sev)
    dist = np.asarray([min(distances.get((pmu, bus), 1.0) for bus in buses) for pmu in observed], dtype=float)
    weighted_distance = float(np.dot(sev, dist) / max(float(np.sum(sev)), 1e-12)) if dist.size else 1.0
    inv_distance = float(np.sum(sev / np.maximum(dist, 1e-9)) / max(float(np.sum(sev)), 1e-12)) if dist.size else 0.0
    cosines = []
    residuals = []
    for tau in (0.05, 0.10, 0.20, 0.40, 0.80):
        expected = []
        for pmu in observed:
            expected.append(max(math.exp(-distances.get((pmu, bus), 1.0) / tau) for bus in buses))
        exp_norm = _normalize(np.asarray(expected, dtype=float))
        cosines.append(float(np.dot(sev_norm, exp_norm)))
        residuals.append(float(np.linalg.norm(sev_norm - exp_norm)))
    sorted_sev = np.sort(sev)[::-1]
    strongest_idx = int(np.nanargmax(sev)) if sev.size else 0
    strongest_pmu = observed[strongest_idx] if observed else -1
    expected_at_strongest = 0.0
    if strongest_pmu > 0 and buses:
        expected_at_strongest = 1.0 / max(min(distances.get((strongest_pmu, bus), 1.0) for bus in buses), 1e-9)
    out = {
        "candidate_is_line": 1.0,
        "candidate_endpoint_observed_count": float(sum(bus in set(observed) for bus in buses)),
        "min_distance_to_observed": float(np.nanmin(dist)) if dist.size else 1.0,
        "mean_distance_to_observed": float(np.nanmean(dist)) if dist.size else 1.0,
        "max_distance_to_observed": float(np.nanmax(dist)) if dist.size else 1.0,
        "weighted_distance_to_severity": weighted_distance,
        "inverse_distance_score": inv_distance,
        "diffusion_cosine_max": float(np.nanmax(cosines)),
        "diffusion_cosine_mean": float(np.nanmean(cosines)),
        "diffusion_residual_min": float(np.nanmin(residuals)),
        "diffusion_residual_mean": float(np.nanmean(residuals)),
        "expected_at_strongest_pmu": float(expected_at_strongest),
        "strongest_pmu_energy": float(sorted_sev[0]) if sorted_sev.size else 0.0,
        "severity_top1_top2_margin": float(sorted_sev[0] - sorted_sev[1]) if sorted_sev.size > 1 else (float(sorted_sev[0]) if sorted_sev.size else 0.0),
        "severity_entropy": _entropy(sev),
        "n_observed_pmus": float(len(observed)),
    }
    if line_meta is not None:
        out.update(line_meta.get(candidate, {}))
    return out


def _build_candidate_rows(
    data: pd.DataFrame,
    lines: list[str],
    distances: dict[tuple[int, int], float],
    line_meta: dict[str, dict[str, float]],
    negatives_per_positive: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(int(random_state))
    rows = []
    y = []
    for _, row in data.iterrows():
        true = str(row["location_label"])
        if true not in lines:
            continue
        observed = _observed_pmus(row)
        severity = _pmu_severity(row, observed)
        negatives = [line for line in lines if line != true]
        # Hard negatives: electrically close to either endpoint, plus random fill.
        true_buses = _line_buses(true)
        scored = []
        for line in negatives:
            buses = _line_buses(line)
            score = min(distances.get((a, b), 1.0) for a in true_buses for b in buses)
            scored.append((score, line))
        scored.sort(key=lambda item: item[0])
        hard_n = max(1, int(0.75 * negatives_per_positive))
        selected = [line for _, line in scored[:hard_n]]
        remaining = [line for _, line in scored[hard_n:]]
        if remaining and len(selected) < negatives_per_positive:
            selected.extend(list(rng.choice(remaining, size=min(len(remaining), negatives_per_positive - len(selected)), replace=False)))
        for candidate in [true] + selected[:negatives_per_positive]:
            feat = _candidate_features(row, candidate, distances, line_meta=line_meta, observed=observed, severity_map=severity)
            feat["sim_id"] = str(row.get("sim_id", ""))
            feat["candidate_label"] = candidate
            feat["true_location"] = true
            rows.append(feat)
            y.append(int(candidate == true))
    frame = pd.DataFrame(rows)
    return frame, pd.Series(y, dtype=int)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"sim_id", "candidate_label", "true_location"}
    return [col for col in frame.columns if col not in excluded and frame[col].dtype.kind in "bifc"]


def _rank(
    rows: pd.DataFrame,
    scores: np.ndarray,
    group_col: str = "sim_id",
) -> pd.DataFrame:
    data = rows[["sim_id", "candidate_label", "true_location"]].copy()
    data["score"] = np.asarray(scores, dtype=float)
    out = []
    for sim_id, group in data.groupby(group_col, sort=False):
        ranked = group.sort_values("score", ascending=False)
        true = str(ranked["true_location"].iloc[0])
        pred = str(ranked["candidate_label"].iloc[0])
        top3 = ranked["candidate_label"].astype(str).head(3).tolist()
        out.append(
            {
                group_col: sim_id,
                "true_location": true,
                "pred_location": pred,
                "exact": int(true == pred),
                "top3_hit": int(true in top3),
                "electrical_distance": electrical_distance(true, pred),
            }
        )
    return pd.DataFrame(out)


def _all_line_rows(data: pd.DataFrame, lines: list[str], distances: dict[tuple[int, int], float], line_meta: dict[str, dict[str, float]], id_col: str) -> pd.DataFrame:
    rows = []
    for _, row in data.iterrows():
        true = str(row["location_label"] if "location_label" in row else row["true_location"])
        sample_id = str(row[id_col])
        observed = _observed_pmus(row)
        severity = _pmu_severity(row, observed)
        for candidate in lines:
            feat = _candidate_features(row, candidate, distances, line_meta=line_meta, observed=observed, severity_map=severity)
            feat["sim_id"] = sample_id
            feat["candidate_label"] = candidate
            feat["true_location"] = true
            rows.append(feat)
    return pd.DataFrame(rows)


def train_and_evaluate(
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    raw_eval_dir: Path = DEFAULT_RAW_EVAL_DIR,
    topology_dir: Path = DEFAULT_TOPOLOGY_DIR,
    out_dir: Path = DEFAULT_OUT,
    random_state: int = 20260504,
    n_estimators: int = 400,
    negatives_per_positive: int = 14,
) -> PipelineResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = _load_features(feature_dir)
    event2 = features[features["event_label"].astype(int).eq(2)].reset_index(drop=True)
    lines = _lines(topology_dir)
    line_meta = _line_metadata(topology_dir)
    distances = _distances(topology_dir)
    train_rows, train_y = _build_candidate_rows(event2, lines, distances, line_meta, negatives_per_positive, random_state)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    idx = np.arange(len(train_rows))
    train_idx, test_idx = next(splitter.split(idx, groups=train_rows["sim_id"].astype(str)))
    cols = _feature_columns(train_rows)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=int(n_estimators),
                    random_state=int(random_state),
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(train_rows.iloc[train_idx].reindex(columns=cols), train_y.iloc[train_idx])
    test_full = _all_line_rows(event2[event2["sim_id"].astype(str).isin(set(train_rows.iloc[test_idx]["sim_id"].astype(str)))], lines, distances, line_meta, "sim_id")
    sim_scores = model.predict_proba(test_full.reindex(columns=cols))[:, 1]
    sim_ranked = _rank(test_full, sim_scores)
    sim_ranked.to_csv(out_dir / "sim_event2_line_predictions.csv", index=False)

    raw_features = _load_raw_features(raw_eval_dir)
    frozen = pd.read_csv(raw_eval_dir / "raw001_frozen_submission_predictions.csv")
    raw_event2 = frozen[frozen["true_event"].astype(int).eq(2)].merge(raw_features, on="chunk_name", how="left")
    raw_event2 = raw_event2.rename(columns={"true_location_x": "true_location"})
    if "true_location" not in raw_event2 and "true_location_y" in raw_event2:
        raw_event2["true_location"] = raw_event2["true_location_y"]
    raw_event2["location_label"] = raw_event2["true_location"].astype(str)
    raw_all = _all_line_rows(raw_event2, lines, distances, line_meta, "chunk_name")
    raw_scores = model.predict_proba(raw_all.reindex(columns=cols))[:, 1]
    raw_ranked = _rank(raw_all, raw_scores)
    raw_ranked.to_csv(out_dir / "raw_event2_line_predictions.csv", index=False)

    before = frozen.copy()
    after = frozen.copy()
    for row in raw_ranked.itertuples(index=False):
        mask = after["chunk_name"].astype(str).eq(str(getattr(row, "sim_id")))
        after.loc[mask & after["pred_event"].astype(int).eq(2), "pred_location"] = str(row.pred_location)
    after["location_exact"] = after["true_location"].astype(str).eq(after["pred_location"].astype(str))
    after["electrical_distance"] = [
        electrical_distance(str(t), str(p)) if str(t) != "none" else 0.0
        for t, p in zip(after["true_location"], after["pred_location"])
    ]
    after.to_csv(out_dir / "raw_frozen_plus_event2_ranker_predictions.csv", index=False)

    loc_mask_before = before["true_location"].astype(str).ne("none")
    loc_mask_after = after["true_location"].astype(str).ne("none")
    report = {
        "feature_contract": {
            "candidate_feature_columns": cols,
            "contains_specific_bus_or_line_feature_names": bool(any(re.search(r"BUS\d+|LINE\d+", col) for col in cols)),
        },
        "sim_event2": {
            "evaluated": int(len(sim_ranked)),
            "top1_accuracy": float(sim_ranked["exact"].mean()) if len(sim_ranked) else 0.0,
            "top3_accuracy": float(sim_ranked["top3_hit"].mean()) if len(sim_ranked) else 0.0,
            "mean_electrical_distance": float(sim_ranked["electrical_distance"].mean()) if len(sim_ranked) else 0.0,
        },
        "raw_event2": {
            "before_location": str(raw_event2["pred_location"].iloc[0]) if "pred_location" in raw_event2 and len(raw_event2) else "",
            "after_location": str(raw_ranked["pred_location"].iloc[0]) if len(raw_ranked) else "",
            "true_location": str(raw_ranked["true_location"].iloc[0]) if len(raw_ranked) else "",
            "exact": bool(raw_ranked["exact"].iloc[0]) if len(raw_ranked) else False,
            "top3_hit": bool(raw_ranked["top3_hit"].iloc[0]) if len(raw_ranked) else False,
            "electrical_distance": float(raw_ranked["electrical_distance"].iloc[0]) if len(raw_ranked) else 1.0,
        },
        "raw_total": {
            "before_exact": float(before.loc[loc_mask_before, "location_exact"].astype(bool).mean()),
            "after_exact": float(after.loc[loc_mask_after, "location_exact"].astype(bool).mean()),
            "before_correct": int(before.loc[loc_mask_before, "location_exact"].astype(bool).sum()),
            "after_correct": int(after.loc[loc_mask_after, "location_exact"].astype(bool).sum()),
            "localized_support": int(loc_mask_after.sum()),
            "after_mean_electrical_distance": float(after.loc[loc_mask_after, "electrical_distance"].mean()),
        },
        "sim_candidate_macro_f1": float(f1_score(train_y.iloc[test_idx], model.predict(train_rows.iloc[test_idx].reindex(columns=cols)), average="macro", zero_division=0)),
    }
    report_path = out_dir / "event2_line_ranker_report.json"
    write_json(report_path, report)
    model_path = out_dir / "event2_line_ranker.joblib"
    joblib.dump({"model": model, "feature_columns": cols, "lines": lines}, model_path)
    result = PipelineResult(
        name="p14_train_event2_line_ranker",
        status="completed",
        outputs={
            "model": str(model_path.resolve()),
            "report": str(report_path.resolve()),
            "raw_predictions": str((out_dir / "raw_frozen_plus_event2_ranker_predictions.csv").resolve()),
        },
        metrics={
            "sim_event2_top1": report["sim_event2"]["top1_accuracy"],
            "sim_event2_top3": report["sim_event2"]["top3_accuracy"],
            "raw_before_exact": report["raw_total"]["before_exact"],
            "raw_after_exact": report["raw_total"]["after_exact"],
            "raw_event2_exact": float(report["raw_event2"]["exact"]),
        },
        notes=[
            "The model receives only candidate-observation relative features; line labels are used only as targets/report identifiers.",
            "RAW evaluation only replaces event2 locations on top of the frozen submission predictions.",
        ],
    )
    write_json(out_dir / "p14_pipeline_result.json", result.to_dict())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train agnostic event2 line-outage ranker and evaluate RAW.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--raw-eval-dir", type=Path, default=DEFAULT_RAW_EVAL_DIR)
    parser.add_argument("--topology-dir", type=Path, default=DEFAULT_TOPOLOGY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--random-state", type=int, default=20260504)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--negatives-per-positive", type=int, default=14)
    args = parser.parse_args()
    result = train_and_evaluate(
        feature_dir=args.feature_dir,
        raw_eval_dir=args.raw_eval_dir,
        topology_dir=args.topology_dir,
        out_dir=args.out_dir,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        negatives_per_positive=args.negatives_per_positive,
    )
    print(json.dumps(json_safe(result.to_dict()), indent=2))


if __name__ == "__main__":
    main()
