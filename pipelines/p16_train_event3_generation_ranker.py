from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

try:
    import _bootstrap  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    from pipelines import _bootstrap  # type: ignore  # noqa: F401

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from pipelines.p15_agnostic_location_guardrails import apply_guardrails
from src.classes import PipelineResult
from src.helpers.paths import DEFAULT_TOPOLOGY_DIR, WORKBENCH_DIR
from src.models.localizer import electrical_distance
from src.utils.io import json_safe, write_json


DEFAULT_FEATURE_DIR = WORKBENCH_DIR / "features" / "sgsma_generated"
DEFAULT_RAW_EVAL_DIR = WORKBENCH_DIR / "raw_current_eval"
DEFAULT_OUT = WORKBENCH_DIR / "event3_generation_ranker"
GENERATOR_BUSES = set(range(30, 40))


def _load_features(feature_dir: Path) -> pd.DataFrame:
    return pd.read_csv(feature_dir / "factory_features_v2.csv")


def _load_raw_features(raw_dir: Path) -> pd.DataFrame:
    return pd.read_csv(raw_dir / "raw001_base_features_v2.csv")


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


def _degree_map(topology_dir: Path) -> dict[int, int]:
    data = pd.read_csv(topology_dir / "branches_physical.csv")
    degree: dict[int, int] = {}
    for row in data.itertuples(index=False):
        a, b = int(row.from_bus), int(row.to_bus)
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    return degree


def _generator_terminal_map(topology_dir: Path) -> dict[int, int]:
    data = pd.read_csv(topology_dir / "branches_physical.csv")
    out: dict[int, int] = {}
    for row in data.itertuples(index=False):
        a, b = int(row.from_bus), int(row.to_bus)
        tap = float(getattr(row, "tap", 1.0))
        shift = float(getattr(row, "shift_deg", 0.0))
        if abs(tap - 1.0) > 1e-6 or abs(shift) > 1e-6 or a in GENERATOR_BUSES or b in GENERATOR_BUSES:
            if a in GENERATOR_BUSES and b not in GENERATOR_BUSES:
                out[a] = b
            elif b in GENERATOR_BUSES and a not in GENERATOR_BUSES:
                out[b] = a
    return out


def _value(row: pd.Series, key: str) -> float:
    try:
        value = float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def _pmu_severity(row: pd.Series, observed: tuple[int, ...]) -> dict[int, float]:
    severity = {}
    for pmu in observed:
        terms = []
        for signal in ("Freq", "ROCOF", "VA_ANG", "IA_ANG"):
            for suffix in ("full__max_abs", "early_pre_delta", "mid_pre_delta", "late_pre_delta", "max_abs_derivative", "max_abs_robust_z"):
                terms.append(abs(_value(row, f"BUS{pmu}__BUS{pmu}_{signal}__{suffix}")))
        for phase in ("A", "B", "C"):
            for suffix in ("early_pre_delta", "mid_pre_delta", "late_pre_delta", "full__max_abs"):
                terms.append(0.35 * abs(_value(row, f"BUS{pmu}__P_PROXY_{phase}__{suffix}")))
        severity[int(pmu)] = float(np.nanmax(terms)) if terms else 0.0
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
    candidate_bus: int,
    distances: dict[tuple[int, int], float],
    degrees: dict[int, int],
    observed: tuple[int, ...] | None = None,
    severity_map: dict[int, float] | None = None,
) -> dict[str, float]:
    observed = observed or _observed_pmus(row)
    severity_map = severity_map or _pmu_severity(row, observed)
    sev = np.asarray([severity_map.get(pmu, 0.0) for pmu in observed], dtype=float)
    sev_norm = _normalize(sev)
    dist = np.asarray([distances.get((pmu, int(candidate_bus)), 1.0) for pmu in observed], dtype=float)
    weighted_distance = float(np.dot(sev, dist) / max(float(np.sum(sev)), 1e-12)) if dist.size else 1.0
    inv_distance = float(np.sum(sev / np.maximum(dist, 1e-9)) / max(float(np.sum(sev)), 1e-12)) if dist.size else 0.0
    cosines = []
    residuals = []
    for tau in (0.05, 0.10, 0.20, 0.40, 0.80):
        expected = np.asarray([math.exp(-distances.get((pmu, int(candidate_bus)), 1.0) / tau) for pmu in observed], dtype=float)
        expected = _normalize(expected)
        cosines.append(float(np.dot(sev_norm, expected)))
        residuals.append(float(np.linalg.norm(sev_norm - expected)))
    sorted_sev = np.sort(sev)[::-1]
    return {
        "candidate_is_bus": 1.0,
        "candidate_is_observed": float(candidate_bus in set(observed)),
        "candidate_is_generator_bus": float(candidate_bus in GENERATOR_BUSES),
        "candidate_degree": float(degrees.get(int(candidate_bus), 0)),
        "min_distance_to_observed": float(np.nanmin(dist)) if dist.size else 1.0,
        "mean_distance_to_observed": float(np.nanmean(dist)) if dist.size else 1.0,
        "max_distance_to_observed": float(np.nanmax(dist)) if dist.size else 1.0,
        "weighted_distance_to_severity": weighted_distance,
        "inverse_distance_score": inv_distance,
        "diffusion_cosine_max": float(np.nanmax(cosines)),
        "diffusion_cosine_mean": float(np.nanmean(cosines)),
        "diffusion_residual_min": float(np.nanmin(residuals)),
        "diffusion_residual_mean": float(np.nanmean(residuals)),
        "strongest_pmu_energy": float(sorted_sev[0]) if sorted_sev.size else 0.0,
        "severity_top1_top2_margin": float(sorted_sev[0] - sorted_sev[1]) if sorted_sev.size > 1 else (float(sorted_sev[0]) if sorted_sev.size else 0.0),
        "severity_entropy": _entropy(sev),
        "n_observed_pmus": float(len(observed)),
    }


def _bus_from_label(label: str) -> int | None:
    match = re.search(r"BUS(\d+)$", str(label))
    return int(match.group(1)) if match else None


def _resolve_generator_bus(label: str, terminal_map: dict[int, int]) -> str:
    bus = _bus_from_label(label)
    if bus is not None and bus in terminal_map:
        return f"BUS{terminal_map[bus]}"
    return label


def _build_rows(data: pd.DataFrame, distances: dict[tuple[int, int], float], degrees: dict[int, int], negatives: int, random_state: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(int(random_state))
    rows = []
    y = []
    all_buses = list(range(1, 40))
    for _, row in data.iterrows():
        true_bus = _bus_from_label(str(row["location_label"]))
        if true_bus is None:
            continue
        observed = _observed_pmus(row)
        severity = _pmu_severity(row, observed)
        scored = []
        for bus in all_buses:
            if bus == true_bus:
                continue
            scored.append((distances.get((true_bus, bus), 1.0), bus))
        scored.sort(key=lambda item: item[0])
        hard = [bus for _, bus in scored[: max(1, int(0.75 * negatives))]]
        remaining = [bus for _, bus in scored if bus not in set(hard)]
        if remaining and len(hard) < negatives:
            hard.extend(list(rng.choice(remaining, size=min(len(remaining), negatives - len(hard)), replace=False)))
        for bus in [true_bus] + hard[:negatives]:
            feat = _candidate_features(row, bus, distances, degrees, observed, severity)
            feat["sim_id"] = str(row["sim_id"])
            feat["candidate_label"] = f"BUS{bus}"
            feat["true_location"] = str(row["location_label"])
            rows.append(feat)
            y.append(int(bus == true_bus))
    return pd.DataFrame(rows), pd.Series(y, dtype=int)


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col not in {"sim_id", "candidate_label", "true_location"} and frame[col].dtype.kind in "bifc"]


def _all_bus_rows(data: pd.DataFrame, distances: dict[tuple[int, int], float], degrees: dict[int, int], id_col: str) -> pd.DataFrame:
    rows = []
    for _, row in data.iterrows():
        observed = _observed_pmus(row)
        severity = _pmu_severity(row, observed)
        true = str(row["location_label"] if "location_label" in row else row["true_location"])
        sample_id = str(row[id_col])
        for bus in range(1, 40):
            feat = _candidate_features(row, bus, distances, degrees, observed, severity)
            feat["sim_id"] = sample_id
            feat["candidate_label"] = f"BUS{bus}"
            feat["true_location"] = true
            rows.append(feat)
    return pd.DataFrame(rows)


def _rank(rows: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    data = rows[["sim_id", "candidate_label", "true_location"]].copy()
    data["score"] = np.asarray(scores, dtype=float)
    out = []
    for sim_id, group in data.groupby("sim_id", sort=False):
        ranked = group.sort_values("score", ascending=False)
        true = str(ranked["true_location"].iloc[0])
        pred = str(ranked["candidate_label"].iloc[0])
        top3 = ranked["candidate_label"].astype(str).head(3).tolist()
        out.append(
            {
                "sim_id": sim_id,
                "true_location": true,
                "pred_location": pred,
                "exact": int(true == pred),
                "top3_hit": int(true in top3),
                "electrical_distance": electrical_distance(true, pred) if true != "none" else 0.0,
                "top1_score": float(ranked["score"].iloc[0]),
                "top2_score": float(ranked["score"].iloc[1]) if len(ranked) > 1 else 0.0,
            }
        )
    return pd.DataFrame(out)


def train_and_evaluate(
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    raw_eval_dir: Path = DEFAULT_RAW_EVAL_DIR,
    topology_dir: Path = DEFAULT_TOPOLOGY_DIR,
    out_dir: Path = DEFAULT_OUT,
    random_state: int = 20260504,
    n_estimators: int = 300,
    negatives_per_positive: int = 16,
) -> PipelineResult:
    out_dir.mkdir(parents=True, exist_ok=True)
    features = _load_features(feature_dir)
    train_data = features[features["event_label"].astype(int).isin([3, 6]) & features["location_label"].astype(str).str.startswith("BUS")].reset_index(drop=True)
    distances = _distances(topology_dir)
    degrees = _degree_map(topology_dir)
    terminal_map = _generator_terminal_map(topology_dir)
    rows, y = _build_rows(train_data, distances, degrees, negatives_per_positive, random_state)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    idx = np.arange(len(rows))
    train_idx, test_idx = next(splitter.split(idx, groups=rows["sim_id"].astype(str)))
    cols = _feature_columns(rows)
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
    model.fit(rows.iloc[train_idx].reindex(columns=cols), y.iloc[train_idx])
    test_sims = set(rows.iloc[test_idx]["sim_id"].astype(str))
    sim_full = _all_bus_rows(train_data[train_data["sim_id"].astype(str).isin(test_sims)], distances, degrees, "sim_id")
    sim_scores = model.predict_proba(sim_full.reindex(columns=cols))[:, 1]
    sim_ranked = _rank(sim_full, sim_scores)
    sim_ranked.to_csv(out_dir / "sim_event3_generation_predictions.csv", index=False)

    raw_features = _load_raw_features(raw_eval_dir)
    frozen = pd.read_csv(raw_eval_dir / "raw001_frozen_submission_predictions.csv")
    guarded = apply_guardrails(frozen, topology_dir)
    raw_targets = guarded[guarded["pred_event"].astype(int).isin([3, 6])].merge(raw_features, on="chunk_name", how="left")
    if "true_location_x" in raw_targets:
        raw_targets["true_location"] = raw_targets["true_location_x"].astype(str)
    elif "true_location_y" in raw_targets:
        raw_targets["true_location"] = raw_targets["true_location_y"].astype(str)
    raw_targets["location_label"] = raw_targets["true_location"].astype(str)
    raw_full = _all_bus_rows(raw_targets, distances, degrees, "chunk_name")
    raw_scores = model.predict_proba(raw_full.reindex(columns=cols))[:, 1]
    raw_ranked = _rank(raw_full, raw_scores)
    raw_ranked.to_csv(out_dir / "raw_event3_generation_predictions.csv", index=False)

    after = guarded.copy()
    for ranked in raw_ranked.itertuples(index=False):
        margin = float(ranked.top1_score) - float(ranked.top2_score)
        # Conservative promotion: only override BUS predictions for generation
        # events when the ranker has a non-trivial margin. Transformer-line
        # guardrail corrections from P15 are left intact.
        mask = after["chunk_name"].astype(str).eq(str(ranked.sim_id))
        current = str(after.loc[mask, "pred_location"].iloc[0]) if mask.any() else ""
        before_guardrail = str(after.loc[mask, "pred_location_before_guardrail"].iloc[0]) if mask.any() else ""
        candidate = _resolve_generator_bus(str(ranked.pred_location), terminal_map)
        if current.startswith("BUS") and current == before_guardrail and margin >= 0.03:
            after.loc[mask, "pred_location"] = candidate
    after["location_exact"] = after["true_location"].astype(str).eq(after["pred_location"].astype(str))
    after["electrical_distance"] = [
        electrical_distance(str(t), str(p)) if str(t) != "none" else 0.0
        for t, p in zip(after["true_location"], after["pred_location"])
    ]
    after.to_csv(out_dir / "raw_guardrail_plus_event3_ranker_predictions.csv", index=False)
    loc_mask_before = frozen["true_location"].astype(str).ne("none")
    loc_mask_guarded = guarded["true_location"].astype(str).ne("none")
    loc_mask_after = after["true_location"].astype(str).ne("none")
    report = {
        "feature_contract": {
            "candidate_feature_columns": cols,
            "contains_specific_bus_or_line_feature_names": bool(any(re.search(r"BUS\d+|LINE\d+", col) for col in cols)),
        },
        "sim_event3_6": {
            "evaluated": int(len(sim_ranked)),
            "top1_accuracy": float(sim_ranked["exact"].mean()) if len(sim_ranked) else 0.0,
            "top3_accuracy": float(sim_ranked["top3_hit"].mean()) if len(sim_ranked) else 0.0,
            "mean_electrical_distance": float(sim_ranked["electrical_distance"].mean()) if len(sim_ranked) else 0.0,
        },
        "raw_event3_6_ranked": raw_ranked.to_dict(orient="records"),
        "raw_total": {
            "frozen_exact": float(frozen.loc[loc_mask_before, "location_exact"].astype(bool).mean()),
            "guarded_exact": float(guarded.loc[loc_mask_guarded, "location_exact"].astype(bool).mean()),
            "after_exact": float(after.loc[loc_mask_after, "location_exact"].astype(bool).mean()),
            "frozen_correct": int(frozen.loc[loc_mask_before, "location_exact"].astype(bool).sum()),
            "guarded_correct": int(guarded.loc[loc_mask_guarded, "location_exact"].astype(bool).sum()),
            "after_correct": int(after.loc[loc_mask_after, "location_exact"].astype(bool).sum()),
            "localized_support": int(loc_mask_after.sum()),
            "after_mean_electrical_distance": float(after.loc[loc_mask_after, "electrical_distance"].mean()),
        },
    }
    report_path = out_dir / "event3_generation_ranker_report.json"
    write_json(report_path, report)
    model_path = out_dir / "event3_generation_ranker.joblib"
    joblib.dump({"model": model, "feature_columns": cols}, model_path)
    result = PipelineResult(
        name="p16_train_event3_generation_ranker",
        status="completed",
        outputs={
            "model": str(model_path.resolve()),
            "report": str(report_path.resolve()),
            "raw_predictions": str((out_dir / "raw_guardrail_plus_event3_ranker_predictions.csv").resolve()),
        },
        metrics={
            "sim_event3_6_top1": report["sim_event3_6"]["top1_accuracy"],
            "sim_event3_6_top3": report["sim_event3_6"]["top3_accuracy"],
            "raw_frozen_exact": report["raw_total"]["frozen_exact"],
            "raw_guarded_exact": report["raw_total"]["guarded_exact"],
            "raw_after_exact": report["raw_total"]["after_exact"],
        },
        notes=[
            "The model uses only candidate-observation relative features; bus labels are targets/report identifiers only.",
            "P15 transformer endpoint guardrail is applied before optional generation-ranker overrides.",
        ],
    )
    write_json(out_dir / "p16_pipeline_result.json", result.to_dict())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train agnostic event3/event6 generation bus ranker.")
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--raw-eval-dir", type=Path, default=DEFAULT_RAW_EVAL_DIR)
    parser.add_argument("--topology-dir", type=Path, default=DEFAULT_TOPOLOGY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--random-state", type=int, default=20260504)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--negatives-per-positive", type=int, default=16)
    args = parser.parse_args()
    result = train_and_evaluate(args.feature_dir, args.raw_eval_dir, args.topology_dir, args.out_dir, args.random_state, args.n_estimators, args.negatives_per_positive)
    print(json.dumps(json_safe(result.to_dict()), indent=2))


if __name__ == "__main__":
    main()
