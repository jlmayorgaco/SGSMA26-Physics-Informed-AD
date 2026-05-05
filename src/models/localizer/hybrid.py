from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from src.features.pmu_discovery import infer_pmu_buses_from_columns


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TOPOLOGY_DIST = ROOT / "data" / "topology" / "ieee39" / "zbus_effective_distance_full.csv"
DEFAULT_BRANCHES = ROOT / "data" / "topology" / "ieee39" / "branches_physical.csv"
OBSERVED_PMUS = (2, 5, 6, 10, 19, 22, 29, 39)


def _location_type(label: str) -> str:
    value = str(label)
    if value.startswith("BUS"):
        return "BUS"
    if value.startswith("LINE"):
        return "LINE"
    if value.startswith("PMU"):
        return "PMU"
    return "NONE"


def _label_bus(label: str) -> int | None:
    match = re.search(r"(?:BUS|PMU)(\d+)$", str(label))
    return int(match.group(1)) if match else None


def _line_endpoints(label: str) -> tuple[int, int] | None:
    match = re.search(r"LINE(\d+)-(\d+)$", str(label))
    if not match:
        return None
    a, b = int(match.group(1)), int(match.group(2))
    return min(a, b), max(a, b)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _normalize_scores(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(list(values.values()), dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {key: 0.0 for key in values}
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if hi - lo < 1e-12:
        return {key: 0.5 for key in values}
    return {key: float((_finite_float(value) - lo) / (hi - lo)) for key, value in values.items()}


class TypedLocalizer:
    """Top-K wrapper around the existing typed sklearn localizers."""

    def __init__(self, localizers: dict[str, Any]) -> None:
        self.localizers = dict(localizers)

    def _route(self, pred_event: int, physical_event: int) -> tuple[str, list[str]]:
        if int(pred_event) == 0:
            return "NONE", []
        if int(pred_event) in {5, 7}:
            return "PMU", ["PMU"]
        loc_type = "LINE" if int(pred_event) == 2 or int(physical_event) == 2 else "BUS"
        keys: list[str] = []
        if loc_type == "LINE" and int(pred_event) == 2:
            keys.append("LINE:event2")
        if int(physical_event) != 0:
            keys.append(f"{loc_type}:event{int(physical_event)}")
        keys.append(loc_type)
        return loc_type, keys

    @staticmethod
    def _model_proba(model: Any, x_row: pd.DataFrame) -> list[tuple[str, float]]:
        if hasattr(model, "predict_proba"):
            proba = np.asarray(model.predict_proba(x_row)[0], dtype=float)
            estimator = getattr(model, "named_steps", {}).get("model") if hasattr(model, "named_steps") else model
            classes = list(getattr(estimator, "classes_", []))
            if len(classes) == len(proba):
                return [(str(label), _finite_float(score)) for label, score in zip(classes, proba)]
        return [(str(model.predict(x_row)[0]), 1.0)]

    def predict_topk(
        self,
        x: pd.DataFrame,
        pred_event: np.ndarray,
        physical_event: np.ndarray,
        k: int = 3,
    ) -> list[list[dict[str, Any]]]:
        rows: list[list[dict[str, Any]]] = []
        for idx in range(len(x)):
            loc_type, keys = self._route(int(pred_event[idx]), int(physical_event[idx]))
            if loc_type == "NONE":
                rows.append([{"candidate": "none", "location_type": "NONE", "ml_score": 1.0}])
                continue
            scores: dict[str, float] = {}
            x_row = x.iloc[[idx]]
            for rank, key in enumerate(keys):
                model = self.localizers.get(key)
                if model is None:
                    continue
                weight = 1.0 / float(rank + 1)
                for candidate, score in self._model_proba(model, x_row):
                    scores[candidate] = max(scores.get(candidate, 0.0), weight * _finite_float(score))
                if scores:
                    break
            ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: max(1, int(k))]
            rows.append(
                [
                    {"candidate": candidate, "location_type": _location_type(candidate), "ml_score": float(score)}
                    for candidate, score in ordered
                ]
            )
        return rows

    def predict(self, x: pd.DataFrame, pred_event: np.ndarray, physical_event: np.ndarray) -> np.ndarray:
        topk = self.predict_topk(x, pred_event, physical_event, k=1)
        return np.asarray([items[0]["candidate"] if items else "none" for items in topk], dtype=object)


class TopologyResidualRanker:
    """Physics-inspired candidate scorer that uses topology and feature signatures only."""

    def __init__(
        self,
        topology_csv: Path = DEFAULT_TOPOLOGY_DIST,
        branches_csv: Path = DEFAULT_BRANCHES,
        observed_pmus: tuple[int, ...] | None = None,
    ) -> None:
        self.topology_csv = Path(topology_csv)
        self.branches_csv = Path(branches_csv)
        self.observed_pmus = tuple(int(bus) for bus in observed_pmus) if observed_pmus is not None else tuple()
        self.distances = self._load_distances(self.topology_csv)
        self.lines = self._load_lines(self.branches_csv)

    def _observed_pmus_for_row(self, row: pd.Series) -> tuple[int, ...]:
        inferred = infer_pmu_buses_from_columns(row.index)
        return inferred or self.observed_pmus or OBSERVED_PMUS

    @staticmethod
    def _load_distances(path: Path) -> dict[tuple[int, int], float]:
        if not path.exists():
            return {}
        data = pd.read_csv(path)
        return {
            (int(row.from_bus), int(row.to_bus)): max(float(row.z_eff_abs), 1e-9)
            for row in data.itertuples(index=False)
        }

    @staticmethod
    def _load_lines(path: Path) -> dict[str, tuple[int, int]]:
        if not path.exists():
            return {}
        data = pd.read_csv(path)
        out: dict[str, tuple[int, int]] = {}
        for row in data.itertuples(index=False):
            a, b = min(int(row.from_bus), int(row.to_bus)), max(int(row.from_bus), int(row.to_bus))
            out[f"LINE{a}-{b}"] = (a, b)
        return out

    def _distance(self, a: int, b: int) -> float:
        return self.distances.get((int(a), int(b)), self.distances.get((int(b), int(a)), 1.0))

    def _pmu_intensity(self, row: pd.Series, pmu: int) -> float:
        terms = [
            f"BUS{pmu}__BUS{pmu}_VA_MAG__full__max_abs",
            f"BUS{pmu}__BUS{pmu}_VB_MAG__full__max_abs",
            f"BUS{pmu}__BUS{pmu}_VC_MAG__full__max_abs",
            f"BUS{pmu}__BUS{pmu}_IA_MAG__full__max_abs",
            f"BUS{pmu}__BUS{pmu}_IB_MAG__full__max_abs",
            f"BUS{pmu}__BUS{pmu}_IC_MAG__full__max_abs",
            f"BUS{pmu}__pmu_angle_spread__max",
            f"META__BUS{pmu}__nan_fraction_max",
        ]
        values = [_finite_float(row.get(term, 0.0)) for term in terms]
        return float(np.nanmax(np.abs(values))) if values else 0.0

    def _observed_signature(self, row: pd.Series) -> dict[int, float]:
        observed_pmus = self._observed_pmus_for_row(row)
        raw = {pmu: self._pmu_intensity(row, pmu) for pmu in observed_pmus}
        normalized = _normalize_scores({str(key): value for key, value in raw.items()})
        return {int(key): value for key, value in normalized.items()}

    def _expected_signature(self, candidate: str, observed_pmus: tuple[int, ...]) -> dict[int, float]:
        loc_type = _location_type(candidate)
        buses: tuple[int, ...]
        if loc_type in {"BUS", "PMU"}:
            bus = _label_bus(candidate)
            buses = (bus,) if bus is not None else tuple()
        elif loc_type == "LINE":
            buses = self.lines.get(candidate) or _line_endpoints(candidate) or tuple()
        else:
            buses = tuple()
        scores: dict[str, float] = {}
        for pmu in observed_pmus:
            if not buses:
                scores[str(pmu)] = 0.0
                continue
            distance = min(self._distance(pmu, bus) for bus in buses)
            scores[str(pmu)] = 1.0 / max(distance, 1e-6)
        normalized = _normalize_scores(scores)
        return {int(key): value for key, value in normalized.items()}

    @staticmethod
    def _cosine(a: dict[int, float], b: dict[int, float]) -> float:
        keys = sorted(set(a) | set(b))
        if not keys:
            return 0.0
        av = np.asarray([a.get(key, 0.0) for key in keys], dtype=float)
        bv = np.asarray([b.get(key, 0.0) for key in keys], dtype=float)
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        return float(np.dot(av, bv) / denom) if denom > 1e-12 else 0.0

    def score_candidates(self, row: pd.Series, candidates: list[str]) -> dict[str, dict[str, float]]:
        observed = {int(key): value for key, value in self._observed_signature(row).items()}
        observed_pmus = tuple(sorted(observed)) or self._observed_pmus_for_row(row)
        out: dict[str, dict[str, float]] = {}
        for candidate in candidates:
            expected = self._expected_signature(candidate, observed_pmus)
            physics = self._cosine(observed, expected)
            endpoints = self.lines.get(candidate) or _line_endpoints(candidate)
            if endpoints:
                topology = max(expected.values()) if expected else 0.0
            else:
                topology = expected.get(_label_bus(candidate) or -1, max(expected.values()) if expected else 0.0)
            out[candidate] = {"physics_score": float(physics), "topology_score": float(topology)}
        return out


class HybridLocalizer:
    """ML Top-K localizer with a topology/residual re-ranker."""

    def __init__(
        self,
        typed_localizer: TypedLocalizer,
        ranker: TopologyResidualRanker,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.typed_localizer = typed_localizer
        self.ranker = ranker
        self.weights = weights or {"ml_score": 0.72, "physics_score": 0.20, "topology_score": 0.08}

    @classmethod
    def from_localizers(
        cls,
        localizers: dict[str, Any],
        topology_csv: Path = DEFAULT_TOPOLOGY_DIST,
        branches_csv: Path = DEFAULT_BRANCHES,
        weights: dict[str, float] | None = None,
    ) -> "HybridLocalizer":
        return cls(TypedLocalizer(localizers), TopologyResidualRanker(topology_csv, branches_csv), weights=weights)

    def predict_topk(
        self,
        x: pd.DataFrame,
        pred_event: np.ndarray,
        physical_event: np.ndarray,
        k: int = 3,
    ) -> list[list[dict[str, Any]]]:
        ml_topk = self.typed_localizer.predict_topk(x, pred_event, physical_event, k=max(5, int(k)))
        rows: list[list[dict[str, Any]]] = []
        for idx, items in enumerate(ml_topk):
            if int(pred_event[idx]) in {0, 5, 7}:
                rows.append(items[: max(1, int(k))])
                continue
            candidates = [str(item["candidate"]) for item in items]
            physics = self.ranker.score_candidates(x.iloc[idx], candidates)
            reranked: list[dict[str, Any]] = []
            for item in items:
                candidate = str(item["candidate"])
                merged = dict(item)
                merged.update(physics.get(candidate, {"physics_score": 0.0, "topology_score": 0.0}))
                merged["final_score"] = float(
                    self.weights.get("ml_score", 0.0) * _finite_float(merged.get("ml_score"))
                    + self.weights.get("physics_score", 0.0) * _finite_float(merged.get("physics_score"))
                    + self.weights.get("topology_score", 0.0) * _finite_float(merged.get("topology_score"))
                )
                reranked.append(merged)
            rows.append(sorted(reranked, key=lambda item: item.get("final_score", item.get("ml_score", 0.0)), reverse=True)[: max(1, int(k))])
        return rows

    def predict(self, x: pd.DataFrame, pred_event: np.ndarray, physical_event: np.ndarray) -> np.ndarray:
        topk = self.predict_topk(x, pred_event, physical_event, k=1)
        return np.asarray([items[0]["candidate"] if items else "none" for items in topk], dtype=object)


def electrical_distance(true_label: str, pred_label: str, ranker: TopologyResidualRanker | None = None) -> float:
    ranker = ranker or TopologyResidualRanker()
    true_type = _location_type(true_label)
    pred_type = _location_type(pred_label)
    if true_label == pred_label:
        return 0.0
    true_buses = _line_endpoints(true_label) if true_type == "LINE" else ((_label_bus(true_label),) if _label_bus(true_label) else tuple())
    pred_buses = _line_endpoints(pred_label) if pred_type == "LINE" else ((_label_bus(pred_label),) if _label_bus(pred_label) else tuple())
    if not true_buses or not pred_buses:
        return 1.0
    return float(min(ranker._distance(a, b) for a in true_buses for b in pred_buses))


def location_ranking_report(
    labels: pd.DataFrame,
    topk: list[list[dict[str, Any]]],
    ranker: TopologyResidualRanker | None = None,
) -> dict[str, Any]:
    ranker = ranker or TopologyResidualRanker()
    mask = labels["location_label"].ne("none").to_numpy(dtype=bool)
    if not mask.any():
        return {"evaluated": 0, "exact_accuracy": 0.0, "top3_accuracy": 0.0, "mean_electrical_distance": 0.0}
    true = labels.loc[mask, "location_label"].reset_index(drop=True).astype(str)
    selected_topk = [items for keep, items in zip(mask, topk) if keep]
    pred = pd.Series([items[0]["candidate"] if items else "none" for items in selected_topk], dtype=object)
    top3_hit = [
        str(target) in [str(item["candidate"]) for item in items[:3]]
        for target, items in zip(true.tolist(), selected_topk)
    ]
    distances = [electrical_distance(str(target), str(candidate), ranker) for target, candidate in zip(true.tolist(), pred.tolist())]
    true_type = true.map(_location_type)
    pred_type = pred.map(_location_type)
    return {
        "evaluated": int(mask.sum()),
        "exact_accuracy": float((true.to_numpy() == pred.to_numpy()).mean()),
        "top3_accuracy": float(np.mean(top3_hit)) if top3_hit else 0.0,
        "mean_electrical_distance": float(np.mean(distances)) if distances else 0.0,
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "type_labels": ["BUS", "LINE", "PMU"],
        "type_confusion_matrix": confusion_matrix(true_type, pred_type, labels=["BUS", "LINE", "PMU"]).tolist(),
    }
