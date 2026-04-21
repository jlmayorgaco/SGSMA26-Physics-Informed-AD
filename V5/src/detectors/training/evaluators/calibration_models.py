from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.detectors.training.evaluators.calibration_evaluator import CalibrationEvaluator


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


class ProbabilityCalibrator:
    method: str = "identity"

    def transform(self, p: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)


@dataclass(slots=True)
class IdentityCalibrator(ProbabilityCalibrator):
    method: str = "identity"


@dataclass(slots=True)
class TemperatureCalibrator(ProbabilityCalibrator):
    temperature: float = 1.0
    method: str = "temperature"

    def transform(self, p: np.ndarray) -> np.ndarray:
        prob = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
        logits = np.log(prob / (1.0 - prob))
        t = max(float(self.temperature), 1e-6)
        return np.clip(_sigmoid(logits / t), 0.0, 1.0)


@dataclass(slots=True)
class PlattCalibrator(ProbabilityCalibrator):
    a: float = 1.0
    b: float = 0.0
    method: str = "platt"

    def transform(self, p: np.ndarray) -> np.ndarray:
        prob = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
        logits = np.log(prob / (1.0 - prob))
        return np.clip(_sigmoid(self.a * logits + self.b), 0.0, 1.0)


@dataclass(slots=True)
class IsotonicBinCalibrator(ProbabilityCalibrator):
    bin_edges: np.ndarray
    bin_values: np.ndarray
    method: str = "isotonic_bin"

    def transform(self, p: np.ndarray) -> np.ndarray:
        prob = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
        idx = np.searchsorted(self.bin_edges, prob, side="right") - 1
        idx = np.clip(idx, 0, len(self.bin_values) - 1)
        return np.clip(self.bin_values[idx], 0.0, 1.0)


@dataclass(slots=True)
class CalibrationSelection:
    selected_name: str
    calibrator: ProbabilityCalibrator
    comparison: pd.DataFrame


class CalibratorFactory:
    @staticmethod
    def fit_temperature(y_true: np.ndarray, y_prob: np.ndarray) -> TemperatureCalibrator:
        yt = np.asarray(y_true, dtype=int)
        yp = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
        logits = np.log(yp / (1.0 - yp))
        best_t = 1.0
        best_nll = 1e18
        for t in np.linspace(0.5, 3.0, 51):
            pred = _sigmoid(logits / t)
            nll = -float(np.mean(yt * np.log(pred + 1e-9) + (1 - yt) * np.log(1.0 - pred + 1e-9)))
            if nll < best_nll:
                best_nll = nll
                best_t = float(t)
        return TemperatureCalibrator(temperature=best_t)

    @staticmethod
    def fit_platt(y_true: np.ndarray, y_prob: np.ndarray) -> PlattCalibrator:
        yt = np.asarray(y_true, dtype=float)
        yp = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
        x = np.log(yp / (1.0 - yp))
        a, b = 1.0, 0.0
        lr = 0.05
        for _ in range(400):
            pred = _sigmoid(a * x + b)
            err = pred - yt
            grad_a = float(np.mean(err * x)) + 1e-4 * a
            grad_b = float(np.mean(err))
            a -= lr * grad_a
            b -= lr * grad_b
        return PlattCalibrator(a=float(a), b=float(b))

    @staticmethod
    def fit_isotonic_bin(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 12) -> IsotonicBinCalibrator:
        yt = np.asarray(y_true, dtype=float)
        yp = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
        edges = np.linspace(0.0, 1.0, int(bins) + 1)
        values = []
        for i in range(len(edges) - 1):
            lo = edges[i]
            hi = edges[i + 1]
            if i == len(edges) - 2:
                mask = (yp >= lo) & (yp <= hi)
            else:
                mask = (yp >= lo) & (yp < hi)
            if not np.any(mask):
                values.append(np.nan)
            else:
                values.append(float(yt[mask].mean()))
        v = np.array(values, dtype=float)
        if np.isnan(v).all():
            v = np.full_like(v, yt.mean() if len(yt) else 0.5)
        # fill NaN gaps then enforce monotonicity with cumulative max
        series = pd.Series(v).interpolate(limit_direction="both").fillna(float(yt.mean() if len(yt) else 0.5))
        mono = np.maximum.accumulate(series.to_numpy(dtype=float))
        mono = np.clip(mono, 0.0, 1.0)
        return IsotonicBinCalibrator(bin_edges=edges, bin_values=mono)

    @staticmethod
    def select_best(y_true: np.ndarray, y_prob: np.ndarray) -> CalibrationSelection:
        candidates: list[tuple[str, ProbabilityCalibrator]] = [
            ("identity", IdentityCalibrator()),
            ("temperature", CalibratorFactory.fit_temperature(y_true, y_prob)),
            ("platt", CalibratorFactory.fit_platt(y_true, y_prob)),
            ("isotonic_bin", CalibratorFactory.fit_isotonic_bin(y_true, y_prob)),
        ]
        evaluator = CalibrationEvaluator(n_bins=10)
        rows = []
        best_name = "identity"
        best_model: ProbabilityCalibrator = candidates[0][1]
        best_obj = 1e18
        for name, model in candidates:
            calibrated = model.transform(y_prob)
            cal = evaluator.evaluate(y_true, calibrated)
            obj = 0.7 * cal.ece + 0.3 * cal.brier
            rows.append({"method": name, "ece": float(cal.ece), "brier": float(cal.brier), "objective": float(obj)})
            if obj < best_obj:
                best_obj = obj
                best_name = name
                best_model = model
        frame = pd.DataFrame(rows).sort_values(["objective", "ece", "brier"], ascending=True).reset_index(drop=True)
        return CalibrationSelection(selected_name=best_name, calibrator=best_model, comparison=frame)


def save_calibrator(path: Path, calibrator: ProbabilityCalibrator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(calibrator, fh)


def load_calibrator(path: Path) -> ProbabilityCalibrator:
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, ProbabilityCalibrator):
        raise TypeError(f"unexpected calibrator type at {path}: {type(obj)!r}")
    return obj
