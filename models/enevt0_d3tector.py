from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DETECTOR_PATH = ROOT / "models" / "event0_detector_config.json"
DEFAULT_RAW_PATH = ROOT / "data" / "metadata" / "IEEE_39_Bus_Power_System.raw"
DEFAULT_CHUNK_ROOT = ROOT / "data" / "chunked"

SIGNAL_SUFFIXES = (
    "VA_ANG",
    "VA_MAG",
    "VB_ANG",
    "VB_MAG",
    "VC_ANG",
    "VC_MAG",
    "IA_ANG",
    "IA_MAG",
    "IB_ANG",
    "IB_MAG",
    "IC_ANG",
    "IC_MAG",
    "Freq",
    "ROCOF",
)


def _bus_csv_paths(chunk_path: Path) -> list[Path]:
    patterns = (
        "Bus*_Competition_Data_nanmask.csv",
        "Bus*_Competition_Data_sim.csv",
        "Bus*_Competition_Data.csv",
    )
    for pattern in patterns:
        paths = sorted(chunk_path.glob(pattern), key=lambda path: _bus_sort_key(path.name))
        if paths:
            return paths
    return []


def _bus_sort_key(text: str) -> tuple[int, str]:
    match = re.search(r"\d+", text)
    return (int(match.group(0)) if match else 10**9, text)


def _json_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _parse_raw_bus_bases(raw_path: Path) -> tuple[float, dict[str, dict[str, float]]]:
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    header = [part.strip() for part in lines[0].split(",")]
    s_base_mva = float(header[1])
    buses: dict[str, dict[str, float]] = {}

    for line in lines[3:]:
        stripped = line.strip()
        if stripped.startswith("0 /end bus section"):
            break
        parts = [part.strip().strip("'\"") for part in line.split(",")]
        if len(parts) < 10:
            continue
        label = parts[1].upper()
        base_kv = float(parts[2])
        v_base_phase_volts = base_kv * 1000.0 / math.sqrt(3.0)
        i_base_amps = s_base_mva * 1_000_000.0 / (math.sqrt(3.0) * base_kv * 1000.0)
        buses[label] = {
            "base_kv_ll": base_kv,
            "v_base_phase_volts": v_base_phase_volts,
            "i_base_amps": i_base_amps,
        }
    return s_base_mva, buses


def _wrap_radians(values: pd.Series) -> pd.Series:
    radians = np.deg2rad(pd.to_numeric(values, errors="coerce"))
    return pd.Series(np.arctan2(np.sin(radians), np.cos(radians)), index=values.index)


def _column_suffix(column: str, bus: str) -> str | None:
    prefix = f"{bus.upper()}_"
    if not column.upper().startswith(prefix):
        return None
    suffix = column[len(prefix) :]
    return suffix if suffix in SIGNAL_SUFFIXES else None


def _feature_name(bus: str, suffix: str, unit: str) -> str:
    if suffix.endswith("_MAG"):
        kind = "PU"
    elif suffix.endswith("_ANG"):
        kind = "RAD_WRAPPED"
    elif suffix == "Freq":
        kind = "HZ_DEV"
    elif suffix == "ROCOF":
        kind = "HZ_PER_S"
    else:
        kind = unit.upper()
    return f"{suffix}_{kind}_FILTERED_{bus.upper()}"


class Event0RangeDetector:
    """Rolling-mean range detector for event0 vs non-event0.

    The detector uses the learned ranges stored in
    data/chunked/rolling5_tuned_range_detector.json.
    """

    def __init__(
        self,
        detector_path: Path | str = DEFAULT_DETECTOR_PATH,
        raw_path: Path | str = DEFAULT_RAW_PATH,
    ) -> None:
        self.detector_path = Path(detector_path)
        self.raw_path = Path(raw_path)
        self.payload = json.loads(self.detector_path.read_text(encoding="utf-8"))
        self.s_base_mva, self.raw_buses = _parse_raw_bus_bases(self.raw_path)
        self.limits: dict[str, dict[str, Any]] = self.payload["limits"]
        detector = self.payload["detector"]
        self.rolling_window = int(detector["filter"]["window"])
        self.outside_rate_threshold = float(detector["outside_rate_threshold"])

    def _normalize(self, values: pd.Series, suffix: str, bus_base: dict[str, float]) -> tuple[pd.Series, str]:
        numeric = pd.to_numeric(values, errors="coerce")
        if suffix.endswith("_MAG") and suffix.startswith("V"):
            return numeric / bus_base["v_base_phase_volts"], "p.u."
        if suffix.endswith("_MAG") and suffix.startswith("I"):
            return numeric / bus_base["i_base_amps"], "p.u."
        if suffix.endswith("_ANG"):
            return _wrap_radians(values), "rad_wrapped"
        if suffix == "Freq":
            return numeric - 60.0, "Hz_deviation_from_60"
        if suffix == "ROCOF":
            return numeric, "Hz_per_second"
        return numeric, "normalized"

    def build_features(self, bus_frames: dict[str, pd.DataFrame], include_derivatives: bool = False) -> pd.DataFrame:
        """Build normalized rolling-mean features from BUS-label keyed frames."""
        feature_parts: list[pd.DataFrame] = []
        timestamps: pd.Series | None = None

        for bus in sorted(bus_frames, key=_bus_sort_key):
            bus_key = bus.upper()
            if bus_key not in self.raw_buses:
                raise KeyError(f"No RAW base found for {bus_key}")
            frame = bus_frames[bus]
            if timestamps is None and "TIMESTAMP" in frame.columns:
                timestamps = pd.to_numeric(frame["TIMESTAMP"], errors="coerce")

            bus_features: dict[str, pd.Series] = {}
            for column in frame.columns:
                suffix = _column_suffix(column, bus_key)
                if suffix is None:
                    continue
                normalized, unit = self._normalize(frame[column], suffix, self.raw_buses[bus_key])
                filtered = normalized.rolling(window=self.rolling_window, min_periods=1, center=False).mean()
                bus_features[_feature_name(bus_key, suffix, unit)] = filtered
            feature_parts.append(pd.DataFrame(bus_features))

        features = pd.concat(feature_parts, axis=1)
        if include_derivatives:
            if timestamps is None:
                dt = pd.Series(np.full(len(features), 1.0 / 30.0), index=features.index)
            else:
                dt = timestamps.diff().replace(0.0, np.nan)
                dt = dt.fillna(dt.median())
                if not np.isfinite(dt).all() or (dt <= 0).any():
                    dt = pd.Series(np.full(len(features), 1.0 / 30.0), index=features.index)
            derivative = features.diff().div(dt.to_numpy(), axis=0).fillna(0.0)
            derivative.columns = [f"{column}_D1_PER_S" for column in features.columns]
            features = pd.concat([features, derivative], axis=1)
        return features

    def predict_features(self, features: pd.DataFrame) -> dict[str, Any]:
        """Predict using already built base features.

        Enhanced derivative columns are ignored by this range detector because
        the saved learned limits are for the base 112 filtered features.
        """
        missing = [feature for feature in self.limits if feature not in features.columns]
        if missing:
            raise KeyError(f"Missing required features: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        outside_masks: dict[str, pd.Series] = {}
        feature_outside_rates: dict[str, float] = {}
        for feature, limit in self.limits.items():
            lower = float(limit["lower"])
            upper = float(limit["upper"])
            values = pd.to_numeric(features[feature], errors="coerce")
            mask = (values < lower) | (values > upper) | values.isna()
            outside_masks[feature] = mask
            feature_outside_rates[feature] = float(mask.mean())

        outside_by_feature = pd.DataFrame(outside_masks, index=features.index)
        sample_outside = outside_by_feature.any(axis=1)
        outside_rate = float(sample_outside.mean())
        pred_non0 = outside_rate >= self.outside_rate_threshold
        top_features = sorted(feature_outside_rates.items(), key=lambda item: item[1], reverse=True)
        top_features = [
            {"feature": feature, "outside_rate": _json_float(rate)}
            for feature, rate in top_features
            if rate > 0
        ][:20]

        return {
            "pred_label": "non0" if pred_non0 else "event0",
            "pred_non0": bool(pred_non0),
            "outside_rate": _json_float(outside_rate),
            "outside_rate_threshold": _json_float(self.outside_rate_threshold),
            "outside_count": int(sample_outside.sum()),
            "n_samples": int(len(sample_outside)),
            "top_outside_features": top_features,
        }

    def predict_frames(self, bus_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
        features = self.build_features(bus_frames=bus_frames, include_derivatives=False)
        return self.predict_features(features)

    def predict_chunk(self, chunk_dir: Path | str) -> dict[str, Any]:
        chunk_path = Path(chunk_dir)
        bus_frames: dict[str, pd.DataFrame] = {}
        for csv_path in _bus_csv_paths(chunk_path):
            bus = csv_path.name.split("_", 1)[0].upper()
            bus_frames[bus] = pd.read_csv(csv_path)
        if not bus_frames:
            raise FileNotFoundError(f"No Bus*_Competition_Data*.csv files found in {chunk_path}")
        result = self.predict_frames(bus_frames)
        result["chunk_name"] = chunk_path.name
        result["chunk_dir"] = str(chunk_path.resolve())
        return result


def predict_chunk_root(chunk_root: Path, detector: Event0RangeDetector) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for chunk_dir in sorted([path for path in chunk_root.iterdir() if path.is_dir() and "_event" in path.name], key=lambda path: _bus_sort_key(path.name)):
        result = detector.predict_chunk(chunk_dir)
        rows.append(
            {
                "chunk_name": result["chunk_name"],
                "pred_label": result["pred_label"],
                "pred_non0": result["pred_non0"],
                "outside_rate": result["outside_rate"],
                "outside_rate_threshold": result["outside_rate_threshold"],
                "outside_count": result["outside_count"],
                "n_samples": result["n_samples"],
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the saved event0/non0 range detector.")
    parser.add_argument("--chunk-dir", type=Path, default=None)
    parser.add_argument("--chunk-root", type=Path, default=None)
    parser.add_argument("--detector-path", type=Path, default=DEFAULT_DETECTOR_PATH)
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = Event0RangeDetector(detector_path=args.detector_path, raw_path=args.raw_path)
    if args.chunk_dir is not None:
        print(json.dumps(detector.predict_chunk(args.chunk_dir), indent=2))
        return

    chunk_root = args.chunk_root or DEFAULT_CHUNK_ROOT
    predictions = predict_chunk_root(chunk_root=chunk_root, detector=detector)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output_csv, index=False)
        print(f"Wrote {args.output_csv.resolve()}")
    else:
        print(predictions.to_string(index=False))


if __name__ == "__main__":
    main()
