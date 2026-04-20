from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from src.calibration.raw_noise_layer import RawPMUNoiseLayer


def _workspace_test_file(prefix: str) -> Path:
    out_dir = Path("output") / "_test_runs" / f"{prefix}_{uuid4().hex[:8]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "profiles.json"


def _profile_json(path: Path) -> None:
    payload = {
        "0": {
            "10": {
                "BUS10_VA_MAG": {
                    "std_dev_abs": 0.01,
                    "std_dev_raw": 0.01,
                    "ar1_rho": 0.1,
                    "ar5_rho": 0.0,
                    "p_outlier": 0.0,
                    "outlier_mag_abs": 0.0,
                }
            }
        },
        "5": {
            "10": {
                "BUS10_VA_MAG": {
                    "std_dev_abs": 0.02,
                    "std_dev_raw": 0.02,
                    "ar1_rho": 0.2,
                    "ar5_rho": 0.0,
                    "p_outlier": 0.0,
                    "outlier_mag_abs": 0.0,
                }
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_stats_exact_match() -> None:
    p = _workspace_test_file("m2_layer_unit")
    _profile_json(p)
    layer = RawPMUNoiseLayer(p)
    _, status = layer.resolve_stats("10", "5", "BUS10_VA_MAG")
    assert status == "exact_match"


def test_resolve_stats_fallback_to_event0() -> None:
    p = _workspace_test_file("m2_layer_unit")
    _profile_json(p)
    layer = RawPMUNoiseLayer(p)
    _, status = layer.resolve_stats("10", "7", "BUS10_VA_MAG")
    assert status == "fallback_to_0"


def test_resolve_stats_default_fallback() -> None:
    p = _workspace_test_file("m2_layer_unit")
    _profile_json(p)
    layer = RawPMUNoiseLayer(p)
    _, status = layer.resolve_stats("99", "7", "BUS99_X")
    assert status == "default_fallback"


def test_apply_empty_signal_returns_empty() -> None:
    p = _workspace_test_file("m2_layer_unit")
    _profile_json(p)
    layer = RawPMUNoiseLayer(p)
    y, status = layer.apply("10", "5", np.array([]), np.array([]), "BUS10_VA_MAG")
    assert len(y) == 0
    assert isinstance(status, str)


def test_apply_returns_same_length() -> None:
    p = _workspace_test_file("m2_layer_unit")
    _profile_json(p)
    layer = RawPMUNoiseLayer(p)
    y = np.ones(100)
    out, _ = layer.apply("10", "5", np.arange(100), y, "BUS10_VA_MAG")
    assert len(out) == len(y)


def test_apply_returns_status_string() -> None:
    p = _workspace_test_file("m2_layer_unit")
    _profile_json(p)
    layer = RawPMUNoiseLayer(p)
    _, status = layer.apply("10", "5", np.arange(5), np.ones(5), "BUS10_VA_MAG")
    assert isinstance(status, str)
