from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.simulation.event0_artifacts import (
    aggregate_family_defaults_from_profiles,
    load_csv_if_exists,
    load_json_if_exists,
    resolve_profile_for_bus_signal,
)


class _FakeNoiseLayer:
    def __init__(self, profiles):
        self.profiles = profiles

    def resolve_stats(self, bus_id, event_type, raw_column_name):
        _ = (bus_id, event_type, raw_column_name)
        return {"eda_stats": {"median": 1.0}, "std_dev_abs": 0.1}, "exact_match"


def _clean_dir(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink()
        else:
            p.rmdir()
    path.rmdir()


def test_load_csv_if_exists() -> None:
    root = Path("tests/fixtures/_tmp_m4_event0_artifacts_csv")
    root.mkdir(parents=True, exist_ok=True)
    p = root / "x.csv"
    try:
        pd.DataFrame({"a": [1]}).to_csv(p, index=False)
        out = load_csv_if_exists(p)
        assert out is not None
        assert list(out.columns) == ["a"]
    finally:
        _clean_dir(root)


def test_load_json_if_exists() -> None:
    root = Path("tests/fixtures/_tmp_m4_event0_artifacts_json")
    root.mkdir(parents=True, exist_ok=True)
    p = root / "x.json"
    try:
        p.write_text(json.dumps({"k": 1}), encoding="utf-8")
        out = load_json_if_exists(p)
        assert out == {"k": 1}
    finally:
        _clean_dir(root)


def test_aggregate_family_defaults_from_profiles_returns_families() -> None:
    profiles = {"0": {"10": {"BUS10_VA_MAG": {"std_dev_abs": 0.2, "eda_stats": {"median": 100.0}}}}}
    fake = _FakeNoiseLayer(profiles)
    out = aggregate_family_defaults_from_profiles(fake)
    assert "voltage_mag" in out


def test_resolve_profile_for_bus_signal_exact_match() -> None:
    artifacts = {"noise_layer": _FakeNoiseLayer({}), "family_defaults": {}}
    profile, status = resolve_profile_for_bus_signal(artifacts, "10", "VA_MAG")
    assert profile is not None
    assert "exact_or_fallback" in status


def test_resolve_profile_for_bus_signal_family_fallback() -> None:
    artifacts = {"noise_layer": None, "family_defaults": {"voltage_mag": {"eda_stats": {"median": 1.0}}}}
    profile, status = resolve_profile_for_bus_signal(artifacts, "10", "VA_MAG")
    assert profile is not None
    assert status == "family_default"


def test_resolve_profile_for_bus_signal_no_profile() -> None:
    artifacts = {"noise_layer": None, "family_defaults": {}}
    profile, status = resolve_profile_for_bus_signal(artifacts, "10", "VA_MAG")
    assert profile is None
    assert status == "no_profile"
