"""Event-0 artifact loading and fallback resolution for m4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_csv_if_exists(path: str | Path) -> pd.DataFrame | None:
    """Read CSV if present, otherwise return None."""
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def load_json_if_exists(path: str | Path) -> dict | None:
    """Read JSON if present, otherwise return None."""
    json_path = Path(path)
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _raw_name_family(raw_name: str) -> str:
    upper = str(raw_name).upper()
    if upper.endswith(("VA_MAG", "VB_MAG", "VC_MAG")):
        return "voltage_mag"
    if upper.endswith(("IA_MAG", "IB_MAG", "IC_MAG")):
        return "current_mag"
    if upper.endswith("FREQ"):
        return "frequency"
    if upper.endswith("ROCOF"):
        return "rocof"
    if upper.endswith(("VA_ANG", "VB_ANG", "VC_ANG")):
        return "angle_voltage"
    if upper.endswith(("IA_ANG", "IB_ANG", "IC_ANG")):
        return "angle_current"
    return "other"


def suffix_to_family(suffix: str) -> str:
    """Map raw suffix (without BUS prefix) to family label."""
    return _raw_name_family(f"BUSX_{suffix}")


def aggregate_family_defaults_from_profiles(noise_layer: Any) -> dict[str, dict]:
    """Aggregate per-family fallback profiles from event-0 m2 profiles."""
    families: dict[str, list[dict]] = {
        "voltage_mag": [],
        "current_mag": [],
        "frequency": [],
        "rocof": [],
        "angle_voltage": [],
        "angle_current": [],
    }
    profiles = getattr(noise_layer, "profiles", {}) or {}
    evt0 = profiles.get("0", {}) if isinstance(profiles, dict) else {}
    for _, bus_payload in evt0.items():
        if not isinstance(bus_payload, dict):
            continue
        for raw_name, profile_obj in bus_payload.items():
            family = _raw_name_family(str(raw_name))
            if family in families and isinstance(profile_obj, dict):
                families[family].append(profile_obj)

    out: dict[str, dict] = {}
    for family, items in families.items():
        if not items:
            out[family] = {}
            continue
        stds: list[float] = []
        outliers: list[float] = []
        rhos: list[float] = []
        meds: list[float] = []
        fitted: list[str] = []
        residual_samples: list[np.ndarray] = []
        for p in items:
            stds.append(float(p.get("std_dev_abs", p.get("std_dev_raw", 0.0))))
            outliers.append(float(p.get("outlier_mag_abs", 0.0)))
            rhos.append(float(p.get("ar1_rho", 0.0)))
            eda = p.get("eda_stats", {}) or {}
            meds.append(float(eda.get("median", eda.get("p50", 0.0))))
            fitted.append(str(p.get("fitted_distribution_type", "gaussian")))
            rs = np.asarray(p.get("residual_samples", []), dtype=float)
            rs = rs[np.isfinite(rs)]
            if len(rs):
                residual_samples.append(rs)
        out[family] = {
            "std_dev_abs": float(np.median(stds)) if stds else 0.0,
            "outlier_mag_abs": float(np.median(outliers)) if outliers else 0.0,
            "ar1_rho": float(np.median(rhos)) if rhos else 0.0,
            "fitted_distribution_type": max(set(fitted), key=fitted.count) if fitted else "gaussian",
            "eda_stats": {
                "median": float(np.median(meds)) if meds else 0.0,
                "std": float(np.median(stds)) if stds else 0.0,
            },
            "residual_samples": np.concatenate(residual_samples).tolist()[:5000] if residual_samples else [],
            "profile_source": "family_default_from_event0_pmuses",
        }
    return out


def load_event0_artifacts(
    event0_profile_path: str | Path,
    event0_current_mapping_csv: str | Path,
    event0_support_matrix_csv: str | Path,
    event0_calibration_json: str | Path,
) -> dict:
    """Load m2/m3 event-0 artifacts with parity-compatible fallbacks."""
    from m2_noise_profiling_raw import RawPMUNoiseLayer

    artifacts: dict[str, Any] = {
        "noise_layer": None,
        "family_defaults": {},
        "current_mappings": {},
        "support": {},
        "aggregate_records": {},
    }

    try:
        noise_layer = RawPMUNoiseLayer(str(event0_profile_path))
        artifacts["noise_layer"] = noise_layer
        artifacts["family_defaults"] = aggregate_family_defaults_from_profiles(noise_layer)
    except Exception:
        artifacts["noise_layer"] = None

    current_map_df = load_csv_if_exists(event0_current_mapping_csv)
    if current_map_df is not None and not current_map_df.empty and "bus_id" in current_map_df.columns:
        for _, row in current_map_df.iterrows():
            artifacts["current_mappings"][str(row["bus_id"])] = str(row.get("chosen_mapping", "bus_injection_current"))

    support_df = load_csv_if_exists(event0_support_matrix_csv)
    if support_df is not None and not support_df.empty and "signal_name" in support_df.columns:
        for _, row in support_df.iterrows():
            artifacts["support"][str(row["signal_name"])] = row.to_dict()

    cal_json = load_json_if_exists(event0_calibration_json)
    if cal_json is not None:
        for rec in cal_json.get("records", []):
            if rec.get("scope") == "aggregate":
                artifacts["aggregate_records"][(str(rec.get("bus_id")), str(rec.get("signal_key")))] = rec

    return artifacts


def calibration_record_for(artifacts: dict, bus_id: str, signal_key: str) -> dict | None:
    """Resolve aggregate calibration record for bus/signal."""
    return artifacts.get("aggregate_records", {}).get((str(bus_id), str(signal_key)))


def resolve_profile_for_bus_signal(artifacts: dict, bus_id: str, raw_suffix: str) -> tuple[dict | None, str]:
    """Resolve signal profile by exact event-0 bus profile or family default."""
    noise_layer = artifacts.get("noise_layer")
    raw_name = f"BUS{bus_id}_{raw_suffix}"
    if noise_layer is not None:
        try:
            profile, status = noise_layer.resolve_stats(str(bus_id), 0, raw_name)
            if profile is not None:
                return profile, f"exact_or_fallback:{status}"
        except Exception:
            pass
    family = _raw_name_family(raw_name)
    family_profile = artifacts.get("family_defaults", {}).get(family, {})
    if family_profile:
        return family_profile, "family_default"
    return None, "no_profile"


def profile_median(profile: dict | None) -> float:
    """Return profile median with robust fallback."""
    if not isinstance(profile, dict):
        return 0.0
    eda = profile.get("eda_stats", {}) or {}
    return float(eda.get("median", eda.get("p50", 0.0)))


def profile_std(profile: dict | None) -> float:
    """Return profile standard deviation fallback."""
    if not isinstance(profile, dict):
        return 0.0
    return float(profile.get("std_dev_abs", profile.get("std_dev_raw", 0.0)))


def event0_drift_targets_from_profile(profile: dict | None, default_family: str) -> dict:
    """Build drift/noise targets from resolved profile."""
    if not isinstance(profile, dict):
        return {
            "family": default_family,
            "center": 0.0,
            "std_dev_abs": 0.0,
            "outlier_mag_abs": 0.0,
            "ar1_rho": 0.0,
            "fitted_distribution_type": "gaussian",
            "residual_samples": [],
        }
    return {
        "family": default_family,
        "center": profile_median(profile),
        "std_dev_abs": profile_std(profile),
        "outlier_mag_abs": float(profile.get("outlier_mag_abs", 0.0)),
        "ar1_rho": float(profile.get("ar1_rho", 0.0)),
        "fitted_distribution_type": str(profile.get("fitted_distribution_type", "gaussian")),
        "residual_samples": list(profile.get("residual_samples", [])),
    }


def scaled_noise_profile(profile: dict | None, noise_scale: float = 1.0, spike_scale: float = 1.0) -> dict:
    """Scale event-0 profile noise magnitudes for fault simulation."""
    base = dict(profile or {})
    base["std_dev_abs"] = float(base.get("std_dev_abs", 0.0)) * float(noise_scale)
    base["std_dev_raw"] = float(base.get("std_dev_raw", base.get("std_dev_abs", 0.0))) * float(noise_scale)
    base["outlier_mag_abs"] = float(base.get("outlier_mag_abs", 0.0)) * float(spike_scale)
    return base
