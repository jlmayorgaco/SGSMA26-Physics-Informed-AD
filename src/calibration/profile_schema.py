"""Schema helpers and constants for raw PMU noise profiles."""

from __future__ import annotations


RESIDUAL_SAMPLE_LIMIT = 2048
EXCLUDED_SIGNAL_COLUMNS = {"TIMESTAMP", "DATA_PRESENT", "Event"}
REQUIRED_PROFILE_KEYS = {"eda_stats", "noise_model", "diagnostics", "sample_count"}


def has_required_profile_keys(profile: dict) -> bool:
    """Return True when a profile contains the required top-level contract keys."""
    return REQUIRED_PROFILE_KEYS.issubset(set(profile.keys()))
