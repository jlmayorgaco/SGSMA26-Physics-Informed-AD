"""Raw PMU noise layer with legacy-compatible profile resolution behavior."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class RawPMUNoiseLayer:
    """Resolver/applicator for raw PMU noise profiles."""

    def __init__(self, profile_path: str | Path):
        with Path(profile_path).open("r", encoding="utf-8") as fh:
            self.profiles = json.load(fh)
        self.missing = set()

    def resolve_stats(self, bus_id, event_type, raw_column_name):
        bus_id, event_type = str(bus_id), str(event_type)
        if event_type in self.profiles and bus_id in self.profiles[event_type]:
            sigs = self.profiles[event_type][bus_id]
            if raw_column_name in sigs:
                return sigs[raw_column_name], "exact_match"
        if "0" in self.profiles and bus_id in self.profiles["0"]:
            sigs = self.profiles["0"][bus_id]
            if raw_column_name in sigs:
                return sigs[raw_column_name], "fallback_to_0"

        key = (bus_id, event_type, raw_column_name)
        if key not in self.missing:
            print(f"[WARN] raw noise profile missing: bus={bus_id}, event={event_type}, signal={raw_column_name}")
            self.missing.add(key)
        return {
            "std_dev_abs": 1e-6,
            "std_dev_raw": 1e-6,
            "ar1_rho": 0.0,
            "ar5_rho": 0.0,
            "p_outlier": 0.0,
            "outlier_mag_abs": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "fitted_distribution_type": "gaussian",
            "signal_family": "unknown",
            "psd_summary": {},
            "fft_low_freq_summary": {},
            "quantile_summary": {},
            "residual_samples": [],
            "profile_source": "default",
        }, "default_fallback"

    def apply(self, bus_id, event_type, t, y, raw_column_name):
        stats, status = self.resolve_stats(bus_id, event_type, raw_column_name)
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n == 0:
            return y, status

        sigma = float(stats.get("std_dev_abs", stats.get("std_dev_raw", 1e-6)))
        rho = float(np.clip(stats.get("ar1_rho", 0.0), -0.999, 0.999))
        white = np.random.normal(0.0, sigma * np.sqrt(max(0.0, 1.0 - rho**2)), n)
        noise = np.zeros(n)
        noise[0] = white[0]
        for i in range(1, n):
            noise[i] = rho * noise[i - 1] + white[i]

        p = float(np.clip(stats.get("p_outlier", 0.0), 0.0, 1.0))
        mag = float(stats.get("outlier_mag_abs", 0.0))
        if p > 0 and mag > 0:
            mask = np.random.choice([0, 1], size=n, p=[1.0 - p, p])
            spikes = mask * np.random.choice([-1.0, 1.0], size=n) * np.abs(
                np.random.normal(mag, max(mag * 0.25, 1e-12), n)
            )
        else:
            spikes = 0.0
        _ = t
        return y + noise + spikes, status
