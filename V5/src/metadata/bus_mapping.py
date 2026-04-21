"""Bus label normalization and alignment helpers."""

from __future__ import annotations

import re


def canonicalize_bus_label(label: str) -> str:
    """Normalize bus labels (BUS1, BUS30x1, etc.) into a canonical uppercase form."""
    raw = str(label).strip().strip("'").strip('"').upper().replace(" ", "")
    if not raw.startswith("BUS"):
        m = re.search(r"(\d+)", raw)
        if m:
            return f"BUS{int(m.group(1))}"
        return raw
    m = re.match(r"BUS(\d+)(.*)", raw)
    if not m:
        return raw
    num = int(m.group(1))
    tail = m.group(2)
    return f"BUS{num}{tail}"


def numeric_bus_sort_key(label: str) -> tuple[int, str]:
    """Sort BUS2 before BUS10 and preserve BUS30x1 suffix order."""
    canon = canonicalize_bus_label(label)
    m = re.match(r"BUS(\d+)(.*)", canon)
    if not m:
        return (10**9, canon)
    return (int(m.group(1)), m.group(2) or "")


def build_bus_alignment(pmu_meta: dict, raw_meta: dict) -> dict:
    """Build alignment dictionary between PMU metadata and RAW metadata bus sets."""
    pmu_buses = {canonicalize_bus_label(b["bus_label_canonical"]) for b in pmu_meta.get("buses", [])}
    raw_buses = {canonicalize_bus_label(b["bus_label_canonical"]) for b in raw_meta.get("buses", [])}
    common = sorted(pmu_buses.intersection(raw_buses), key=numeric_bus_sort_key)
    return {
        "pmu_bus_set": sorted(pmu_buses, key=numeric_bus_sort_key),
        "raw_bus_set": sorted(raw_buses, key=numeric_bus_sort_key),
        "common_bus_set": common,
        "pmu_only_buses": sorted(pmu_buses - raw_buses, key=numeric_bus_sort_key),
        "raw_only_buses": sorted(raw_buses - pmu_buses, key=numeric_bus_sort_key),
        "aligned_ok": pmu_buses == raw_buses,
    }
