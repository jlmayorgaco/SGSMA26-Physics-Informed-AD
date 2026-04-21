"""Canonical bus/topology naming helpers shared across pipelines."""

from __future__ import annotations

import re


def canonical_bus_name(value: str | int) -> str:
    """Normalize BUS naming across RAW/metadata/CSV inputs."""
    raw = str(value).strip().strip("'").strip('"').upper().replace(" ", "")
    if raw.isdigit():
        return f"BUS{int(raw)}"
    if raw.startswith("BUS"):
        match = re.match(r"BUS(\d+)(.*)", raw)
        if match:
            suffix = (match.group(2) or "").upper()
            return f"BUS{int(match.group(1))}{suffix}"
        return raw
    match = re.search(r"(\d+)", raw)
    if match:
        return f"BUS{int(match.group(1))}"
    return raw


def bus_token(value: str | int) -> str:
    """Return numeric bus token as plain string (e.g. BUS39 -> '39')."""
    canon = canonical_bus_name(value)
    match = re.search(r"(\d+)", canon)
    if not match:
        raise ValueError(f"Unable to extract numeric bus token from {value!r}")
    return str(int(match.group(1)))


def bus_sort_key(value: str | int) -> tuple[int, str]:
    """Sort BUS2 before BUS10 and keep suffix ordering for BUS30X1 names."""
    canon = canonical_bus_name(value)
    match = re.match(r"BUS(\d+)(.*)", canon)
    if not match:
        return (10**9, canon)
    return (int(match.group(1)), match.group(2) or "")

