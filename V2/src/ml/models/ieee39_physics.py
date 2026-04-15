"""IEEE-39 topology and approximate Ybus helpers used by physics-aware models."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np


ALL_BUSES = list(range(1, 40))
PMU_BUSES = [2, 5, 6, 10, 19, 22, 29, 39]
IEEE39_BRANCHES: list[tuple[int, int]] = [
    (1, 2),
    (1, 39),
    (2, 3),
    (2, 25),
    (2, 30),
    (3, 4),
    (3, 18),
    (4, 5),
    (4, 14),
    (5, 6),
    (5, 8),
    (6, 7),
    (6, 11),
    (6, 31),
    (7, 8),
    (8, 9),
    (9, 39),
    (10, 11),
    (10, 13),
    (10, 32),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (16, 19),
    (16, 21),
    (16, 24),
    (17, 18),
    (17, 27),
    (19, 20),
    (19, 33),
    (19, 39),
    (20, 34),
    (21, 22),
    (22, 23),
    (22, 35),
    (23, 24),
    (23, 36),
    (25, 26),
    (25, 37),
    (26, 27),
    (26, 28),
    (26, 29),
    (28, 29),
    (29, 38),
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def bus_index(bus: int) -> int:
    return int(bus) - 1


@lru_cache(maxsize=1)
def branch_reactance() -> dict[tuple[int, int], float]:
    """Read branch X from the IEEE-39 RAW file when available.

    The parser is intentionally small and tolerant. If the RAW file is absent
    or a branch cannot be parsed, the caller receives a conservative topology
    default for that branch.
    """

    raw_path = project_root() / "data" / "metadata" / "IEEE 39 Bus Power System.raw"
    if not raw_path.exists():
        return {}
    values: dict[tuple[int, int], float] = {}
    for line in raw_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [part.strip().strip("'") for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            left = int(parts[0])
            right = int(parts[1])
            x = abs(float(parts[4]))
        except ValueError:
            continue
        edge = tuple(sorted((left, right)))
        if 1 <= edge[0] <= 39 and 1 <= edge[1] <= 39 and x > 1e-6:
            values[edge] = x
    return values


@lru_cache(maxsize=1)
def adjacency_matrix() -> np.ndarray:
    adjacency = np.zeros((39, 39), dtype=float)
    reactance = branch_reactance()
    for left, right in IEEE39_BRANCHES:
        edge = tuple(sorted((left, right)))
        weight = 1.0 / max(float(reactance.get(edge, 0.10)), 1e-4)
        i = bus_index(left)
        j = bus_index(right)
        adjacency[i, j] = weight
        adjacency[j, i] = weight
    return adjacency


@lru_cache(maxsize=1)
def normalized_adjacency() -> np.ndarray:
    adjacency = adjacency_matrix()
    with_self = adjacency + np.eye(39, dtype=float)
    degree = np.sum(with_self, axis=1)
    degree[degree <= 0.0] = 1.0
    inv_sqrt = np.diag(1.0 / np.sqrt(degree))
    return inv_sqrt @ with_self @ inv_sqrt


@lru_cache(maxsize=1)
def ybus_matrix() -> np.ndarray:
    """Build a simple complex Ybus from branch reactance."""

    ybus = np.zeros((39, 39), dtype=complex)
    reactance = branch_reactance()
    for left, right in IEEE39_BRANCHES:
        edge = tuple(sorted((left, right)))
        x = max(float(reactance.get(edge, 0.10)), 1e-4)
        admittance = 1.0 / complex(0.0, x)
        i = bus_index(left)
        j = bus_index(right)
        ybus[i, i] += admittance
        ybus[j, j] += admittance
        ybus[i, j] -= admittance
        ybus[j, i] -= admittance
    return ybus


@lru_cache(maxsize=1)
def electrical_distance_matrix() -> np.ndarray:
    """Approximate weighted shortest-path distance over the IEEE-39 graph."""

    distances = np.full((39, 39), np.inf, dtype=float)
    np.fill_diagonal(distances, 0.0)
    reactance = branch_reactance()
    for left, right in IEEE39_BRANCHES:
        edge = tuple(sorted((left, right)))
        weight = max(float(reactance.get(edge, 0.10)), 1e-4)
        i = bus_index(left)
        j = bus_index(right)
        distances[i, j] = min(distances[i, j], weight)
        distances[j, i] = min(distances[j, i], weight)
    for pivot in range(39):
        distances = np.minimum(distances, distances[:, [pivot]] + distances[[pivot], :])
    finite = distances[np.isfinite(distances)]
    fallback = float(np.max(finite)) if finite.size else 1.0
    return np.where(np.isfinite(distances), distances, fallback)


def phasor(magnitude: np.ndarray, angle_deg: np.ndarray) -> np.ndarray:
    return np.asarray(magnitude, dtype=float) * np.exp(1j * np.deg2rad(np.asarray(angle_deg, dtype=float)))


def safe_log1p(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(np.asarray(values, dtype=float), 0.0))


def finite_or_zero(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def phase_delta_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    return ((delta + 180.0) % 360.0) - 180.0


def radians_to_unit(angle_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(np.asarray(angle_deg, dtype=float))
    return np.cos(radians), np.sin(radians)


def impedance_weighted_distance(bus: int, pmu_buses: list[int] | None = None) -> np.ndarray:
    pmus = PMU_BUSES if pmu_buses is None else [int(pmu) for pmu in pmu_buses]
    distances = electrical_distance_matrix()[bus_index(bus), [bus_index(pmu) for pmu in pmus]]
    scale = max(float(np.nanmedian(distances)), 1e-6)
    return np.exp(-distances / scale)


def angle_between(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.atan2(right[1] - left[1], right[0] - left[0])
