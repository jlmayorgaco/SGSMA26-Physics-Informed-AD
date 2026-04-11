"""Shared topology cosine localizer."""

from __future__ import annotations

import numpy as np

from poc.schema import ALL_BUSES, PMU_BUSES


class CosineLocalizer:
    """Topological localizer using residual-energy cosine matching.

    Method: convert an innovation window to an eight-PMU energy vector
    ``nu_bar`` and compare it against synthetic sensitivity profiles
    ``J_k = exp(-d(k, pmu)/tau)`` for all 39 buses.  In the full journal run
    these profiles can be replaced by power-flow Jacobian columns without
    changing this API.  Parameter count: zero.
    """

    def __init__(self, tau: float = 2.5) -> None:
        self.tau = tau
        self.profiles_ = np.vstack([self._profile(bus) for bus in ALL_BUSES])

    def localize(self, innovation_window: np.ndarray, top_k: int = 3) -> list[int]:
        energy = pmu_energy(innovation_window)
        if np.linalg.norm(energy) < 1e-12:
            return [29, 2, 39][:top_k]
        vec = energy / (np.linalg.norm(energy) + 1e-12)
        profiles = self.profiles_ / (np.linalg.norm(self.profiles_, axis=1, keepdims=True) + 1e-12)
        sims = profiles @ vec
        order = np.argsort(sims)[::-1][:top_k]
        return [ALL_BUSES[int(i)] for i in order]

    def batch_localize(self, windows: np.ndarray, top_k: int = 3) -> list[list[int]]:
        return [self.localize(window, top_k=top_k) for window in windows]

    def _profile(self, bus: int) -> np.ndarray:
        dist = np.array([_grid_distance(bus, pmu) for pmu in PMU_BUSES], dtype=float)
        return np.exp(-dist / self.tau)


def pmu_energy(innovation_window: np.ndarray) -> np.ndarray:
    arr = np.asarray(innovation_window, dtype=float)
    if arr.ndim == 2:
        grouped = arr.reshape(arr.shape[0], len(PMU_BUSES), 14)
    elif arr.ndim == 3:
        grouped = arr
    else:
        raise ValueError("innovation_window must have shape (T,112) or (T,8,14)")
    clean = np.where(np.isnan(grouped), 0.0, grouped)
    return np.sqrt(np.mean(clean * clean, axis=(0, 2)))


_BRANCHES = [
    (1, 2),
    (1, 39),
    (2, 3),
    (2, 25),
    (3, 4),
    (3, 18),
    (4, 5),
    (4, 14),
    (5, 6),
    (5, 8),
    (6, 7),
    (6, 11),
    (7, 8),
    (8, 9),
    (9, 39),
    (10, 11),
    (10, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    (16, 19),
    (16, 21),
    (16, 24),
    (17, 18),
    (17, 27),
    (21, 22),
    (22, 23),
    (23, 24),
    (25, 26),
    (26, 27),
    (26, 28),
    (26, 29),
    (28, 29),
]
_GRAPH: dict[int, list[int]] = {}
for _a, _b in _BRANCHES:
    _GRAPH.setdefault(_a, []).append(_b)
    _GRAPH.setdefault(_b, []).append(_a)


def _grid_distance(src: int, dst: int) -> int:
    if src == dst:
        return 0
    queue = [(src, 0)]
    seen = {src}
    while queue:
        node, dist = queue.pop(0)
        for nb in _GRAPH.get(node, []):
            if nb == dst:
                return dist + 1
            if nb not in seen:
                seen.add(nb)
                queue.append((nb, dist + 1))
    return 12

