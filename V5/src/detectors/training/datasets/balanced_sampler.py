from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.detectors.domain.models.window_sample import WindowSample


def sample_balanced_windows(samples: list[WindowSample], max_per_class: int | None = None, seed: int = 12345) -> list[WindowSample]:
    if not samples:
        return []
    groups: dict[int, list[WindowSample]] = defaultdict(list)
    for sample in samples:
        groups[int(sample.y_binary)].append(sample)
    if len(groups) <= 1:
        return list(samples)
    min_size = min(len(v) for v in groups.values())
    target = min_size if max_per_class is None else min(min_size, int(max_per_class))
    rng = np.random.default_rng(seed)
    out: list[WindowSample] = []
    for label, group in sorted(groups.items()):
        if len(group) <= target:
            out.extend(group)
            continue
        idx = rng.choice(len(group), size=target, replace=False)
        out.extend(group[int(i)] for i in idx)
    out.sort(key=lambda sample: (sample.scenario_id, sample.window_index))
    return out

