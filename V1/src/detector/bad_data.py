"""Bad-data detector based on three-phase measurement consistency.

Label 7 events are instrumentation artefacts: DATA_PRESENT remains true, but
one phase can jump away from the other two.  A pure system-wide chi-square
threshold is intentionally conservative and can miss these short, local,
non-physical frames.  This module adds a small deterministic detector for
single-PMU phase unbalance without lowering the main physical-event threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from src.detector.debounce import alarm_onsets, debounce
from src.io.load_csv import PMU_BUSES


@dataclass(frozen=True)
class BadDataDetection:
    """Container for bad-data detector outputs."""

    score: np.ndarray
    alarm: np.ndarray
    alarm_indices: np.ndarray
    top_buses: dict[int, int]


def voltage_phase_unbalance(df: "pd.DataFrame") -> tuple[np.ndarray, np.ndarray]:
    """Return per-row voltage phase-unbalance score and top PMU bus.

    The score is ``max_bus((max(|Vabc|)-min(|Vabc|))/median(|Vabc|))``.  Balanced
    physical disturbances move all three phase magnitudes together, while the
    visible label-7 corruptions perturb a single phase at one PMU.
    """
    scores: list[np.ndarray] = []
    for bus in PMU_BUSES:
        cols = [f"BUS{bus}_{phase}_MAG" for phase in ("VA", "VB", "VC")]
        arr = df[cols].to_numpy(dtype=float)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(arr, axis=1)
            spread = (np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)) / np.maximum(
                np.abs(med), 1.0
            )
        scores.append(np.where(np.isfinite(spread), spread, 0.0))

    score_mat = np.vstack(scores).T if scores else np.zeros((len(df), 0))
    if score_mat.shape[1] == 0:
        return np.zeros(len(df), dtype=float), np.full(len(df), -1, dtype=int)
    top_idx = np.argmax(score_mat, axis=1)
    return score_mat[np.arange(len(df)), top_idx], np.array(
        [PMU_BUSES[int(i)] for i in top_idx], dtype=int
    )


def rank_bad_data_buses(
    df: "pd.DataFrame",
    frame: int,
    *,
    fps: float = 30.0,
    window_sec: float = 0.5,
) -> list[tuple[int, float]]:
    """Rank PMU buses by maximum voltage phase unbalance near ``frame``."""
    half = max(1, int(round(fps * window_sec / 2.0)))
    t0 = max(0, int(frame) - half)
    t1 = min(len(df), int(frame) + half + 1)
    ranked: list[tuple[int, float]] = []
    for bus in PMU_BUSES:
        cols = [f"BUS{bus}_{phase}_MAG" for phase in ("VA", "VB", "VC")]
        arr = df.iloc[t0:t1][cols].to_numpy(dtype=float)
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(arr, axis=1)
            spread = (np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)) / np.maximum(
                np.abs(med), 1.0
            )
        score = float(np.nanmax(spread)) if len(spread) else 0.0
        ranked.append((bus, score if np.isfinite(score) else 0.0))
    ranked.sort(key=lambda item: -item[1])
    return ranked[:3]


def _segments_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i, is_on in enumerate(mask):
        if start is None and is_on:
            start = i
        elif start is not None and not is_on:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def _segment_containing(segments: list[tuple[int, int]], frame: int) -> tuple[int, int] | None:
    for start, end in segments:
        if start <= frame < end:
            return start, end
    return None


def _data_present_mask(df: "pd.DataFrame") -> np.ndarray:
    present = np.ones(len(df), dtype=bool)
    for bus in PMU_BUSES:
        col = f"BUS{bus}_DATA_PRESENT"
        if col in df.columns:
            present &= df[col].to_numpy(dtype=float) >= 1.0
    return present


def _near_missing_recovery(df: "pd.DataFrame", frame: int, guard_frames: int) -> bool:
    present = _data_present_mask(df)
    missing = ~present
    if not missing.any():
        return False
    recoveries = np.where(missing[:-1] & present[1:])[0] + 1
    if len(recoveries) == 0:
        return False
    return bool(np.any((frame >= recoveries) & (frame - recoveries <= guard_frames)))


def detect_bad_data_onsets(
    df: "pd.DataFrame",
    *,
    existing_alarm: np.ndarray | None = None,
    fps: float = 30.0,
    threshold: float = 0.04,
    k_on: int = 2,
    k_off: int = 5,
    min_separation_sec: float = 2.0,
    recovery_guard_sec: float = 2.0,
) -> BadDataDetection:
    """Detect short label-7 bad-data bursts from voltage phase unbalance.

    Existing physical/cyber alarm frames and PMU-recovery transients are
    filtered out so this detector contributes only independent bad-data onsets.
    """
    score, top_bus = voltage_phase_unbalance(df)
    present = _data_present_mask(df)
    raw = (score > threshold) & present
    raw_alarm = debounce(raw, k_on=k_on, k_off=k_off)
    raw_onsets = alarm_onsets(raw_alarm)
    raw_segments = _segments_from_mask(raw_alarm)

    min_sep = max(1, int(round(min_separation_sec * fps)))
    guard = max(1, int(round(recovery_guard_sec * fps)))
    existing = np.asarray(existing_alarm, dtype=bool) if existing_alarm is not None else None

    kept: list[int] = []
    for idx in raw_onsets.tolist():
        idx = int(idx)
        if kept and idx - kept[-1] < min_sep:
            continue
        if existing is not None and 0 <= idx < len(existing) and existing[idx]:
            continue
        if _near_missing_recovery(df, idx, guard):
            continue
        kept.append(idx)

    filtered_alarm = np.zeros(len(df), dtype=bool)
    for idx in kept:
        seg = _segment_containing(raw_segments, idx)
        if seg is not None:
            filtered_alarm[seg[0]:seg[1]] = True
        else:
            filtered_alarm[idx] = True

    return BadDataDetection(
        score=score,
        alarm=filtered_alarm,
        alarm_indices=np.array(kept, dtype=int),
        top_buses={int(i): int(top_bus[int(i)]) for i in kept},
    )
