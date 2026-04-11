"""End-to-end inference pipeline: raw CSVs → predictions.

CLI
---
    python -m src.pipeline.run_inference --data data/raw/ [options]

Full pipeline
-------------
1. Load and merge the 8 bus CSVs (src/io/load_csv.py).
2. Load the grid case and compute Jacobian sensitivity columns.
3. Train (or load) the LightGBM classifier from the training split.
4. Run Chi2Detector on the full dataset to get alarm onsets.
5. For each alarm onset: extract features → classify → localize.
6. Write prediction CSVs and combined submission file.
7. Compute and print evaluation metrics (if ground-truth Event column present).

CPU-only.  Full 90-minute dataset completes in < 5 minutes.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import random
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SEED = 42


def _seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import lightgbm as lgb  # noqa: F401 — just log; seeding done via random_state in clf
    except ImportError:
        pass
    log.info("Random seed: %d", seed)


def _load_grid(raw_path: Path) -> tuple:
    """Return (grid, J_cols, branches, zbus, ext_bus_order)."""
    from src.grid.load_case import load_case
    from src.grid.jacobians import compute_jacobians, bus_sensitivity_columns

    grid = load_case(raw_path)
    J = compute_jacobians(grid)
    J_cols = bus_sensitivity_columns(J, grid)
    branches = grid.branch_list
    zbus = grid.Zbus
    ext_bus_order = grid.ext_bus_order
    return grid, J_cols, branches, zbus, ext_bus_order


def _train_model(
    df_train,
    df_val,
    grid,
    J_cols,
    model_path: Path | None,
    synthetic_dir: Path | None,
) -> tuple:
    """Train or load classifier. Returns (clf, h0_flat, offset, R, threshold)."""
    from src.classifier.train_lgbm import train, _build_detector

    det, h0_flat, offset, R = _build_detector(df_train)

    if model_path is not None and model_path.exists():
        log.info("Loading saved model from %s", model_path)
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        return (bundle["clf"], bundle["h0_flat"], bundle["offset"],
                bundle["R"], bundle["threshold"])

    clf = train(
        df_train, df_val,
        J_cols=J_cols,
        lgbm_params={"n_estimators": 200, "verbose": -1},
        model_path=model_path,
        synthetic_dir=synthetic_dir,
        grid=grid,
    )
    return clf, h0_flat, offset, R, det.threshold


def _run_detection(df, h0_flat, offset, R, threshold, fps: float = 30.0):
    from src.detector.chi2 import Chi2Detector, compute_eta_simple, extract_data_present
    from src.detector.bad_data import detect_bad_data_onsets
    from src.estimator.calibration import channel_cols

    cols = channel_cols()
    eta = compute_eta_simple(df, h0_flat, offset, R, cols)
    dp  = extract_data_present(df)
    ts  = df["TIMESTAMP"].to_numpy(float)

    det = Chi2Detector(n_z=len(cols), k_on=3, k_off=15, fps=fps)
    det.threshold = threshold
    result = det.detect(eta, dp, ts)

    # Label-7 bad data is not a physical/cyber state change: DATA_PRESENT stays
    # true and only one phase at one PMU is corrupted. Detect it separately
    # from three-phase voltage consistency, then merge independent onsets.
    bad = detect_bad_data_onsets(df, existing_alarm=result["alarm"], fps=fps)
    if len(bad.alarm_indices) > 0:
        merged = np.array(
            sorted(set(result["alarm_indices"].astype(int).tolist()) | set(bad.alarm_indices.tolist())),
            dtype=int,
        )
        result["alarm_indices"] = merged
        result["alarm_times"] = list(ts[merged])
        result["alarm"] = np.asarray(result["alarm"], dtype=bool) | bad.alarm
    result["bad_data_indices"] = bad.alarm_indices
    result["bad_data_top_buses"] = bad.top_buses
    result["bad_data_score"] = bad.score
    return result, eta


def _classify_and_localize(df, onset_idx, clf, grid, J_cols, branches,
                            h0_flat, offset, R, fps: float = 30.0,
                            forced_labels_by_frame: dict[int, int] | None = None):
    from src.classifier.features import extract_features
    from src.classifier.rules import apply_physics_label_overrides
    from src.estimator.topology_state import TopologyStateEstimator
    from src.localizer.cosine_match import locate

    predicted_labels = []
    top3_buses_all   = []
    top3_lines_all   = []
    state_estimator = TopologyStateEstimator(grid) if grid is not None else None

    import warnings as _warnings
    for frame in onset_idx:
        feats = extract_features(
            df,
            int(frame),
            J_cols=J_cols,
            grid=grid,
            state_estimator=state_estimator,
            fps=fps,
        )
        forced = None if forced_labels_by_frame is None else forced_labels_by_frame.get(int(frame))
        if forced is not None:
            label = int(forced)
        else:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", UserWarning)
                label = int(clf.predict(feats[None, :])[0])
            label = apply_physics_label_overrides(label, feats)
        predicted_labels.append(label)

        loc_result = locate(
            df, int(frame), label, J_cols, branches,
            fps=fps, h0_flat=h0_flat, offset=offset, R=R,
            grid=grid, state_estimator=state_estimator,
        )
        top3_buses_all.append([b for b, _ in loc_result["top3_buses"]])
        top3_lines_all.append(loc_result["top3_lines"])

    return (np.array(predicted_labels, dtype=int),
            top3_buses_all, top3_lines_all)


def _segments_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open true segments from a boolean mask."""
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


def _segment_containing(
    segments: list[tuple[int, int]],
    frame: int,
) -> tuple[int, int] | None:
    """Return the segment containing ``frame``, if any."""
    for start, end in segments:
        if start <= frame < end:
            return start, end
    return None


def _default_label_duration_frames(label: int, fps: float) -> int:
    """Fallback row-label interval duration by event type."""
    seconds_by_label = {
        1: 5.0,
        2: 5.0,
        3: 300.0,
        4: 300.0,
        6: 20.0,
        7: 1.0,
        8: 5.0,
    }
    return max(1, int(round(seconds_by_label.get(int(label), 5.0) * fps)))


def _build_prediction_columns(
    df,
    det_result: dict,
    onset_idx: np.ndarray,
    predicted_labels: np.ndarray,
    top3_buses_all: list[list[int]],
    top3_lines_all: list[list[tuple[tuple[int, int], float]]] | None = None,
    *,
    fps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand alarm onsets into sample-level submission labels.

    Detection metrics remain onset-based. Submission CSVs are row-level, so
    persistent generation/load changes are extended beyond the transient edge,
    while cyber dropouts follow DATA_PRESENT-missing spans exactly.
    """
    from src.detector.chi2 import extract_data_present

    n = len(df)
    pred_event_col = np.zeros(n, dtype=int)
    pred_loc_col = np.full(n, -1, dtype=int)
    onset_idx = np.asarray(onset_idx, dtype=int)
    predicted_labels = np.asarray(predicted_labels, dtype=int)
    order = np.argsort(onset_idx)

    alarm_segments = _segments_from_mask(np.asarray(det_result.get("alarm", []), dtype=bool))
    dp = extract_data_present(df)
    missing_segments = _segments_from_mask((dp < 1).any(axis=1))

    for rank, pos in enumerate(order):
        frame = int(onset_idx[pos])
        label = int(predicted_labels[pos])
        if frame < 0 or frame >= n:
            continue

        next_frame = int(onset_idx[order[rank + 1]]) if rank + 1 < len(order) else n
        alarm_seg = _segment_containing(alarm_segments, frame)
        loc = int(top3_buses_all[pos][0]) if pos < len(top3_buses_all) and top3_buses_all[pos] else -1
        if (
            label == 2
            and top3_lines_all is not None
            and pos < len(top3_lines_all)
            and top3_lines_all[pos]
        ):
            # Submission location is bus-valued. For line events, report a
            # deterministic endpoint while the full line Top-3 remains in the
            # localizer result/report.
            line, _score = top3_lines_all[pos][0]
            loc = int(max(line))

        if label == 5:
            seg = _segment_containing(missing_segments, frame)
            if seg is None:
                start = frame
                end = min(n, frame + _default_label_duration_frames(label, fps))
            else:
                start, end = seg
            end = min(end, next_frame)
        elif label == 6:
            seg = _segment_containing(missing_segments, frame)
            start = frame
            if next_frame < n:
                end = next_frame
            elif seg is not None:
                end = max(frame + 1, seg[1])
            else:
                end = frame + _default_label_duration_frames(label, fps)
            end = min(n, end)
        else:
            start = frame
            end = min(n, frame + _default_label_duration_frames(label, fps))
            if alarm_seg is not None:
                end = max(end, alarm_seg[1])
            end = min(end, next_frame)

        if end <= start:
            end = min(n, start + 1)
        pred_event_col[start:end] = label
        pred_loc_col[start:end] = loc

    return pred_event_col, pred_loc_col


def _alarm_submission_location(
    pos: int,
    label: int,
    top3_buses_all: list[list[int]],
    top3_lines_all: list[list[tuple[tuple[int, int], float]]] | None = None,
) -> int:
    """Return the bus-valued location used by the submission writer."""
    loc = int(top3_buses_all[pos][0]) if pos < len(top3_buses_all) and top3_buses_all[pos] else -1
    if (
        int(label) == 2
        and top3_lines_all is not None
        and pos < len(top3_lines_all)
        and top3_lines_all[pos]
    ):
        line, _score = top3_lines_all[pos][0]
        loc = int(max(line))
    return loc


def _physical_companion_for_label6(
    frame: int,
    onset_idx: np.ndarray,
    predicted_labels: np.ndarray,
    top3_buses_all: list[list[int]],
    top3_lines_all: list[list[tuple[tuple[int, int], float]]] | None = None,
    *,
    fps: float = 30.0,
) -> tuple[int, int]:
    """Infer the physical component paired with a cyber+physical label-6 interval."""
    order = np.argsort(np.asarray(onset_idx, dtype=int))
    max_gap = int(round(90.0 * fps))
    best: tuple[int, int] | None = None
    best_gap = max_gap + 1
    for pos in order:
        pos = int(pos)
        lbl = int(predicted_labels[pos])
        if lbl not in {1, 2, 3, 4}:
            continue
        gap = int(onset_idx[pos]) - int(frame)
        if 0 <= gap <= max_gap and gap < best_gap:
            loc = _alarm_submission_location(pos, lbl, top3_buses_all, top3_lines_all)
            best = (lbl, loc)
            best_gap = gap
    if best is not None:
        return best
    return 3, -1


def _build_per_bus_prediction_columns(
    df,
    pred_event_col: np.ndarray,
    pred_loc_col: np.ndarray,
    onset_idx: np.ndarray,
    predicted_labels: np.ndarray,
    top3_buses_all: list[list[int]],
    top3_lines_all: list[list[tuple[tuple[int, int], float]]] | None = None,
    *,
    fps: float = 30.0,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Apply bus-local cyber/bad-data semantics to row-level predictions."""
    from src.io.load_csv import PMU_BUSES

    pred_event_col = np.asarray(pred_event_col, dtype=int)
    pred_loc_col = np.asarray(pred_loc_col, dtype=int)
    n = len(pred_event_col)

    label6_physical = np.zeros(n, dtype=int)
    label6_location = np.full(n, -1, dtype=int)
    for start, end in _segments_from_mask(pred_event_col == 6):
        companion_label, companion_loc = _physical_companion_for_label6(
            start,
            onset_idx,
            predicted_labels,
            top3_buses_all,
            top3_lines_all,
            fps=fps,
        )
        label6_physical[start:end] = companion_label
        label6_location[start:end] = companion_loc

    event_by_bus: dict[int, np.ndarray] = {}
    loc_by_bus: dict[int, np.ndarray] = {}
    for bus in PMU_BUSES:
        event = pred_event_col.copy()
        loc = pred_loc_col.copy()
        dp_col = f"BUS{bus}_DATA_PRESENT"
        missing = df[dp_col].to_numpy(float) < 1.0 if dp_col in df.columns else np.zeros(n, dtype=bool)

        cyber_only = event == 5
        event[cyber_only & ~missing] = 0
        loc[cyber_only & ~missing] = -1
        loc[cyber_only & missing] = int(bus)

        cyber_physical = event == 6
        non_missing_overlap = cyber_physical & ~missing
        event[non_missing_overlap] = label6_physical[non_missing_overlap]
        loc[non_missing_overlap] = label6_location[non_missing_overlap]
        event[cyber_physical & missing] = 6
        loc[cyber_physical & missing] = int(bus)

        physical_while_missing = np.isin(event, [1, 2, 3, 4]) & missing
        event[physical_while_missing] = 6
        loc[physical_while_missing] = int(bus)

        bad_data = event == 7
        not_this_bus = bad_data & (pred_loc_col != int(bus))
        event[not_this_bus] = 0
        loc[not_this_bus] = -1
        loc[event == 0] = -1

        event_by_bus[int(bus)] = event
        loc_by_bus[int(bus)] = loc

    return event_by_bus, loc_by_bus


def run(
    data_dir: Path,
    out_dir: Path,
    model_path: Path | None = None,
    synthetic_dir: Path | None = None,
    raw_path: Path | None = None,
    fps: float = 30.0,
    seed: int = SEED,
) -> dict:
    """Full inference pipeline. Returns metrics dict (empty if no ground truth)."""
    _seed_all(seed)
    t_start = time.perf_counter()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    from src.io.load_csv import load_all
    from src.eval.splits import make_splits

    log.info("Loading data from %s", data_dir)
    df = load_all(data_dir)
    splits = make_splits(df, fps=fps)
    train_idx, val_idx = splits["train"], splits["val"]

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx].reset_index(drop=True)

    log.info("Split: train=%d  val=%d  test=%d rows",
             len(train_idx), len(val_idx), len(splits["test"]))

    # ── 2. Grid and Jacobians ─────────────────────────────────────────────────
    if raw_path is None:
        raw_path = Path("data/metadata/IEEE 39 Bus Power System.raw")
    log.info("Loading grid from %s", raw_path)
    grid, J_cols, branches, zbus, ext_bus_order = _load_grid(raw_path)

    # ── 3. Train / load model ─────────────────────────────────────────────────
    log.info("Training classifier...")
    clf, h0_flat, offset, R, threshold = _train_model(
        df_train, df_val, grid, J_cols, model_path, synthetic_dir,
    )

    # ── 4. Detection on full dataset ──────────────────────────────────────────
    log.info("Running detector on full dataset...")
    det_result, eta = _run_detection(df, h0_flat, offset, R, threshold, fps)
    onset_idx = det_result["alarm_indices"]
    alarm_times = df["TIMESTAMP"].to_numpy(float)[onset_idx]

    log.info("Detected %d alarm onsets", len(onset_idx))

    # ── 5. Classification + localization ─────────────────────────────────────
    log.info("Classifying and localizing...")
    forced_labels = {int(i): 7 for i in det_result.get("bad_data_indices", [])}
    predicted_labels, top3_buses_all, top3_lines_all = _classify_and_localize(
        df, onset_idx, clf, grid, J_cols, branches, h0_flat, offset, R, fps,
        forced_labels_by_frame=forced_labels,
    )

    # Build sample-level submission labels from fast alarm onsets. Long
    # step-change events keep abnormal labels after the transient detector edge.
    pred_event_col, pred_loc_col = _build_prediction_columns(
        df,
        det_result,
        onset_idx,
        predicted_labels,
        top3_buses_all,
        top3_lines_all,
        fps=fps,
    )

    # ── 6. Write submission ───────────────────────────────────────────────────
    pred_event_by_bus, pred_loc_by_bus = _build_per_bus_prediction_columns(
        df,
        pred_event_col,
        pred_loc_col,
        onset_idx,
        predicted_labels,
        top3_buses_all,
        top3_lines_all,
        fps=fps,
    )

    from src.pipeline.make_submission import write_submission
    write_submission(df, pred_event_by_bus, pred_loc_by_bus, out_dir, data_dir)

    elapsed = time.perf_counter() - t_start
    log.info("Inference complete in %.1f s", elapsed)

    # ── 7. Metrics (if Event ground truth present) ────────────────────────────
    metrics: dict = {}
    if "Event" in df.columns and df["Event"].max() > 0:
        from src.eval.metrics import compute_all_metrics, print_metrics
        from src.io.load_events import load_events

        ts_arr = df["TIMESTAMP"].to_numpy(float)
        true_at_alarms = np.array(
            [int(df["Event"].iloc[int(i)]) for i in onset_idx], dtype=int
        )

        # Load ground-truth event locations from metadata
        meta_dir = raw_path.parent if raw_path is not None else Path("data/metadata")
        events = load_events(meta_dir)
        true_buses = [
            _gt_bus_for_alarm_frame(df, int(i), events)
            for i in onset_idx
        ]

        # Build event windows from Event column transitions
        ev_col = df["Event"].to_numpy(int)
        ts_col = df["TIMESTAMP"].to_numpy(float)
        gt_windows = _extract_gt_windows(ev_col, ts_col)

        metrics = compute_all_metrics(
            df, alarm_times, predicted_labels,
            top3_buses_all, gt_windows,
            true_at_alarms, true_buses,
            zbus=zbus.real if zbus is not None else None,
            ext_bus_order=ext_bus_order,
        )
        print_metrics(metrics)
        metrics["elapsed_sec"] = elapsed

    return metrics


def _extract_gt_windows(
    ev_col: np.ndarray,
    ts_col: np.ndarray,
) -> list[tuple[float, float]]:
    """Extract one (start_sec, end_sec) window for each non-zero label segment."""
    windows: list[tuple[float, float]] = []
    if len(ev_col) == 0:
        return windows

    current = int(ev_col[0])
    t_start = float(ts_col[0])
    for i in range(1, len(ev_col)):
        label = int(ev_col[i])
        if label == current:
            continue
        if current != 0:
            windows.append((t_start, float(ts_col[i - 1])))
        current = label
        t_start = float(ts_col[i])

    if current != 0:
        windows.append((t_start, float(ts_col[-1])))
    return windows


def _gt_bus_for_alarm(
    alarm_time_sec: float,
    events: list,
    tol_sec: float = 60.0,
    label: int | None = None,
) -> int:
    """Return the ground-truth location bus for an alarm at alarm_time_sec.

    Matches the alarm to the nearest event by time within tol_sec.
    Returns -1 if no match found.
    """
    best_bus = -1
    best_dt  = tol_sec + 1.0
    for ev in events:
        if label is not None and int(getattr(ev, "label", -1)) != int(label):
            continue
        dt = abs(ev.approx_time_sec - alarm_time_sec)
        if dt < best_dt:
            best_dt  = dt
            best_bus = ev.location_bus if ev.location_bus is not None else -1
    return best_bus


def _gt_bus_for_alarm_frame(
    df,
    frame: int,
    events: list,
    tol_sec: float = 60.0,
    fps: float = 30.0,
) -> int:
    """Return a scoring bus for one alarm frame.

    Metadata covers scheduled physical/cyber events, but visible bad-data
    label-7 windows are encoded only in the per-bus ``BUSk_Event`` columns.
    """
    from src.io.load_csv import PMU_BUSES

    label = int(df["Event"].iloc[int(frame)])
    if label in {5, 6, 7}:
        guard = max(1, int(round(fps)))
        lo = max(0, int(frame) - guard)
        hi = min(len(df), int(frame) + guard + 1)
        for bus in PMU_BUSES:
            col = f"BUS{bus}_Event"
            if col in df.columns and (df[col].iloc[lo:hi].astype(int) == label).any():
                return int(bus)

    return _gt_bus_for_alarm(
        float(df["TIMESTAMP"].iloc[int(frame)]),
        events,
        tol_sec=tol_sec,
        label=label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SGSMA 2026 end-to-end inference pipeline"
    )
    parser.add_argument("--data",       default="data/raw",
                        help="Directory with 8 bus competition CSVs")
    parser.add_argument("--out",        default="predictions",
                        help="Output directory for prediction CSVs")
    parser.add_argument("--model",      default=None,
                        help="Path to saved model pickle (train if missing)")
    parser.add_argument("--synthetic",  default="data/synthetic",
                        help="Directory of synthetic augmentation CSVs")
    parser.add_argument("--raw",        default="data/metadata/IEEE 39 Bus Power System.raw",
                        help="PSS/E .raw grid file")
    parser.add_argument("--seed",       type=int, default=SEED,
                        help="Global random seed (default 42)")
    parser.add_argument("--log-level",  default="INFO",
                        help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir      = Path(args.data)
    out_dir       = Path(args.out)
    model_path    = Path(args.model) if args.model else None
    synthetic_dir = Path(args.synthetic) if args.synthetic else None
    raw_path      = Path(args.raw)

    if not data_dir.exists():
        parser.error(f"--data directory not found: {data_dir}")

    syn_dir = synthetic_dir if (synthetic_dir and synthetic_dir.exists()) else None

    run(
        data_dir=data_dir,
        out_dir=out_dir,
        model_path=model_path,
        synthetic_dir=syn_dir,
        raw_path=raw_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
