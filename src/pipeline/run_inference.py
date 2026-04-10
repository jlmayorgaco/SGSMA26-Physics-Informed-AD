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
    )
    return clf, h0_flat, offset, R, det.threshold


def _run_detection(df, h0_flat, offset, R, threshold, fps: float = 30.0):
    from src.detector.chi2 import Chi2Detector, compute_eta_simple, extract_data_present
    from src.estimator.calibration import channel_cols

    cols = channel_cols()
    eta = compute_eta_simple(df, h0_flat, offset, R, cols)
    dp  = extract_data_present(df)
    ts  = df["TIMESTAMP"].to_numpy(float)

    det = Chi2Detector(n_z=len(cols), k_on=3, k_off=15, fps=fps)
    det.threshold = threshold
    return det.detect(eta, dp, ts), eta


def _classify_and_localize(df, onset_idx, clf, J_cols, branches,
                            h0_flat, offset, R, fps: float = 30.0):
    from src.classifier.features import extract_features
    from src.localizer.cosine_match import locate

    predicted_labels = []
    top3_buses_all   = []
    top3_lines_all   = []

    import warnings as _warnings
    for frame in onset_idx:
        feats = extract_features(df, int(frame), J_cols=J_cols, fps=fps)
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", UserWarning)
            label = int(clf.predict(feats[None, :])[0])
        predicted_labels.append(label)

        loc_result = locate(
            df, int(frame), label, J_cols, branches,
            fps=fps, h0_flat=h0_flat, offset=offset, R=R,
        )
        top3_buses_all.append([b for b, _ in loc_result["top3_buses"]])
        top3_lines_all.append(loc_result["top3_lines"])

    return (np.array(predicted_labels, dtype=int),
            top3_buses_all, top3_lines_all)


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
    _, J_cols, branches, zbus, ext_bus_order = _load_grid(raw_path)

    # ── 3. Train / load model ─────────────────────────────────────────────────
    log.info("Training classifier...")
    clf, h0_flat, offset, R, threshold = _train_model(
        df_train, df_val, J_cols, model_path, synthetic_dir,
    )

    # ── 4. Detection on full dataset ──────────────────────────────────────────
    log.info("Running detector on full dataset...")
    det_result, eta = _run_detection(df, h0_flat, offset, R, threshold, fps)
    onset_idx = det_result["alarm_indices"]
    alarm_times = df["TIMESTAMP"].to_numpy(float)[onset_idx]

    log.info("Detected %d alarm onsets", len(onset_idx))

    # ── 5. Classification + localization ─────────────────────────────────────
    log.info("Classifying and localizing...")
    predicted_labels, top3_buses_all, top3_lines_all = _classify_and_localize(
        df, onset_idx, clf, J_cols, branches, h0_flat, offset, R, fps,
    )

    # Build per-row prediction arrays
    n = len(df)
    pred_event_col = np.zeros(n, dtype=int)
    pred_loc_col   = np.full(n, -1, dtype=int)

    for i, (frame, lbl) in enumerate(zip(onset_idx, predicted_labels)):
        # Mark the full alarm window (k_on=3 to k_off=15 frames)
        # For submission we mark a ±1 s window around the onset
        half = int(round(fps * 1.0))
        lo = max(0, int(frame) - half)
        hi = min(n, int(frame) + half)
        pred_event_col[lo:hi] = lbl
        top3 = top3_buses_all[i]
        pred_loc_col[lo:hi] = top3[0] if top3 else -1

    # ── 6. Write submission ───────────────────────────────────────────────────
    from src.pipeline.make_submission import write_submission
    write_submission(df, pred_event_col, pred_loc_col, out_dir, data_dir)

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
            _gt_bus_for_alarm(float(ts_arr[int(i)]), events)
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
    """Extract (start_sec, end_sec) windows from Event column transitions."""
    windows: list[tuple[float, float]] = []
    in_event = False
    t_start = 0.0
    for i in range(len(ev_col)):
        if not in_event and ev_col[i] != 0:
            in_event = True
            t_start = ts_col[i]
        elif in_event and ev_col[i] == 0:
            in_event = False
            windows.append((t_start, ts_col[i - 1]))
    if in_event:
        windows.append((t_start, ts_col[-1]))
    return windows


def _gt_bus_for_alarm(
    alarm_time_sec: float,
    events: list,
    tol_sec: float = 60.0,
) -> int:
    """Return the ground-truth location bus for an alarm at alarm_time_sec.

    Matches the alarm to the nearest event by time within tol_sec.
    Returns -1 if no match found.
    """
    best_bus = -1
    best_dt  = tol_sec + 1.0
    for ev in events:
        dt = abs(ev.approx_time_sec - alarm_time_sec)
        if dt < best_dt:
            best_dt  = dt
            best_bus = ev.location_bus if ev.location_bus is not None else -1
    return best_bus


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
