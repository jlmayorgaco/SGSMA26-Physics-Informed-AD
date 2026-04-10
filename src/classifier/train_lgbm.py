"""Train LightGBM event classifier.

Workflow
--------
1. Load merged DataFrame and build calibrated Chi2Detector.
2. Run detector on the full dataset to get alarm onsets.
3. For each onset, extract a 37-feature vector and assign label = Event at that frame.
4. Split onsets by time block (train / val) — NO shuffling across splits.
5. Train LightGBM with early stopping on val split.
6. Serialize model to disk.

Synthetic augmentation (M6)
--------------------------
If ``synthetic_dirs`` is provided, load each CSV and extract events from it before
merging into the training set. Synthetic events go into train ONLY.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.classifier.features import extract_all_events, extract_features, N_FEATURES
from src.detector.chi2 import Chi2Detector, compute_eta_simple, extract_data_present
from src.estimator.calibration import channel_cols, calibrate_R
from src.io.load_csv import PMU_BUSES
from src.io.label_utils import event_transition_frames

if TYPE_CHECKING:
    import lightgbm as lgb
    import pandas as pd

log = logging.getLogger(__name__)

# Default hyper-parameters per CLAUDE.md §M4
LGBM_DEFAULTS = dict(
    num_leaves=31,
    n_estimators=200,
    learning_rate=0.05,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)


# ── label assignment ──────────────────────────────────────────────────────────

def _label_at_frame(df: "pd.DataFrame", frame: int) -> int:
    """Return the Event label at the given row index (0 = normal)."""
    val = df["Event"].iloc[frame]
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def _collect_labeled_events(
    df: "pd.DataFrame",
    alarm_indices: np.ndarray,
    J_cols: dict[int, np.ndarray] | None,
    fps: float,
    window_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (X, y) for real alarm onsets, using frame labels."""
    labels = np.array([_label_at_frame(df, int(i)) for i in alarm_indices], dtype=int)
    return extract_all_events(df, alarm_indices, labels, J_cols=J_cols, fps=fps,
                              window_sec=window_sec)


def _sample_normal_frames(
    df: "pd.DataFrame",
    *,
    fps: float,
    max_samples: int = 24,
    guard_sec: float = 5.0,
) -> np.ndarray:
    """Sample stable normal-operation frames for label-0 training examples."""
    if "Event" not in df.columns or len(df) == 0:
        return np.empty(0, dtype=int)

    ev = df["Event"].to_numpy(dtype=int)
    candidates = np.where(ev == 0)[0]
    if len(candidates) == 0:
        return np.empty(0, dtype=int)

    guard = max(1, int(round(guard_sec * fps)))
    trans = event_transition_frames(df, include_label_0=True)
    keep: list[int] = []
    for idx in candidates.tolist():
        if all(abs(idx - int(t)) >= guard for t in trans):
            keep.append(int(idx))
    if not keep:
        return np.empty(0, dtype=int)

    stride = max(1, len(keep) // max_samples)
    sampled = np.array(keep[::stride][:max_samples], dtype=int)
    return sampled


def _collect_transition_events(
    df: "pd.DataFrame",
    J_cols: dict[int, np.ndarray] | None,
    fps: float,
    window_sec: float,
    *,
    ignore_labels: set[int] | None = None,
    add_normal_samples: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract real labeled examples from ground-truth Event transitions."""
    trans_idx = event_transition_frames(df, ignore_labels=ignore_labels or {7})
    trans_labels = np.array([_label_at_frame(df, int(i)) for i in trans_idx], dtype=int)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    if len(trans_idx) > 0:
        X_evt, y_evt = extract_all_events(
            df,
            trans_idx,
            trans_labels,
            J_cols=J_cols,
            fps=fps,
            window_sec=window_sec,
        )
        X_parts.append(X_evt)
        y_parts.append(y_evt)

    if add_normal_samples:
        normal_idx = _sample_normal_frames(df, fps=fps)
        if len(normal_idx) > 0:
            normal_labels = np.zeros(len(normal_idx), dtype=int)
            X_norm, y_norm = extract_all_events(
                df,
                normal_idx,
                normal_labels,
                J_cols=J_cols,
                fps=fps,
                window_sec=window_sec,
            )
            X_parts.append(X_norm)
            y_parts.append(y_norm)

    if not X_parts:
        return np.empty((0, N_FEATURES), dtype=float), np.empty(0, dtype=int)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


# ── calibration helpers ───────────────────────────────────────────────────────

def _build_detector(df: "pd.DataFrame") -> tuple[Chi2Detector, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate Chi2Detector from the first 60 s of Event=0 data.

    Returns (detector, h0_flat, offset) needed for eta computation.
    """
    cols = channel_cols()
    calib_mask = (df["TIMESTAMP"] <= 60.0) & (df["Event"] == 0)
    calib_df = df[calib_mask]

    # h0 = mean of calibration data (approximation to h(x0))
    h0_flat = np.array([
        calib_df[c].dropna().mean() if c in calib_df.columns else 0.0
        for c in cols
    ], dtype=float)
    h0_flat = np.where(np.isfinite(h0_flat), h0_flat, 0.0)

    R, offset = calibrate_R(df, h0_flat)

    eta_full = compute_eta_simple(df, h0_flat, offset, R, cols)
    dp_full  = extract_data_present(df)

    eta_calib = eta_full[calib_mask.to_numpy()]
    dp_calib  = dp_full[calib_mask.to_numpy()]

    det = Chi2Detector(n_z=len(cols), k_on=3, k_off=15, fps=30.0)
    det.calibrate(eta_calib, dp_calib)
    return det, h0_flat, offset, R


# ── synthetic loading ─────────────────────────────────────────────────────────

def _load_synthetic_events(
    syn_dir: Path,
    h0_flat: np.ndarray,
    offset: np.ndarray,
    R: np.ndarray,
    cols: list,
    J_cols: dict | None,
    fps: float,
    window_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load all synthetic events from syn_dir and extract feature/label pairs.

    Each synthetic run is a 6-second CSV window with exactly one event.
    We use the midpoint of the event region (where Event != 0) as the onset frame.
    Falls back to frame 60 (2s in at 30fps) if no event rows found.
    """
    from src.augmentation.andes_sim import load_synthetic
    from src.detector.chi2 import compute_eta_simple

    # Enumerate unique run IDs from file names
    run_ids: set[int] = set()
    for p in syn_dir.glob("syn*_Bus2_Competition_Data_nanmask.csv"):
        try:
            run_ids.add(int(p.name[3:7]))
        except ValueError:
            pass

    if not run_ids:
        log.info("No synthetic events found in %s", syn_dir)
        return np.empty((0, N_FEATURES), dtype=float), np.empty(0, dtype=int)

    X_parts, y_parts = [], []
    for run_id in sorted(run_ids):
        try:
            df_syn = load_synthetic(syn_dir, run_id)
            ev_col = df_syn["Event"]
            ev_frames = np.where(ev_col.to_numpy() != 0)[0]
            if len(ev_frames) == 0:
                continue
            onset = int(ev_frames[len(ev_frames) // 2])
            label = int(ev_col.iloc[onset])
            feats = extract_features(df_syn, onset, J_cols=J_cols,
                                     fps=fps, window_sec=window_sec)
            X_parts.append(feats)
            y_parts.append(label)
        except Exception as e:
            log.debug("Skipping synthetic run %d: %s", run_id, e)

    if not X_parts:
        return np.empty((0, N_FEATURES), dtype=float), np.empty(0, dtype=int)
    return np.stack(X_parts), np.array(y_parts, dtype=int)


# ── training ──────────────────────────────────────────────────────────────────

def train(
    df_train: "pd.DataFrame",
    df_val: "pd.DataFrame",
    J_cols: dict[int, np.ndarray] | None = None,
    fps: float = 30.0,
    window_sec: float = 3.0,
    lgbm_params: dict | None = None,
    model_path: Path | str | None = None,
    synthetic_dir: Path | str | None = None,
) -> "lgb.LGBMClassifier":
    """Train and optionally save the LightGBM classifier.

    Args:
        df_train:      Training split DataFrame.
        df_val:        Validation split DataFrame.
        J_cols:        Sensitivity columns for cosine features (optional).
        fps:           Sampling rate.
        window_sec:    Feature window around each onset.
        lgbm_params:   Override default LightGBM parameters.
        model_path:    If provided, pickle model here.
        synthetic_dir: Directory of synthetic CSVs from andes_sim.generate_all().
                       All synthetic events are merged into the training set only.

    Returns:
        Trained LGBMClassifier.
    """
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("lightgbm not installed — run: pip install lightgbm") from exc

    params = {**LGBM_DEFAULTS, **(lgbm_params or {})}

    # ── detect events in training data ────────────────────────────────────────
    # Use full df for calibration (calibration uses first 60s regardless of split)
    df_combined = df_train  # calibrate on train split only
    det, h0_flat, offset, R = _build_detector(df_combined)
    cols = channel_cols()

    eta_train = compute_eta_simple(df_train, h0_flat, offset, R, cols)
    dp_train  = extract_data_present(df_train)
    ts_train  = df_train["TIMESTAMP"].to_numpy(float)
    res_train = det.detect(eta_train, dp_train, ts_train)

    X_train, y_train = _collect_transition_events(
        df_train,
        J_cols,
        fps,
        window_sec,
        ignore_labels={7},
        add_normal_samples=True,
    )

    log.info("Train (real): %d labeled windows, label distribution: %s",
             len(y_train), dict(zip(*np.unique(y_train, return_counts=True))))

    # ── synthetic augmentation (train only) ───────────────────────────────────
    if synthetic_dir is not None:
        X_syn, y_syn = _load_synthetic_events(
            Path(synthetic_dir), h0_flat, offset, R, cols,
            J_cols, fps, window_sec,
        )
        if len(X_syn) > 0:
            X_train = np.concatenate([X_train, X_syn], axis=0)
            y_train = np.concatenate([y_train, y_syn])
            log.info("Synthetic: +%d events, new label dist: %s",
                     len(y_syn), dict(zip(*np.unique(y_train, return_counts=True))))

    # ── detect events in validation data ──────────────────────────────────────
    eta_val = compute_eta_simple(df_val, h0_flat, offset, R, cols)
    dp_val  = extract_data_present(df_val)
    ts_val  = df_val["TIMESTAMP"].to_numpy(float)
    res_val = det.detect(eta_val, dp_val, ts_val)

    X_val, y_val = _collect_transition_events(
        df_val,
        J_cols,
        fps,
        window_sec,
        ignore_labels={7},
        add_normal_samples=True,
    )

    log.info("Val: %d labeled windows, label distribution: %s",
             len(y_val), dict(zip(*np.unique(y_val, return_counts=True))))

    if len(X_train) == 0:
        raise RuntimeError("No alarm onsets found in training data — check detector calibration")

    # ── fit model ─────────────────────────────────────────────────────────────
    clf = lgb.LGBMClassifier(**params)

    fit_kwargs: dict = {}
    if len(X_val) > 0:
        # LightGBM requires val labels to be a subset of train labels.
        # If val has unseen labels (e.g. only one event type in a split),
        # drop eval_set to avoid a crash; early stopping is a nice-to-have.
        val_labels_seen = set(np.unique(y_val)).issubset(set(np.unique(y_train)))
        if val_labels_seen:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(20, verbose=False),
                                       lgb.log_evaluation(period=50)]
        else:
            log.info("Val labels %s not subset of train labels %s — skipping early stopping",
                     sorted(np.unique(y_val)), sorted(np.unique(y_train)))

    clf.fit(X_train, y_train, **fit_kwargs)

    log.info("LightGBM trained: %d trees, %d features",
             clf.n_estimators_, N_FEATURES)

    if model_path is not None:
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"clf": clf, "h0_flat": h0_flat, "offset": offset,
                         "R": R, "threshold": det.threshold}, f)
        log.info("Model saved to %s", model_path)

    return clf


def predict(
    df: "pd.DataFrame",
    clf: "lgb.LGBMClassifier",
    h0_flat: np.ndarray,
    offset: np.ndarray,
    R: np.ndarray,
    threshold: float,
    J_cols: dict[int, np.ndarray] | None = None,
    fps: float = 30.0,
    window_sec: float = 3.0,
) -> dict:
    """Run full detection + classification pipeline on a DataFrame.

    Returns dict with keys: alarm, alarm_indices, alarm_times, predicted_labels, eta.
    """
    cols = channel_cols()
    det = Chi2Detector(n_z=len(cols), k_on=3, k_off=15, fps=fps)
    det.threshold = threshold

    eta = compute_eta_simple(df, h0_flat, offset, R, cols)
    dp  = extract_data_present(df)
    ts  = df["TIMESTAMP"].to_numpy(float)
    det_result = det.detect(eta, dp, ts)

    onset_idx = det_result["alarm_indices"]
    predicted_labels = []
    for frame in onset_idx:
        feats = extract_features(df, int(frame), J_cols=J_cols, fps=fps, window_sec=window_sec)
        pred = int(clf.predict(feats[None, :])[0])
        predicted_labels.append(pred)

    return {
        **det_result,
        "predicted_labels": np.array(predicted_labels, dtype=int),
        "eta": eta,
    }
