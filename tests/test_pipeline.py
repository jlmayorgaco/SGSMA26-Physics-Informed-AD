"""Tests for end-to-end pipeline: splits, metrics, submission writer."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path("data/raw")
SKIP_NO_DATA = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason="data/raw not present — skipping real-data tests",
)


# ── splits ────────────────────────────────────────────────────────────────────

class TestSplits:
    def _make_df(self, n: int = 10_000) -> pd.DataFrame:
        return pd.DataFrame({
            "TIMESTAMP": np.arange(n) / 30.0,
            "Event": np.zeros(n, dtype=int),
        })

    def test_split_coverage_is_full(self):
        from src.eval.splits import make_splits
        df = self._make_df(10_000)
        s = make_splits(df, fps=30.0)
        total = len(s["train"]) + len(s["val"]) + len(s["test"])
        # Buffer applied on both sides of each split boundary:
        # train/val boundary: buf excluded at end of train + buf at start of val
        # val/test boundary:  buf excluded at end of val   + buf at start of test
        # Total excluded = 4 * buf
        buf = int(round(30.0 * 3.0))
        assert total == 10_000 - 4 * buf

    def test_splits_disjoint(self):
        from src.eval.splits import make_splits
        df = self._make_df(10_000)
        s = make_splits(df, fps=30.0)
        ti = set(s["train"].tolist())
        vi = set(s["val"].tolist())
        xi = set(s["test"].tolist())
        assert ti.isdisjoint(vi), "train and val overlap"
        assert ti.isdisjoint(xi), "train and test overlap"
        assert vi.isdisjoint(xi), "val and test overlap"

    def test_split_order_preserved(self):
        from src.eval.splits import make_splits
        df = self._make_df(10_000)
        s = make_splits(df, fps=30.0)
        # All train indices < val indices < test indices (buffer enforces gap)
        assert s["train"].max() < s["val"].min(), "train overlaps val buffer"
        assert s["val"].max()   < s["test"].min(), "val overlaps test buffer"

    def test_split_fractions_approximate(self):
        from src.eval.splits import make_splits
        df = self._make_df(10_000)
        s = make_splits(df, fps=30.0)
        n = len(df)
        # Train ≈ 70%, val ≈ 15%, test ≈ 15% — allow 1% tolerance after buffering
        assert abs(len(s["train"]) / n - 0.70) < 0.02
        assert abs(len(s["val"])   / n - 0.15) < 0.02

    @SKIP_NO_DATA
    def test_real_data_splits(self):
        from src.io.load_csv import load_all
        from src.eval.splits import make_splits
        df = load_all(DATA_DIR)
        s = make_splits(df)
        assert len(s["train"]) > 0
        assert len(s["val"])   > 0
        assert len(s["test"])  > 0
        # Buffer: no window within 90 frames of a split boundary
        assert s["train"].max() + 90 <= s["val"].min()
        assert s["val"].max()   + 90 <= s["test"].min()


# ── detection metrics ─────────────────────────────────────────────────────────

class TestDetectionMetrics:
    def test_perfect_detection(self):
        from src.eval.metrics import detection_metrics
        gt = [(10.0, 20.0), (50.0, 60.0)]
        alarm_times = np.array([11.0, 52.0])
        res = detection_metrics(alarm_times, gt, total_duration_sec=90.0)
        assert res["tp"] == 2
        assert res["fp"] == 0
        assert res["fn"] == 0
        assert abs(res["precision"] - 1.0) < 1e-9
        assert abs(res["recall"] - 1.0) < 1e-9

    def test_false_alarm(self):
        from src.eval.metrics import detection_metrics
        gt = [(50.0, 60.0)]
        alarm_times = np.array([11.0, 52.0])   # 11s = FP, 52s = TP
        res = detection_metrics(alarm_times, gt, total_duration_sec=90.0)
        assert res["tp"] == 1
        assert res["fp"] == 1
        assert res["fn"] == 0
        assert abs(res["precision"] - 0.5) < 1e-9

    def test_missed_event(self):
        from src.eval.metrics import detection_metrics
        gt = [(10.0, 20.0), (50.0, 60.0)]
        alarm_times = np.array([52.0])
        res = detection_metrics(alarm_times, gt, total_duration_sec=90.0)
        assert res["fn"] == 1
        assert res["tp"] == 1
        assert abs(res["recall"] - 0.5) < 1e-9

    def test_no_alarms(self):
        from src.eval.metrics import detection_metrics
        gt = [(10.0, 20.0)]
        res = detection_metrics(np.array([]), gt, total_duration_sec=90.0)
        assert res["tp"] == 0
        assert res["fn"] == 1
        assert res["precision"] == 0.0
        assert res["recall"] == 0.0

    def test_delay_computed(self):
        from src.eval.metrics import detection_metrics
        gt = [(10.0, 20.0)]
        alarm_times = np.array([12.5])
        res = detection_metrics(alarm_times, gt, total_duration_sec=90.0)
        assert abs(res["mean_delay_sec"] - 2.5) < 1e-9


# ── classification metrics ────────────────────────────────────────────────────

class TestClassificationMetrics:
    def test_perfect(self):
        from src.eval.metrics import classification_metrics
        y = np.array([1, 2, 3, 4, 5])
        res = classification_metrics(y, y)
        assert abs(res["macro_f1"] - 1.0) < 1e-9

    def test_all_wrong(self):
        from src.eval.metrics import classification_metrics
        y_true = np.array([1, 1, 1])
        y_pred = np.array([2, 2, 2])
        res = classification_metrics(y_true, y_pred)
        assert res["macro_f1"] == 0.0

    def test_confusion_matrix_shape(self):
        from src.eval.metrics import classification_metrics
        y_true = np.array([0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3])
        res = classification_metrics(y_true, y_pred)
        assert res["confusion_matrix"].shape == (9, 9)

    def test_per_class_keys(self):
        from src.eval.metrics import classification_metrics
        y = np.array([1, 2])
        res = classification_metrics(y, y)
        for lbl in [1, 2]:
            assert lbl in res["per_class"]
            assert "f1" in res["per_class"][lbl]


# ── localization metrics ──────────────────────────────────────────────────────

class TestLocalizationMetrics:
    def test_perfect_top1(self):
        from src.eval.metrics import localization_metrics
        top3 = [[2, 5, 6], [39, 22, 29]]
        true = [2, 39]
        res = localization_metrics(top3, true)
        assert abs(res["top1_acc"] - 1.0) < 1e-9
        assert abs(res["top3_acc"] - 1.0) < 1e-9

    def test_top3_but_not_top1(self):
        from src.eval.metrics import localization_metrics
        top3 = [[5, 2, 6], [22, 39, 29]]
        true = [2, 39]
        res = localization_metrics(top3, true)
        assert res["top1_acc"] == 0.0
        assert abs(res["top3_acc"] - 1.0) < 1e-9

    def test_empty_input(self):
        from src.eval.metrics import localization_metrics
        res = localization_metrics([], [])
        assert res["n_events"] == 0

    def test_electrical_distance_zero_for_same_bus(self):
        from src.eval.metrics import localization_metrics
        # Simple 2-bus Zbus: diagonal entries non-zero, off-diagonal small
        zbus = np.array([[0.1+0j, 0.01+0j], [0.01+0j, 0.2+0j]])
        ext_bus_order = [2, 39]
        top3 = [[2], [39]]
        true = [2, 39]
        res = localization_metrics(top3, true, zbus=zbus, ext_bus_order=ext_bus_order)
        # d_ii = |Z_ii + Z_ii - 2*Z_ii| = 0
        assert abs(res.get("mean_elec_dist_error", 1.0)) < 1e-9


# ── make_submission ───────────────────────────────────────────────────────────

class TestMakeSubmission:
    @SKIP_NO_DATA
    def test_write_and_verify(self, tmp_path):
        from src.io.load_csv import load_all
        from src.pipeline.make_submission import write_submission, verify_timestamps

        df = load_all(DATA_DIR)
        n = len(df)
        pred_event = np.zeros(n, dtype=int)
        pred_loc   = np.full(n, -1, dtype=int)

        written = write_submission(df, pred_event, pred_loc, tmp_path, DATA_DIR)
        assert len(written) == 9  # 8 bus CSVs + combined

        ok = verify_timestamps(tmp_path, DATA_DIR)
        for bus, is_ok in ok.items():
            assert is_ok, f"Bus{bus} TIMESTAMP mismatch in prediction CSV"

    @SKIP_NO_DATA
    def test_combined_csv_has_bus_column(self, tmp_path):
        from src.io.load_csv import load_all
        from src.pipeline.make_submission import write_submission

        df = load_all(DATA_DIR)
        n = len(df)
        write_submission(df, np.zeros(n, dtype=int), np.full(n, -1, dtype=int),
                         tmp_path, DATA_DIR)
        combined = pd.read_csv(tmp_path / "submission.csv")
        assert "Bus" in combined.columns
        assert "Predicted_Event" in combined.columns
        assert "Predicted_Location" in combined.columns
        assert "TIMESTAMP" in combined.columns
        # 8 buses × n rows
        assert len(combined) == 8 * n

    def test_synthetic_write(self, tmp_path):
        """write_submission works even without real data CSVs (no src files)."""
        from src.pipeline.make_submission import write_submission
        from src.io.load_csv import PMU_BUSES

        n = 180
        df = pd.DataFrame({
            "TIMESTAMP": np.round(np.arange(n) / 30.0, 3),
            "Event": np.zeros(n, dtype=int),
        })
        pred_event = np.ones(n, dtype=int)
        pred_loc   = np.full(n, 2, dtype=int)

        # Use empty dir so fallback TIMESTAMP path is taken
        empty_src = tmp_path / "empty_src"
        empty_src.mkdir()
        written = write_submission(df, pred_event, pred_loc, tmp_path / "out", empty_src)
        assert len(written) == 9

        # Check each bus CSV
        out_dir = tmp_path / "out"
        for bus in PMU_BUSES:
            p = out_dir / f"Bus{bus}_Predictions.csv"
            assert p.exists()
            sub_df = pd.read_csv(p)
            assert list(sub_df.columns) == ["TIMESTAMP", "Predicted_Event", "Predicted_Location"]
            assert all(sub_df["Predicted_Event"] == 1)
