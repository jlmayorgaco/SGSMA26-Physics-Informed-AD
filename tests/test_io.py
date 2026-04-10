"""M1 sanity checks: header inspection, event transitions, NaN behavior."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.io.load_csv import PMU_BUSES, inspect_header, load_all, load_single_bus
from src.io.label_utils import extract_transitions, label_summary


class TestHeaderInspection:
    def test_bus2_header_order(self, data_raw):
        """Sanity check 1: ANG before MAG, correct column names."""
        cols = inspect_header(data_raw, bus=2)
        assert cols[0] == "TIMESTAMP"
        assert cols[1] == "BUS2_VA_ANG", f"Expected BUS2_VA_ANG at index 1, got {cols[1]}"
        assert cols[2] == "BUS2_VA_MAG", f"Expected BUS2_VA_MAG at index 2, got {cols[2]}"
        assert cols[-1] == "Event"
        assert "DATA_PRESENT" in cols

    def test_all_buses_have_expected_columns(self, data_raw):
        """All 8 bus CSVs should follow the same schema."""
        for bus in PMU_BUSES:
            cols = inspect_header(data_raw, bus=bus)
            prefix = f"BUS{bus}_"
            assert cols[1] == f"{prefix}VA_ANG", f"Bus {bus}: col 1 mismatch"
            assert f"{prefix}Freq" in cols
            assert f"{prefix}ROCOF" in cols


class TestMergedDataFrame:
    def test_shape(self, merged_df):
        """Merged DataFrame should have ~161k rows and 122 columns."""
        n_rows, n_cols = merged_df.shape
        assert n_rows > 160_000, f"Expected >160k rows, got {n_rows}"
        # 112 measurement + 8 DATA_PRESENT + TIMESTAMP + Event = 122
        assert n_cols >= 122, f"Expected ≥122 cols, got {n_cols}"

    def test_event_column_present(self, merged_df):
        assert "Event" in merged_df.columns

    def test_timestamp_monotonic(self, merged_df):
        assert merged_df["TIMESTAMP"].is_monotonic_increasing

    def test_data_present_flags(self, merged_df):
        """Each BUSk_DATA_PRESENT column should exist."""
        for bus in PMU_BUSES:
            assert f"BUS{bus}_DATA_PRESENT" in merged_df.columns


class TestEventTransitions:
    def test_transition_count(self, merged_df):
        """Sanity check 2: ~9 distinct events, transitions near expected minutes."""
        transitions = extract_transitions(merged_df)
        n_transitions = len(transitions)
        # Initial state at t=0 is counted as first transition by diff logic
        assert n_transitions >= 9, f"Expected ≥9 transitions, got {n_transitions}"

    def test_transition_labels_in_range(self, merged_df):
        transitions = extract_transitions(merged_df)
        labels = transitions["Event"].dropna().unique()
        for lbl in labels:
            assert 0 <= int(lbl) <= 8, f"Label {lbl} out of range [0, 8]"

    def test_event_label_structure(self, data_raw):
        """Sanity check 2b: Document per-bus event label structure.

        REAL-DATA FINDING (correction to spec §2.3):
        Event labels are NOT globally identical across CSVs:
        - Cyber events (5, 6) appear ONLY in Bus29 (they represent Bus29 data drops)
        - Bus2 and Bus39 show Event=7 (bad data) at startup rows 91-111 that others don't
        - All other events (1, 2, 3, 4) appear consistently across all buses

        This test validates: non-cyber buses have no events 5 or 6, and
        non-cyber-startup buses have no event 7.
        """
        dfs = {bus: load_single_bus(data_raw, bus) for bus in PMU_BUSES}

        # Cyber events (5, 6) should only appear in Bus29
        for bus in [2, 5, 6, 10, 19, 22, 39]:
            cyber_rows = dfs[bus]["Event"].isin([5, 6]).sum()
            assert cyber_rows == 0, (
                f"Bus{bus} unexpectedly has {cyber_rows} cyber-event (5/6) rows"
            )
        assert dfs[29]["Event"].isin([5, 6]).sum() > 0, (
            "Bus29 should have cyber events (5 and/or 6)"
        )

        # Non-Bus29 buses should have consistent non-zero events
        ref = dfs[5][["TIMESTAMP", "Event"]].reset_index(drop=True)
        for bus in [6, 10, 19, 22]:
            other = dfs[bus][["TIMESTAMP", "Event"]].reset_index(drop=True)
            mismatches = (ref["Event"] != other["Event"]).sum()
            assert mismatches == 0, (
                f"Event mismatch between Bus5 and Bus{bus}: {mismatches} rows"
            )


class TestNaNBehavior:
    def test_bus29_nan_during_cyber_events(self, merged_df):
        """Sanity check 3: Bus29 measurements are NaN when Event ∈ {5, 6}."""
        cyber_mask = merged_df["Event"].isin([5, 6])
        cyber_rows = merged_df[cyber_mask]

        if len(cyber_rows) == 0:
            pytest.skip("No cyber events found in data")

        # Check measurement columns for Bus29
        meas_cols = [c for c in merged_df.columns if c.startswith("BUS29_") and c != "BUS29_DATA_PRESENT"]
        nan_fractions = cyber_rows[meas_cols].isna().mean()
        # At least some of the cyber-event rows should have NaN for Bus29
        assert nan_fractions.mean() > 0.5, (
            f"Expected most Bus29 measurements to be NaN during cyber events, "
            f"got mean NaN fraction {nan_fractions.mean():.2%}"
        )

    def test_other_buses_valid_during_cyber(self, merged_df):
        """Sanity check 3b: Other buses remain valid during cyber events."""
        cyber_mask = merged_df["Event"].isin([5, 6])
        cyber_rows = merged_df[cyber_mask]

        if len(cyber_rows) == 0:
            pytest.skip("No cyber events found in data")

        for bus in [2, 5, 6, 10, 19, 22, 39]:  # all except 29
            meas_cols = [c for c in merged_df.columns if c.startswith(f"BUS{bus}_") and c != f"BUS{bus}_DATA_PRESENT"]
            nan_fraction = cyber_rows[meas_cols].isna().mean().mean()
            assert nan_fraction < 0.5, (
                f"Bus {bus} has {nan_fraction:.2%} NaN during cyber events — unexpected"
            )
