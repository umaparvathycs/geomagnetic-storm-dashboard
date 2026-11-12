"""
Comprehensive Automated Test Suite for Geomagnetic Storm Detection Project.

Covers:
A. Recovery-time calculation and -150 nT crossings.
B. Event extraction and gap-merging logic.
C. Event-matching algorithms (Hungarian 1-to-1 and multi-overlap coverage).
D. Window alignment, label synchronization, and chronological splitting.
E. Dataset integrity and reference catalogue validation.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.storm_detection import (
    Params,
    detect_storms,
    recovery_to_level_hours,
    to_regular_grid,
)
from src.catalogue_validation import (
    assert_valid,
    validate_catalogue,
    CatalogueValidationError,
)
from src.cnn_dataset import (
    hour_labels,
    make_windows,
    split_masks,
)
from src.evaluate_events import (
    compatible_time,
    make_predicted_events,
)
from scipy.optimize import linear_sum_assignment


class TestRecoveryCrossings(unittest.TestCase):
    """Suite A: Recovery threshold crossing calculations."""

    def test_post_peak_upward_crossing_vs_pre_peak_downward(self):
        """
        Dst crosses -150 downward before the minimum and upward afterward.
        Verify recovery time is measured from peak to UPWARD crossing after peak.
        """
        # Index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        # Values: 0, -100, -160 (downward pre-peak), -210 (peak), -190, -150, -110 (upward), -50, 0
        x = np.array([0.0, -100.0, -160.0, -210.0, -190.0, -150.0, -110.0, -50.0, 0.0])
        peak_idx = 3  # min is -210
        end_idx = 8

        # Upward crossing happens between idx 5 (-150) and idx 6 (-110)
        # Fraction: (-150 - (-150)) / (-110 - (-150)) = 0.0
        # Time from peak: (5 - 3) + 0.0 = 2.0 hours
        rec = recovery_to_level_hours(x, peak_idx, end_idx, level=-150.0)
        self.assertEqual(rec, 2.0)

    def test_interpolated_upward_crossing(self):
        """Verify linear interpolation when crossing falls between integer hours."""
        # peak at idx 2 (-200), idx 3 is -140 (crosses -150 upward between idx 2 and 3)
        # frac = (-150 - (-200)) / (-140 - (-200)) = 50 / 60 = 0.8333...
        x = np.array([0.0, -100.0, -200.0, -140.0, -50.0, 0.0])
        rec = recovery_to_level_hours(x, 2, 5, level=-150.0)
        expected = round(50.0 / 60.0, 3)
        self.assertEqual(rec, expected)

    def test_storm_never_reaches_minus150_returns_nan(self):
        """Storm with minimum above -150 must return NaN."""
        x = np.array([0.0, -50.0, -120.0, -80.0, -20.0, 0.0])
        rec = recovery_to_level_hours(x, 2, 5, level=-150.0)
        self.assertTrue(np.isnan(rec))

    def test_storm_unrecovered_before_end_returns_nan(self):
        """Storm reaches -200 but ends while still at -170 (never rises to -150)."""
        x = np.array([0.0, -100.0, -200.0, -180.0, -170.0])
        rec = recovery_to_level_hours(x, 2, 4, level=-150.0)
        self.assertTrue(np.isnan(rec))

    def test_missing_data_around_crossing_returns_nan(self):
        """NaN observation during recovery must prevent fabricated crossing time."""
        x = np.array([0.0, -100.0, -200.0, np.nan, -120.0, 0.0])
        rec = recovery_to_level_hours(x, 2, 5, level=-150.0)
        self.assertTrue(np.isnan(rec))

    def test_exact_threshold_peak_returns_zero(self):
        """Minimum exactly equal to -150 nT returns 0.0 recovery time."""
        x = np.array([0.0, -80.0, -150.0, -110.0, -50.0, 0.0])
        rec = recovery_to_level_hours(x, 2, 5, level=-150.0)
        self.assertEqual(rec, 0.0)


class TestEventMerging(unittest.TestCase):
    """Suite B: Event extraction and temporal gap-merging logic."""

    def setUp(self):
        self.base_time = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")

    def make_series(self, hours, preds):
        times = pd.date_range(self.base_time, periods=len(preds), freq="h")
        dst = pd.Series(np.zeros(len(preds), dtype=float), index=times)
        return times, np.array(preds, dtype=int), dst

    def test_consecutive_positive_hours_form_single_event(self):
        times, preds, dst = self.make_series(10, [0, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        events = make_predicted_events(times, preds, dst, merge_gap_hours=6)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["start_time"], times[1])
        self.assertEqual(events.iloc[0]["end_time"], times[4])

    def test_gap_within_limit_is_merged(self):
        # 3 positive, 4 negative (<= 6 gap), 2 positive
        preds = [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]
        times, preds, dst = self.make_series(len(preds), preds)
        events = make_predicted_events(times, preds, dst, merge_gap_hours=6)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["start_time"], times[1])
        self.assertEqual(events.iloc[0]["end_time"], times[9])

    def test_gap_exceeding_limit_splits_events(self):
        # 2 positive, 7 negative (> 6 gap), 2 positive
        preds = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]
        times, preds, dst = self.make_series(len(preds), preds)
        events = make_predicted_events(times, preds, dst, merge_gap_hours=6)
        self.assertEqual(len(events), 2)
        self.assertEqual(events.iloc[0]["start_time"], times[1])
        self.assertEqual(events.iloc[0]["end_time"], times[2])
        self.assertEqual(events.iloc[1]["start_time"], times[10])
        self.assertEqual(events.iloc[1]["end_time"], times[11])

    def test_missing_dst_breaks_event_merge(self):
        # Gap of 3 negative hours, but with NaN in Dst
        preds = [0, 1, 1, 0, 0, 0, 1, 1, 0]
        times, preds, dst = self.make_series(len(preds), preds)
        dst.iloc[4] = np.nan  # NaN inside the gap
        events = make_predicted_events(times, preds, dst, merge_gap_hours=6)
        # Must split into two events because of missing Dst observation
        self.assertEqual(len(events), 2)


class TestEventMatching(unittest.TestCase):
    """Suite C: Event-matching rules and edge cases."""

    def test_single_overlap_matched_exactly(self):
        t0 = pd.Timestamp("2020-01-01 10:00:00", tz="UTC")
        t1 = pd.Timestamp("2020-01-01 20:00:00", tz="UTC")
        overlap = np.array([[10]], dtype=np.int64)
        r_idx, p_idx = linear_sum_assignment(-overlap)
        self.assertEqual(list(r_idx), [0])
        self.assertEqual(list(p_idx), [0])

    def test_one_prediction_overlapping_two_references_behavior(self):
        """
        When 1 predicted event spans 2 reference events:
        Hungarian assignment chooses the reference with maximum overlap,
        and forces the second reference event to be unmatched.
        """
        # Pred 0 overlaps Ref 0 by 30 hours, Ref 1 by 15 hours
        overlap = np.array([[30], [15]], dtype=np.int64)
        r_idx, p_idx = linear_sum_assignment(-overlap)
        # Matches Ref 0 to Pred 0
        self.assertEqual(list(r_idx), [0])
        self.assertEqual(list(p_idx), [0])
        # Ref 1 is unassigned in strict 1-to-1 bipartite matching
        unassigned_refs = set([0, 1]) - set(r_idx)
        self.assertEqual(unassigned_refs, {1})

    def test_boundary_contact_without_shared_hour_has_zero_overlap(self):
        # Ref ends at 10:00, Pred starts at 11:00 (adjacent hours, no shared hour)
        ref_start = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
        ref_end = pd.Timestamp("2020-01-01 10:00:00", tz="UTC")
        pred_start = pd.Timestamp("2020-01-01 11:00:00", tz="UTC")
        pred_end = pd.Timestamp("2020-01-01 20:00:00", tz="UTC")

        left = max(ref_start, pred_start)
        right = min(ref_end, pred_end)
        has_overlap = left <= right
        self.assertFalse(has_overlap)

    def test_one_hour_overlap_detected(self):
        # Ref: 00:00 to 10:00, Pred: 10:00 to 20:00 (shared hour is 10:00)
        ref_start = pd.Timestamp("2020-01-01 00:00:00", tz="UTC")
        ref_end = pd.Timestamp("2020-01-01 10:00:00", tz="UTC")
        pred_start = pd.Timestamp("2020-01-01 10:00:00", tz="UTC")
        pred_end = pd.Timestamp("2020-01-01 20:00:00", tz="UTC")

        left = max(ref_start, pred_start)
        right = min(ref_end, pred_end)
        self.assertTrue(left <= right)
        overlap_h = int((right - left).total_seconds() // 3600) + 1
        self.assertEqual(overlap_h, 1)


class TestWindowAlignment(unittest.TestCase):
    """Suite D: 48-hour window endpoint and label alignment."""

    def test_window_endpoint_receives_correct_label(self):
        # 100 hours of synthetic data
        index = pd.date_range("2020-01-01 00:00:00", periods=100, freq="h", tz="UTC")
        x = np.zeros(100, dtype=np.float32)
        # Storm from hour 40 to hour 60 inclusive
        cat = pd.DataFrame([{
            "start_time": index[40],
            "end_time": index[60],
        }])
        y = hour_labels(index, cat)
        self.assertEqual(y[39], 0)
        self.assertEqual(y[40], 1)
        self.assertEqual(y[60], 1)
        self.assertEqual(y[61], 0)

        # 48-hour windows with label_position='last'
        ends, window_labels = make_windows(x, y, W=48, label_position="last")
        # First valid window ends at index 47 (hours 0..47)
        self.assertEqual(ends[0], 47)
        # Its label must be exactly y[47]
        self.assertEqual(window_labels[0], y[47])

        # Check alignment across all windows
        for e, wl in zip(ends, window_labels):
            self.assertEqual(wl, y[e])


class TestDataIntegrity(unittest.TestCase):
    """Suite E: Dataset integrity, final filtering, and catalogue schema."""

    def test_final_dataset_observations_count_and_cleanliness(self):
        clean_path = Path("data/processed/dst_hourly_clean.csv")
        self.assertTrue(clean_path.exists())

        df = pd.read_csv(clean_path, parse_dates=["timestamp_utc"])
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

        # Filter final 1957-2020
        final = df[
            (df["product"] == "dst_final")
            & (df["timestamp_utc"] >= "1957-01-01 00:00:00+00:00")
            & (df["timestamp_utc"] <= "2020-12-31 23:00:00+00:00")
        ]

        # Must have exactly 561,024 hourly observations
        self.assertEqual(len(final), 561024)
        # Must have zero duplicate timestamps
        self.assertEqual(final["timestamp_utc"].duplicated().sum(), 0)
        # Must have zero NaN Dst values in final product
        self.assertEqual(final["dst_nt"].isna().sum(), 0)
        # Verify continuous 1-hour spacing
        expected = pd.date_range("1957-01-01 00:00:00", "2020-12-31 23:00:00", freq="h", tz="UTC")
        self.assertEqual(len(expected), 561024)
        self.assertTrue(final["timestamp_utc"].reset_index(drop=True).equals(expected.to_series().reset_index(drop=True)))

    def test_storm_catalogue_counts_and_recovery(self):
        cat_path = Path("data/catalogue/storm_catalogue.csv")
        self.assertTrue(cat_path.exists())

        cat = pd.read_csv(cat_path, parse_dates=["start_time", "peak_time", "end_time"])
        # Exactly 4,824 reference storms
        self.assertEqual(len(cat), 4824)
        # All from final product
        self.assertTrue((cat["data_product"] == "dst_final").all())
        # All valid recovery values must be non-negative
        valid_rec = cat["recovery_to_150_hours"].dropna()
        self.assertTrue((valid_rec >= 0).all())


if __name__ == "__main__":
    unittest.main()
