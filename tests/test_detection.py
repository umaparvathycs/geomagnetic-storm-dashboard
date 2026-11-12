import unittest
import numpy as np, pandas as pd
from src.storm_detection import detect_storms, Params, recovery_to_level_hours
from src.catalogue_validation import validate_catalogue, assert_valid, CatalogueValidationError
from src.common import classify_intensity


def mk(vals, start="2000-01-01 00:00"):
    return pd.Series(np.array(vals, float), index=pd.date_range(start, periods=len(vals), freq="h"))


def storm(pre=10, down=(-10, -35, -90, -160, -210), up=(-190, -150, -110, -70, -40, -25, -15, -5), post=10):
    return [0] * pre + list(down) + list(up) + [0] * post


class TestDetection(unittest.TestCase):
    def test_single_storm_boundaries(self):
        s = mk(storm())
        c = detect_storms(s)
        self.assertEqual(len(c), 1)
        r = c.iloc[0]
        # onset = local max before descent (last 0 at index 9), peak = -210 (index 14), end = first >= -20 -> -15 (index 21)
        self.assertEqual(r.start_time, s.index[9]); self.assertEqual(r.peak_time, s.index[14])
        self.assertEqual(r.end_time, s.index[21]); self.assertEqual(r.min_dst_nt, -210)
        self.assertEqual(r.intensity_tier, "Intense")
        self.assertEqual(r.total_duration_hours, r.main_phase_hours + r.recovery_phase_hours)
        assert_valid(c, s)

    def test_minus150_uses_post_peak_crossing(self):
        # first -150 crossing (down) happens BEFORE the peak; metric must use the later UPWARD crossing
        s = mk(storm()); c = detect_storms(s); r = c.iloc[0]
        # peak idx 14 (-210); idx15=-190, idx16=-150 (<= -150), idx17=-110 -> upward crossing between 16 and 17: frac=0
        self.assertAlmostEqual(r.recovery_to_150_hours, 2.0, places=6)
        self.assertGreater(r.recovery_to_150_hours, 0)
        self.assertLessEqual(r.recovery_to_150_hours, r.recovery_phase_hours)
        self.assertNotEqual(r.recovery_to_150_hours, r.recovery_phase_hours)

    def test_minus150_interpolated(self):
        x = np.array([0, -100, -200, -140, -50, 0.0])
        self.assertAlmostEqual(recovery_to_level_hours(x, 2, 5), (-150 + 200) / 60, places=3)

    def test_minus150_not_reached_is_nan(self):
        c = detect_storms(mk(storm(down=(-10, -35, -60, -90, -120), up=(-100, -70, -40, -25, -15))))
        self.assertTrue(np.isnan(c.iloc[0].recovery_to_150_hours))

    def test_double_dip_merged_and_peak_is_global_min(self):
        v = [0] * 5 + [-20, -40, -80, -45, -25, -40, -120, -60, -15, 0] + [0] * 5   # gap of 2 non-trigger hours
        c = detect_storms(mk(v))
        self.assertEqual(len(c), 1); self.assertEqual(c.iloc[0].min_dst_nt, -120)

    def test_separate_storms_no_overlap_unique_ids(self):
        v = [0] * 5 + [-40, -80, -40, -10, 0, 0, 0, 0, 0, 0, 0, 0] + [-35, -70, -35, -10] + [0] * 5
        c = detect_storms(mk(v)); self.assertEqual(len(c), 2)
        self.assertLess(c.end_time.iloc[0], c.start_time.iloc[1]); self.assertTrue(c.storm_id.is_unique)
        assert_valid(c, mk(v))

    def test_split_without_recovery(self):
        # second storm starts before Dst ever returns to -20: boundary at max between them
        v = [0] * 3 + [-40, -90, -50, -26, -25, -24, -23, -22, -21, -22, -23, -24, -60, -100, -50, -10, 0, 0]
        c = detect_storms(mk(v)); self.assertEqual(len(c), 2)
        self.assertIn("split_without_recovery", c.iloc[0].data_quality_flag)
        self.assertLess(c.end_time.iloc[0], c.start_time.iloc[1])
        assert_valid(c, mk(v))

    def test_gap_and_open_end(self):
        v = [0] * 3 + [-40, -90, -50] + [np.nan] * 5 + [-10, 0]
        c = detect_storms(mk(v)); self.assertIn("end_truncated_by_gap", c.iloc[0].data_quality_flag)
        v2 = [0] * 3 + [-40, -90, -50, -45]
        c2 = detect_storms(mk(v2)); self.assertIn("open_end", c2.iloc[0].data_quality_flag)

    def test_nan_is_not_a_storm_and_no_merge_over_gap(self):
        v = [0] * 3 + [-40, -50] + [np.nan] * 2 + [-45, -40, 0, 0]
        self.assertEqual(len(detect_storms(mk(v))), 2)

    def test_tier_boundaries_non_overlapping(self):
        exp = {-30: "Weak", -49: "Weak", -50: "Moderate", -99: "Moderate", -100: "Intense", -249: "Intense",
               -250: "Severe", -349: "Severe", -350: "Extreme", -589: "Extreme", -29: "None"}
        for k, v in exp.items():
            self.assertEqual(classify_intensity(k), v, k)

    def test_validator_catches_tampering(self):
        s = mk(storm()); c = detect_storms(s)
        bad = c.copy(); bad.loc[0, "min_dst_nt"] = -100.0
        with self.assertRaises(CatalogueValidationError): assert_valid(bad, s)
        bad = c.copy(); bad.loc[0, "recovery_to_150_hours"] = -3.0
        self.assertTrue(validate_catalogue(bad, s))
        dup = pd.concat([c, c], ignore_index=True)
        self.assertTrue(any("duplicate" in p or "overlap" in p for p in validate_catalogue(dup, s)))

    def test_random_series_invariants(self):
        rng = np.random.default_rng(0)
        x = np.zeros(20000)
        for i in range(1, len(x)): x[i] = 0.97 * x[i - 1] + rng.normal(0, 7)   # mean-reverting Dst-like series
        x = np.round(x - 8)
        s = mk(x); c = detect_storms(s)
        self.assertGreater(len(c), 20); assert_valid(c, s)


if __name__ == "__main__":
    unittest.main()
