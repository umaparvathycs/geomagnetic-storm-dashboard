import unittest
import numpy as np, pandas as pd
from src.cnn_dataset import hour_labels, make_windows, split_masks, build_dataset, class_distribution
from src.storm_detection import detect_storms


def series(n=24 * 60):
    rng = np.random.default_rng(1)
    x = np.zeros(n)
    for i in range(1, n): x[i] = 0.97 * x[i - 1] + rng.normal(0, 7)
    return pd.Series(np.round(x - 8), index=pd.date_range("2000-01-01", periods=n, freq="h"))


CFG = {"cnn": {"window_hours": 48, "label_position": "last", "scale_nt": 100.0,
               "split": {"train": ["2000-01-01", "2000-01-31"], "val": ["2000-02-01", "2000-02-14"], "test": ["2000-02-15", None]}}}


class T(unittest.TestCase):
    def setUp(self):
        self.s = series(); self.cat = detect_storms(self.s)

    def test_labels_come_from_catalogue(self):
        y = hour_labels(self.s.index, self.cat)
        inside = np.zeros(len(self.s), bool)
        for r in self.cat.itertuples(): inside |= (self.s.index >= r.start_time) & (self.s.index <= r.end_time)
        self.assertTrue((y == inside).all())

    def test_shapes_and_labels(self):
        d, s, y = build_dataset(self.s, self.cat, CFG)
        for k in d:
            self.assertEqual(d[k]["X"].shape[1:], (48, 1)); self.assertEqual(len(d[k]["X"]), len(d[k]["y"]))
        k = "train"; j = 10
        e = d[k]["end_idx"][j]
        np.testing.assert_allclose(d[k]["X"][j, :, 0] * 100, s.to_numpy()[e - 47:e + 1])
        self.assertEqual(d[k]["y"][j], y[e])                      # 'last' label = hour e

    def test_no_overlap_between_splits(self):
        d, s, y = build_dataset(self.s, self.cat, CFG)
        hours = {k: set() for k in d}
        for k in d:
            for e in d[k]["end_idx"]: hours[k].update(range(e - 47, e + 1))
        self.assertFalse(hours["train"] & hours["val"]); self.assertFalse(hours["val"] & hours["test"])
        self.assertFalse(hours["train"] & hours["test"])
        self.assertLess(d["train"]["end_time"].max(), d["val"]["end_time"].min())     # chronological
        self.assertLess(d["val"]["end_time"].max(), d["test"]["end_time"].min())

    def test_windows_with_nan_excluded(self):
        x = np.arange(100, dtype=float); x[50] = np.nan
        e, _ = make_windows(x, np.zeros(100, np.uint8), 10)
        for k in e: self.assertFalse(np.isnan(x[k - 9:k + 1]).any())
        self.assertNotIn(55, e); self.assertIn(60, e)

    def test_distribution(self):
        d, _, _ = build_dataset(self.s, self.cat, CFG)
        cd = class_distribution(d); self.assertTrue((cd.windows == cd.storm_windows + cd.non_storm_windows).all())


if __name__ == "__main__":
    unittest.main()
