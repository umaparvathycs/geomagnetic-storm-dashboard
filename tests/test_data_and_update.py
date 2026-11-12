import copy, json, tempfile, unittest
from pathlib import Path
import numpy as np, pandas as pd
from src.common import load_config
from src.data_cleaning import parse_wdc_text, clean_series, validate_clean
from src.event_metrics import event_metrics, segments
from src.update_catalogue import run_update


def synth(start="2000-01-01", n=24 * 400, seed=3):
    rng = np.random.default_rng(seed); x = np.zeros(n)
    for i in range(1, n): x[i] = 0.97 * x[i - 1] + rng.normal(0, 7)
    x = np.round(x - 8)
    return pd.Series(x, index=pd.date_range(start, periods=n, freq="h"))


def fake_wdc_line(y, m, d, vals, base=0):          # SELF-CONSISTENCY test of the parser's assumed layout only
    head = f"DST{y % 100:02d}{m:02d}*{d:02d}RRX0{y // 100:02d}{base:4d}"
    return head + "".join(f"{v:4d}" for v in vals) + f"{int(np.mean(vals)):4d}"


class TestData(unittest.TestCase):
    def test_wdc_parser_layout_and_missing(self):
        vals = list(range(-5, 19)); vals[3] = 9999
        txt = "header\n" + fake_wdc_line(1989, 3, 14, vals) + "\n"
        df = parse_wdc_text(txt, "dst_final", "t")
        self.assertEqual(len(df), 24); self.assertEqual(df.timestamp_utc.iloc[0], pd.Timestamp("1989-03-14 00:00", tz="UTC"))
        self.assertEqual(df.dst_raw.iloc[3], 9999); self.assertEqual(df.dst_raw.iloc[0], -5)

    def test_cleaning_no_interpolation_and_gaps(self):
        cfg = load_config(); s = synth(n=200)
        raw = pd.DataFrame({"timestamp_utc": s.index, "dst_raw": s.values, "product": "dst_final", "source": "x"})
        raw.loc[50:55, "dst_raw"] = 9999
        raw = pd.concat([raw, raw.iloc[[10]]])                              # duplicate timestamp
        raw = raw.drop(index=raw.index[100:103])                            # missing rows
        clean, rep = clean_series(raw.reset_index(drop=True), cfg)
        self.assertEqual(validate_clean(clean, cfg), [])
        self.assertEqual(rep["interpolated_hours"], 0); self.assertEqual(clean.dst_nt.isna().sum(), 6 + 3)
        self.assertEqual(len(clean), 200)
        # raw values untouched (9999 still in raw)
        self.assertTrue((raw.dst_raw == 9999).sum() == 6)

    def test_event_metrics(self):
        idx = pd.date_range("2000-01-01", periods=100, freq="h")
        cat = pd.DataFrame({"storm_id": ["a", "b"], "start_time": [idx[10], idx[60]], "end_time": [idx[30], idx[80]],
                            "intensity_tier": "Weak", "min_dst_nt": -40.0})
        pred = np.zeros(100, np.uint8); pred[12:30] = 1; pred[90:95] = 1          # hits storm a, misses b, one false alarm
        ev, per = event_metrics(idx, pred, np.ones(100, bool), cat, gap=6)
        self.assertEqual(ev["n_storms_scored"], 2); self.assertAlmostEqual(ev["event_recall_any_overlap"], 0.5)
        self.assertEqual(ev["false_alarm_segments"], 1); self.assertEqual(per.onset_delay_hours.iloc[0], 2)
        self.assertEqual(segments(np.array([1, 0, 0, 1]), 2), [(0, 3)])


class TestLivingCatalogue(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.cfg = load_config(); self.cfg["_root"] = str(self.tmp)
        self.s = synth(); df = pd.DataFrame({"timestamp_utc": self.s.index.strftime("%Y-%m-%d %H:%M:%S"), "dst_nt": self.s.values})
        df.loc[500:520, "dst_nt"] = 9999
        self.part1 = self.tmp / "p1.csv"; df.iloc[:24 * 300].to_csv(self.part1, index=False)
        self.part2 = self.tmp / "p2.csv"; df.iloc[24 * 300:].to_csv(self.part2, index=False)

    def tearDown(self):
        import shutil
        if hasattr(self, "tmp") and self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_update_flow(self):
        cat1, d1 = run_update(self.cfg, csv=str(self.part1), product="dst_final")
        self.assertTrue(cat1.storm_id.is_unique); n1 = len(cat1)
        cat2, d2 = run_update(self.cfg, csv=str(self.part2), product="dst_final")
        self.assertGreater(len(cat2), n1 - 1)
        self.assertTrue(cat2.storm_id.is_unique)
        self.assertGreater(len(d2["new"]), 0)
        pngs = list((self.tmp / "outputs/plots").glob("GS_*.png")); self.assertEqual(len(pngs), len(cat2))
        # re-running with nothing new: no redraw, nothing new/changed
        _, d3 = run_update(self.cfg, csv=str(self.part2), product="dst_final")
        self.assertEqual((d3["new"], d3["changed"], d3["removed"]), ([], [], []))
        self.assertTrue((self.tmp / "data/catalogue/update_log.jsonl").exists())

    def test_protected_storms_abort_on_rule_change(self):
        run_update(self.cfg, csv=str(self.part1), product="dst_final")
        bad = copy.deepcopy(self.cfg); bad["detection"]["release_nt"] = -5.0
        with self.assertRaises(SystemExit): run_update(bad, csv=str(self.part1), product="dst_final")


if __name__ == "__main__":
    unittest.main()
