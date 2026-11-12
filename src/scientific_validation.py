"""Scientific validation helpers (run AFTER real data are ingested).
  python -m src.scientific_validation --cases 1989-03-14 2003-10-30 2015-03-17 2024-05-11
      -> prints the catalogue row covering each date and writes outputs/figures/validation_<date>.png
  python -m src.scientific_validation --sensitivity
      -> storm counts for a grid of (release, merge_gap, min_core_hours) so the rule choices can be defended
Compare printed minima/timings with the Kyoto WDC monthly page by eye; do not trust code output blindly."""
from __future__ import annotations
import argparse, itertools
import pandas as pd
from .common import load_config, resolve
from .storm_detection import Params, detect_storms
from .plotting import plot_storm


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--cases", nargs="*"); ap.add_argument("--sensitivity", action="store_true")
    a = ap.parse_args(argv); cfg = load_config(a.config)
    clean = pd.read_csv(resolve(cfg, "clean_series"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
    base = Params.from_config(cfg)
    if a.cases:
        cat = pd.read_csv(resolve(cfg, "catalogue"), parse_dates=["start_time", "peak_time", "end_time"])
        for d in a.cases:
            t = pd.Timestamp(d); hit = cat[(cat.start_time <= t + pd.Timedelta(days=1)) & (cat.end_time >= t)]
            print(f"\n=== {d}: {len(hit)} storm(s)"); print(hit.T.to_string() if len(hit) else "no storm in catalogue")
            for r in hit.itertuples(): plot_storm(r, clean["dst_nt"], resolve(cfg, "figures_dir"), base.trigger_nt, base.release_nt, base.crossing_level_nt)
    if a.sensitivity:
        rows = []
        for rel, gap, minc in itertools.product([-15, -20, -25], [3, 6, 12], [1, 2]):
            p = Params(**{**base.__dict__, "release_nt": rel, "merge_gap_hours": gap, "min_core_hours": minc, "tiers": cfg["detection"]["tiers"]})
            c = detect_storms(clean["dst_nt"], p)
            rows.append({"release_nt": rel, "merge_gap_h": gap, "min_core_h": minc, "n_storms": len(c), "median_duration_h": c.total_duration_hours.median(),
                         "p95_duration_h": c.total_duration_hours.quantile(.95), "max_duration_h": c.total_duration_hours.max()})
        out = pd.DataFrame(rows); out.to_csv(resolve(cfg, "metrics_dir") / "detection_sensitivity.csv", index=False); print(out.to_string(index=False))


if __name__ == "__main__":
    main()
