"""Storm plots generated ONLY from the final catalogue + clean series (no separately computed boundaries).
Incremental: a manifest (row hash per storm_id) lets the update pipeline redraw only new/changed storms."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIER_COLOR = {"Weak": "#4daf4a", "Moderate": "#ff7f00", "Intense": "#e41a1c", "Severe": "#984ea3", "Extreme": "#000000"}


def row_hash(row: pd.Series) -> str:
    return hashlib.sha1("|".join(str(v) for v in row.values).encode()).hexdigest()[:12]


def plot_storm(row, series: pd.Series, out_dir: Path, trigger=-30.0, release=-20.0, level=-150.0, pad_days=2) -> Path:
    t0, tp, t1 = pd.Timestamp(row.start_time), pd.Timestamp(row.peak_time), pd.Timestamp(row.end_time)
    sub = series[t0 - pd.Timedelta(days=pad_days): t1 + pd.Timedelta(days=pad_days)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(sub.index, sub.values, color="black", lw=1.2, label="Hourly Dst")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.axhline(trigger, color="orange", ls="--", lw=1, label=f"Trigger ({trigger:.0f} nT)")
    ax.axhline(release, color="tab:cyan", ls="--", lw=0.8, label=f"Release ({release:.0f} nT)")
    ax.axvspan(t0, tp, color="red", alpha=0.10, label=f"Main phase ({row.main_phase_hours:.0f} h)")
    ax.axvspan(tp, t1, color="blue", alpha=0.10, label=f"Recovery phase ({row.recovery_phase_hours:.0f} h)")
    ax.scatter([t0], [row.onset_dst_nt], c="green", s=70, zorder=5, label=f"Start {t0:%Y-%m-%d %H:%M}")
    ax.scatter([tp], [row.min_dst_nt], c="red", marker="^", s=100, zorder=5,
               label=f"Peak {row.min_dst_nt:.0f} nT {tp:%Y-%m-%d %H:%M}")
    end_val = series.get(t1, np.nan)
    ax.scatter([t1], [end_val], c="blue", marker="s", s=70, zorder=5, label=f"End {t1:%Y-%m-%d %H:%M}")
    r = row.recovery_to_150_hours
    if row.min_dst_nt <= level:
        ax.axhline(level, color="purple", ls="-.", lw=0.9, label=f"{level:.0f} nT level")
        if pd.notna(r):
            tc = tp + pd.Timedelta(hours=float(r))
            ax.scatter([tc], [level], c="purple", marker="D", s=70, zorder=6, label=f"Post-peak crossing +{r:.1f} h")
    flag = "" if row.data_quality_flag == "ok" else f" | flag: {row.data_quality_flag}"
    ax.set_title(f"{row.storm_id} | {row.intensity_tier} | Min Dst {row.min_dst_nt:.0f} nT{flag}", fontsize=10, fontweight="bold",
                 color=TIER_COLOR.get(row.intensity_tier, "black"))
    ax.set_xlabel("Time (UTC)"); ax.set_ylabel("Dst (nT)"); ax.grid(True, ls="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.autofmt_xdate(); fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{row.storm_id}.png"
    fig.savefig(path, dpi=110); plt.close(fig)
    return path


def plot_catalogue(cat: pd.DataFrame, series: pd.Series, out_dir: Path, params=None, only_ids=None, force=False) -> list[str]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    man_p = out_dir / ".manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    kw = {}
    if params is not None:
        kw = dict(trigger=params.trigger_nt, release=params.release_nt, level=params.crossing_level_nt)
    done = []
    for row in cat.itertuples(index=False):
        if only_ids is not None and row.storm_id not in only_ids:
            continue
        h = row_hash(pd.Series(row._asdict()))
        if not force and man.get(row.storm_id) == h and (out_dir / f"{row.storm_id}.png").exists():
            continue
        plot_storm(row, series, out_dir, **kw); man[row.storm_id] = h; done.append(row.storm_id)
    # drop plots of storms that no longer exist in the catalogue (e.g. an open storm that changed ID)
    live = set(cat.storm_id)
    for pid in list(man):
        if pid not in live:
            (out_dir / f"{pid}.png").unlink(missing_ok=True); man.pop(pid)
    man_p.write_text(json.dumps(man))
    return done


def main(argv=None):
    import argparse
    from .common import load_config, resolve
    from .storm_detection import Params
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--ids", nargs="*")
    a = ap.parse_args(argv); cfg = load_config(a.config)
    cat = pd.read_csv(resolve(cfg, "catalogue"), parse_dates=["start_time", "peak_time", "end_time"])
    ser = pd.read_csv(resolve(cfg, "clean_series"), parse_dates=["timestamp_utc"]).set_index("timestamp_utc")["dst_nt"]
    p = Params.from_config(cfg)
    done = plot_catalogue(cat, ser, resolve(cfg, "plots_dir"), p, set(a.ids) if a.ids else None, a.force)
    print(f"{len(done)} plots written to {resolve(cfg, 'plots_dir')}")


if __name__ == "__main__":
    main()
