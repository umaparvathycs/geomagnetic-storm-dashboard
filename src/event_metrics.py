"""Event-level comparison of hourly predictions against the deterministic catalogue (numpy/pandas only)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def segments(pred: np.ndarray, gap: int) -> list[tuple[int, int]]:
    idx = np.flatnonzero(pred > 0)
    if idx.size == 0:
        return []
    segs, a, b = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - b - 1 <= gap:
            b = i
        else:
            segs.append((a, b)); a = b = i
    segs.append((a, b))
    return segs


def event_metrics(index: pd.DatetimeIndex, pred: np.ndarray, known: np.ndarray, cat: pd.DataFrame, gap: int = 6):
    """pred: 0/1 per hour of `index` (0 where no prediction exists). known: bool, hours that HAVE a prediction.
    Only catalogue storms lying completely inside the evaluated hours are scored."""
    t_ok = index[known]
    lo, hi = t_ok.min(), t_ok.max()
    c = cat[(pd.to_datetime(cat.start_time) >= lo) & (pd.to_datetime(cat.end_time) <= hi)]
    segs = segments(pred, gap)
    truth = np.zeros(len(index), bool)
    for r in c.itertuples():
        truth[index.searchsorted(pd.Timestamp(r.start_time)): index.searchsorted(pd.Timestamp(r.end_time), "right")] = True
    rows = []
    for r in c.itertuples():
        i0, i1 = index.searchsorted(pd.Timestamp(r.start_time)), index.searchsorted(pd.Timestamp(r.end_time), "right")
        p = pred[i0:i1] > 0
        cov = float(p.mean()) if i1 > i0 else 0.0
        delay = float(np.argmax(p)) if p.any() else np.nan
        rows.append({"storm_id": r.storm_id, "intensity_tier": r.intensity_tier, "min_dst_nt": r.min_dst_nt,
                     "coverage": round(cov, 3), "detected_any": bool(p.any()), "detected_cov50": cov >= 0.5,
                     "onset_delay_hours": delay})
    per = pd.DataFrame(rows)
    fa = sum(1 for a, b in segs if not truth[a:b + 1].any())
    out = {"n_storms_scored": int(len(c)), "n_predicted_segments": len(segs), "false_alarm_segments": int(fa),
           "event_recall_any_overlap": float(per.detected_any.mean()) if len(per) else float("nan"),
           "event_recall_coverage>=50%": float(per.detected_cov50.mean()) if len(per) else float("nan"),
           "event_precision": float(1 - fa / len(segs)) if segs else float("nan"),
           "median_onset_delay_hours": float(np.nanmedian(per.onset_delay_hours)) if len(per) and per.detected_any.any() else float("nan")}
    return out, per
