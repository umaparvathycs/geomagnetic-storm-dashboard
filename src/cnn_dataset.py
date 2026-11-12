"""CNN dataset: labels derived from the verified catalogue.

Features:
- Hourly Dst input
- 48-hour sliding windows
- Catalogue-derived storm/non-storm labels
- Chronological train/validation/test split
- No sliding-window leakage between splits
- Fixed Dst scaling
- Timezone-safe split handling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def hour_labels(index: pd.DatetimeIndex, cat: pd.DataFrame) -> np.ndarray:
    """
    Create an hourly binary label.

    y[t] = 1 if timestamp t lies inside a catalogue storm interval
    [start_time, end_time], inclusive.
    """
    y = np.zeros(len(index), dtype=np.uint8)

    for r in cat.itertuples():
        start = pd.Timestamp(r.start_time)
        end = pd.Timestamp(r.end_time)

        # Make catalogue timestamps use the same timezone as the data.
        if index.tz is not None:
            if start.tzinfo is None:
                start = start.tz_localize(index.tz)
            else:
                start = start.tz_convert(index.tz)

            if end.tzinfo is None:
                end = end.tz_localize(index.tz)
            else:
                end = end.tz_convert(index.tz)

        i0 = index.searchsorted(start, side="left")
        i1 = index.searchsorted(end, side="right")

        y[i0:i1] = 1

    return y


def make_windows(
    x: np.ndarray,
    y: np.ndarray,
    W: int,
    label_position: str = "last",
):
    """
    Create valid sliding-window end indices and labels.

    W:
        Number of consecutive hours in each window.

    label_position='last':
        label of the final hour in the window.

    label_position='center':
        label of the centre hour.
    """
    if label_position not in {"last", "center"}:
        raise ValueError(
            "label_position must be either 'last' or 'center'."
        )

    n = len(x)

    if n < W:
        return np.array([], dtype=int), np.array([], dtype=np.uint8)

    # Detect NaN values.
    bad = np.isnan(x).astype(np.int32)

    # Cumulative count of missing values.
    csum = np.r_[0, np.cumsum(bad)]

    # Candidate window end positions.
    ends = np.arange(W - 1, n)

    # Keep only windows containing no missing values.
    ok = (
        csum[ends + 1] - csum[ends + 1 - W]
    ) == 0

    ends = ends[ok]

    # Select the hour whose label represents the window.
    if label_position == "last":
        offset = 0
    else:
        offset = W // 2

    window_labels = y[ends - offset]

    return ends, window_labels


def split_masks(
    index: pd.DatetimeIndex,
    ends: np.ndarray,
    W: int,
    split_cfg: dict,
) -> dict[str, np.ndarray]:
    """
    Create chronological split masks.

    A window belongs to a split only when ALL W hours of that
    window are inside that split.

    This prevents sliding-window leakage across train/validation/test
    boundaries.

    Handles UTC-aware data and timezone-naive config dates safely.
    """
    out = {}

    # The actual Dst data is UTC-aware.
    data_tz = index.tz

    def make_timestamp(value):
        ts = pd.Timestamp(value)

        if data_tz is not None:
            if ts.tzinfo is None:
                ts = ts.tz_localize(data_tz)
            else:
                ts = ts.tz_convert(data_tz)

        return ts

    # Start and end timestamps of every candidate window.
    first = index[ends - (W - 1)]
    last = index[ends]

    for name, (lo, hi) in split_cfg.items():

        # Lower boundary.
        if lo:
            lo_t = make_timestamp(lo)
        else:
            lo_t = index[0]

        # Upper boundary.
        if hi:
            hi_t = make_timestamp(hi)

            # Config dates are dates, so include the entire final day.
            hi_t = hi_t + pd.Timedelta(hours=23)
        else:
            hi_t = index[-1]

        # IMPORTANT:
        # Every hour in the window must belong to this split.
        out[name] = (
            (first >= lo_t)
            & (last <= hi_t)
        )

    return out


def build_dataset(
    series: pd.Series,
    cat: pd.DataFrame,
    cfg: dict,
):
    """
    Build the complete CNN dataset.

    Returns:
        data:
            Dictionary containing train/val/test arrays.

        s:
            Regular hourly Dst series.

        y:
            Hour-level storm labels.
    """

    c = cfg["cnn"]

    W = int(c["window_hours"])

    # Sort the series chronologically.
    s = series.sort_index()

    # Reindex onto a complete hourly grid.
    # Missing hours become NaN and are removed from windows.
    full_index = pd.date_range(
        s.index.min(),
        s.index.max(),
        freq="h",
        tz=s.index.tz,
    )

    s = s.reindex(full_index)

    x = s.to_numpy(dtype=np.float32)

    idx = s.index

    # Generate hourly storm labels from the verified catalogue.
    y = hour_labels(idx, cat)

    # Generate valid 48-hour windows.
    ends, window_labels = make_windows(
        x,
        y,
        W,
        c["label_position"],
    )

    # Apply chronological train/validation/test boundaries.
    masks = split_masks(
        idx,
        ends,
        W,
        c["split"],
    )

    # Efficient sliding-window view.
    view = sliding_window_view(x, W)

    data = {}

    for name, mask in masks.items():

        e = ends[mask]

        # Starting position of each selected window.
        starts = e - (W - 1)

        X = (
            view[starts]
            / np.float32(c["scale_nt"])
        ).astype(np.float32)

        # CNN expects:
        # (samples, time_steps, channels)
        X = X[..., None]

        data[name] = {
            "X": X,
            "y": window_labels[mask].astype(np.float32),
            "end_idx": e,
            "end_time": idx[e],
        }

    return data, s, y


def class_distribution(data: dict) -> pd.DataFrame:
    """
    Return class distribution for each dataset split.
    """

    rows = []

    for name, d in data.items():

        n = len(d["y"])

        positives = int(d["y"].sum())

        negatives = n - positives

        rows.append(
            {
                "split": name,
                "windows": n,
                "storm_windows": positives,
                "non_storm_windows": negatives,
                "storm_fraction": (
                    round(positives / n, 4)
                    if n
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)