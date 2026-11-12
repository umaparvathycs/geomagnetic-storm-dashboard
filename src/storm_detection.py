"""
Deterministic geomagnetic storm detection on hourly Dst data.

Detection rules
---------------

1. CORE
   A storm core is a consecutive run of Dst <= trigger_nt.

2. MERGE
   Two cores are merged when the gap between them is <= merge_gap_hours
   and there is no missing Dst value in the separating interval.

3. MINIMUM CORE
   An event must contain at least min_core_hours trigger hours.

4. ONSET
   Starting from the first core hour, move backward while the Dst value
   is strictly increasing toward the core:

       Dst[i-1] > Dst[i]

   The first point where this condition stops becomes the storm start.

5. END
   The event ends at the first hour after the core where:

       Dst >= release_nt

   If another core begins before recovery, the preceding event ends at the
   maximum Dst value between the two cores.

6. MISSING DATA
   Missing observations are never treated as storm observations.

7. RECOVERY-TO-150
   The metric is measured only AFTER the storm minimum and only for the
   upward crossing of -150 nT.

8. STORM ID
   IDs are generated from the final start timestamp:

       GS_YYYYMMDD_HH
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import classify_intensity


PRODUCT_RANK = {
    "dst_final": 0,
    "dst_provisional": 1,
    "dst_realtime": 2,
}


@dataclass(frozen=True)
class Params:
    trigger_nt: float = -30.0
    release_nt: float = -20.0
    merge_gap_hours: int = 6
    min_core_hours: int = 1
    max_lookback_hours: int = 24
    crossing_level_nt: float = -150.0

    tiers: dict | None = field(
        default=None,
        hash=False,
        compare=False,
    )

    @staticmethod
    def from_config(cfg: dict) -> "Params":
        """Create detection parameters from config.yaml."""

        if "detection" not in cfg:
            raise KeyError(
                "config.yaml is missing the 'detection' section."
            )

        d = cfg["detection"]

        required = [
            "trigger_nt",
            "release_nt",
            "merge_gap_hours",
            "min_core_hours",
            "max_lookback_hours",
            "crossing_level_nt",
            "tiers",
        ]

        missing = [k for k in required if k not in d]

        if missing:
            raise KeyError(
                "Missing detection configuration values: "
                + ", ".join(missing)
            )

        tiers = d["tiers"]

        if not isinstance(tiers, dict):
            raise TypeError(
                "detection.tiers must be a dictionary."
            )

        for name, bounds in tiers.items():

            if not isinstance(bounds, (list, tuple)):
                raise TypeError(
                    f"detection.tiers.{name} must be "
                    "[lower, upper]."
                )

            if len(bounds) != 2:
                raise ValueError(
                    f"detection.tiers.{name} must contain "
                    "exactly two values."
                )

            lo, hi = bounds

            if lo is not None and not isinstance(lo, (int, float)):
                raise TypeError(
                    f"detection.tiers.{name}[0] must be numeric or null."
                )

            if not isinstance(hi, (int, float)):
                raise TypeError(
                    f"detection.tiers.{name}[1] must be numeric."
                )

        return Params(
            trigger_nt=float(d["trigger_nt"]),
            release_nt=float(d["release_nt"]),
            merge_gap_hours=int(d["merge_gap_hours"]),
            min_core_hours=int(d["min_core_hours"]),
            max_lookback_hours=int(d["max_lookback_hours"]),
            crossing_level_nt=float(d["crossing_level_nt"]),
            tiers=tiers,
        )


COLUMNS = [
    "storm_id",
    "start_time",
    "peak_time",
    "end_time",
    "onset_dst_nt",
    "min_dst_nt",
    "intensity_tier",
    "total_duration_hours",
    "main_phase_hours",
    "recovery_phase_hours",
    "descent_rate_nt_hr",
    "recovery_to_150_hours",
    "data_quality_flag",
    "data_product",
]


def to_regular_grid(s: pd.Series) -> pd.Series:
    """
    Return a sorted hourly grid with missing hours represented by NaN.
    """

    s = s[
        ~s.index.duplicated(keep="first")
    ].sort_index()

    if s.empty:
        return s

    grid = pd.date_range(
        s.index.min(),
        s.index.max(),
        freq="h",
    )

    return s.reindex(grid)


def _runs(mask: np.ndarray):
    """Return inclusive start/end indexes of True runs."""

    idx = np.flatnonzero(mask)

    if idx.size == 0:
        return []

    breaks = np.flatnonzero(
        np.diff(idx) > 1
    )

    starts = np.r_[
        idx[0],
        idx[breaks + 1],
    ]

    ends = np.r_[
        idx[breaks],
        idx[-1],
    ]

    return list(
        zip(
            starts.tolist(),
            ends.tolist(),
        )
    )


def recovery_to_level_hours(
    x: np.ndarray,
    peak: int,
    end: int,
    level: float = -150.0,
) -> float:
    """
    Calculate time from the storm minimum to the first upward
    crossing of the specified recovery level.

    IMPORTANT:

    - The downward crossing before the minimum is NEVER used.
    - Only observations after the minimum are examined.
    - If the minimum is exactly equal to the recovery level,
      the recovery time is defined as 0 hours.
    - If the minimum is above the recovery level, NaN is returned.
    - If the storm goes below the recovery level but never
      recovers through it before the event ends, NaN is returned.
    """

    if not np.isfinite(x[peak]):
        return float("nan")

    # Storm never reached the recovery level.
    if x[peak] > level:
        return float("nan")

    # Exact boundary case:
    # minimum is already at the recovery level.
    if np.isclose(x[peak], level, atol=1e-9):
        return 0.0

    # Search ONLY after the minimum.
    for j in range(peak + 1, end + 1):

        if np.isnan(x[j]) or np.isnan(x[j - 1]):
            return float("nan")

        previous = x[j - 1]
        current = x[j]

        # Upward crossing:
        #
        # previous <= level < current
        #
        if previous <= level and current > level:

            denominator = current - previous

            if denominator == 0:
                return float("nan")

            fraction = (
                (level - previous)
                / denominator
            )

            return round(
                (j - 1 - peak) + fraction,
                3,
            )

    return float("nan")


def detect_storms(
    series: pd.Series,
    params: Params = Params(),
    product: pd.Series | None = None,
) -> pd.DataFrame:
    """Detect geomagnetic storms from an hourly Dst series."""

    s = to_regular_grid(
        series.astype(float)
    )

    if s.empty:
        return pd.DataFrame(
            columns=COLUMNS
        )

    x = s.to_numpy()
    t = s.index
    n = len(x)

    # --------------------------------------------------------
    # Product information
    # --------------------------------------------------------

    prod = None

    if product is not None:

        prod = (
            product
            .reindex(t)
            .map(PRODUCT_RANK)
            .to_numpy(dtype=float)
        )

    # --------------------------------------------------------
    # CORE DETECTION
    # --------------------------------------------------------

    below = np.zeros(
        n,
        dtype=bool,
    )

    valid = ~np.isnan(x)

    below[valid] = (
        x[valid] <= params.trigger_nt
    )

    # --------------------------------------------------------
    # MERGE CORES
    # --------------------------------------------------------

    merged = []

    for a, b in _runs(below):

        if merged:

            previous_a, previous_b = merged[-1]

            gap = (
                a
                - previous_b
                - 1
            )

            if (
                gap <= params.merge_gap_hours
                and not np.isnan(
                    x[previous_b + 1:a]
                ).any()
            ):

                merged[-1] = (
                    previous_a,
                    b,
                )

                continue

        merged.append(
            (a, b)
        )

    # --------------------------------------------------------
    # MINIMUM CORE LENGTH
    # --------------------------------------------------------

    cores = []

    for a, b in merged:

        core_hours = int(
            (
                x[a:b + 1]
                <= params.trigger_nt
            ).sum()
        )

        if core_hours >= params.min_core_hours:

            cores.append(
                (a, b)
            )

    # --------------------------------------------------------
    # BUILD EVENTS
    # --------------------------------------------------------

    rows = []

    previous_end = -1

    for k, (a, b) in enumerate(cores):

        next_a = (
            cores[k + 1][0]
            if k + 1 < len(cores)
            else None
        )

        flags = []

        # ----------------------------------------------------
        # ONSET
        # ----------------------------------------------------

        lower_bound = previous_end + 1

        floor = max(
            lower_bound,
            a - params.max_lookback_hours,
            0,
        )

        start_index = a

        while (
            start_index - 1 >= floor
            and not np.isnan(
                x[start_index - 1]
            )
            and x[start_index - 1] > x[start_index]
        ):
            start_index -= 1

        previous_index = start_index - 1

        if previous_index < 0:

            flags.append(
                "start_at_data_edge"
            )

        elif (
            previous_index >= lower_bound
            and np.isnan(x[previous_index])
        ):

            flags.append(
                "onset_truncated_by_gap"
            )

        elif (
            start_index == floor
            and (
                a - start_index
                >= params.max_lookback_hours
            )
        ):

            flags.append(
                "onset_lookback_capped"
            )

        # ----------------------------------------------------
        # PEAK / MINIMUM
        # ----------------------------------------------------

        peak_index = (
            a
            + int(
                np.argmin(
                    x[a:b + 1]
                )
            )
        )

        # A legitimate edge case:
        #
        # The first storm/core hour can already be the
        # minimum Dst of the event.
        #
        # Therefore start_time == peak_time is allowed,
        # but it must be explicitly flagged so the validator
        # knows this is intentional.
        if start_index == peak_index:

            flags.append(
                "start_at_peak"
            )

        # ----------------------------------------------------
        # END / RECOVERY
        # ----------------------------------------------------

        end_index = None

        j = b + 1

        limit = (
            next_a
            if next_a is not None
            else n
        )

        while j < n:

            # Missing data terminates the measurable event.
            if np.isnan(x[j]):

                end_index = j - 1

                flags.append(
                    "end_truncated_by_gap"
                )

                break

            # Another core starts before recovery.
            if (
                next_a is not None
                and j >= limit
            ):

                interval = x[
                    b + 1:next_a
                ]

                if len(interval) == 0:

                    end_index = b

                else:

                    end_index = (
                        b
                        + 1
                        + int(
                            np.argmax(
                                interval
                            )
                        )
                    )

                flags.append(
                    "split_without_recovery"
                )

                break

            # Normal recovery.
            if (
                x[j]
                >= params.release_nt
            ):

                end_index = j

                break

            j += 1

        if end_index is None:

            end_index = n - 1

            flags.append(
                "open_end"
            )

        previous_end = end_index

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        start_time = t[start_index]
        peak_time = t[peak_index]
        end_time = t[end_index]

        main_hours = (
            peak_time - start_time
        ) / pd.Timedelta(hours=1)

        recovery_hours = (
            end_time - peak_time
        ) / pd.Timedelta(hours=1)

        total_hours = (
            end_time - start_time
        ) / pd.Timedelta(hours=1)

        minimum_dst = float(
            x[peak_index]
        )

        if main_hours > 0:

            descent_rate = round(
                (
                    x[start_index]
                    - minimum_dst
                )
                / main_hours,
                3,
            )

        else:

            descent_rate = float("nan")

        # ----------------------------------------------------
        # DATA PRODUCT
        # ----------------------------------------------------

        data_product = "unknown"

        if prod is not None:

            section = prod[
                start_index:end_index + 1
            ]

            finite = section[
                np.isfinite(section)
            ]

            if len(finite):

                highest_rank = int(
                    np.max(finite)
                )

                reverse_rank = {
                    value: name
                    for name, value
                    in PRODUCT_RANK.items()
                }

                data_product = reverse_rank.get(
                    highest_rank,
                    "unknown",
                )

        # ----------------------------------------------------
        # RECOVERY TO -150
        # ----------------------------------------------------

        recovery_to_150 = (
            recovery_to_level_hours(
                x,
                peak_index,
                end_index,
                params.crossing_level_nt,
            )
        )

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        rows.append(
            {
                "storm_id": (
                    f"GS_{start_time:%Y%m%d_%H}"
                ),
                "start_time": start_time,
                "peak_time": peak_time,
                "end_time": end_time,
                "onset_dst_nt": float(
                    x[start_index]
                ),
                "min_dst_nt": minimum_dst,
                "intensity_tier": classify_intensity(
                    minimum_dst,
                    params.tiers,
                ),
                "total_duration_hours": total_hours,
                "main_phase_hours": main_hours,
                "recovery_phase_hours": recovery_hours,
                "descent_rate_nt_hr": descent_rate,
                "recovery_to_150_hours": recovery_to_150,
                "data_quality_flag": (
                    ";".join(flags)
                    if flags
                    else "ok"
                ),
                "data_product": data_product,
            }
        )

    return pd.DataFrame(
        rows,
        columns=COLUMNS,
    )


def main(argv=None):

    import argparse

    from .catalogue_validation import (
        assert_valid,
    )

    from .common import (
        load_config,
        resolve,
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config"
    )

    args = parser.parse_args(argv)

    cfg = load_config(
        args.config
    )

    # --------------------------------------------------------
    # Validate configuration BEFORE reading data
    # --------------------------------------------------------

    params = Params.from_config(
        cfg
    )

    clean_path = resolve(
        cfg,
        "clean_series",
    )

    if not clean_path.exists():

        raise FileNotFoundError(
            f"Clean Dst file not found:\n"
            f"{clean_path}\n\n"
            f"Run data cleaning first."
        )

    clean = pd.read_csv(clean_path)

    # Require product metadata so that provisional/realtime records
    # cannot accidentally enter the final research catalogue.
    required_columns = {
        "timestamp_utc",
        "dst_nt",
        "product",
    }

    missing = required_columns - set(clean.columns)
    if missing:
        raise ValueError(
            "Clean dataset is missing required columns: "
            + ", ".join(sorted(missing))
            + ". The final-only detector requires timestamp_utc, dst_nt, and product."
        )

    # Normalize timestamps to UTC and Dst to numeric values.
    clean["timestamp_utc"] = pd.to_datetime(
        clean["timestamp_utc"],
        utc=True,
        errors="coerce",
    )
    clean["dst_nt"] = pd.to_numeric(
        clean["dst_nt"],
        errors="coerce",
    )
    clean["product"] = clean["product"].astype("string").str.strip()

    # Approved study scope: Kyoto WDC final Dst only, 1957-01 through 2020-12.
    start = pd.Timestamp("1957-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2020-12-31 23:00:00", tz="UTC")

    mask = (
        clean["timestamp_utc"].notna()
        & clean["product"].eq("dst_final")
        & clean["timestamp_utc"].between(start, end)
    )
    clean = clean.loc[mask].copy()
    clean = clean.sort_values("timestamp_utc")

    if clean.empty:
        raise ValueError(
            "No dst_final observations were found between 1957-01-01 and 2020-12-31."
        )

    if clean["timestamp_utc"].duplicated().any():
        duplicates = int(clean["timestamp_utc"].duplicated().sum())
        raise ValueError(
            f"Found {duplicates} duplicate timestamps in the final Dst study period."
        )

    clean = clean.set_index("timestamp_utc")

    # Require exactly one observation per hour over the complete study period.
    expected_index = pd.date_range(
        start=start,
        end=end,
        freq="h",
        name="timestamp_utc",
    )

    if not clean.index.equals(expected_index):
        missing_hours = expected_index.difference(clean.index)
        unexpected_hours = clean.index.difference(expected_index)
        raise ValueError(
            "Final Dst hourly coverage check failed. "
            f"Expected {len(expected_index)} hourly rows; found {len(clean)}. "
            f"Missing hours: {len(missing_hours)}; unexpected hours: {len(unexpected_hours)}. "
            "Check the cleaned CSV before running storm detection."
        )

    if clean["dst_nt"].isna().any():
        bad_count = int(clean["dst_nt"].isna().sum())
        raise ValueError(
            f"Found {bad_count} missing or non-numeric Dst values in the final study period."
        )

    product = clean["product"]

    print("\nInput data validated:")
    print("  Product: dst_final only")
    print(f"  Start (UTC): {clean.index.min()}")
    print(f"  End (UTC):   {clean.index.max()}")
    print(f"  Hourly rows: {len(clean)}")

    catalogue = detect_storms(
        clean["dst_nt"],
        params,
        product,
    )

    # --------------------------------------------------------
    # Validate catalogue
    # --------------------------------------------------------

    assert_valid(
        catalogue,
        clean["dst_nt"],
        params,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = resolve(
        cfg,
        "catalogue",
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalogue.to_csv(
        output,
        index=False,
    )

    print(
        f"\n{len(catalogue)} storms detected."
    )

    print(
        f"Catalogue saved to:\n{output}"
    )

    if not catalogue.empty:

        print(
            "\nIntensity distribution:"
        )

        print(
            catalogue[
                "intensity_tier"
            ]
            .value_counts()
            .to_string()
        )


if __name__ == "__main__":
    main()