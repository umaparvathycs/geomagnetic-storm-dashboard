"""
Catalogue validation.

validate_catalogue() returns a list of problems.

assert_valid() raises CatalogueValidationError
if any validation problem is found.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import classify_intensity
from .storm_detection import (
    Params,
    recovery_to_level_hours,
    to_regular_grid,
)


TOL = 1e-6


class CatalogueValidationError(Exception):
    """Raised when the storm catalogue fails validation."""


def validate_catalogue(
    cat: pd.DataFrame,
    series: pd.Series | None = None,
    params: Params = Params(),
) -> list[str]:

    P: list[str] = []

    if cat.empty:
        return P

    c = cat.reset_index(drop=True).copy()

    # --------------------------------------------------------
    # REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = {
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
    }

    missing_columns = (
        required_columns
        - set(c.columns)
    )

    if missing_columns:

        P.append(
            "missing catalogue columns: "
            + ", ".join(sorted(missing_columns))
        )

        return P

    # --------------------------------------------------------
    # TIMESTAMPS
    # --------------------------------------------------------

    for col in (
        "start_time",
        "peak_time",
        "end_time",
    ):

        c[col] = pd.to_datetime(
            c[col],
            utc=True,
        )

    # --------------------------------------------------------
    # BASIC IDENTIFIERS
    # --------------------------------------------------------

    if c.storm_id.duplicated().any():

        P.append(
            "duplicate storm_id: "
            f"{c.storm_id[c.storm_id.duplicated()].tolist()[:5]}"
        )

    if c.duplicated(
        ["start_time", "end_time"]
    ).any():

        P.append(
            "duplicate intervals"
        )

    if not c.start_time.is_monotonic_increasing:

        P.append(
            "catalogue not sorted by start_time"
        )

    # --------------------------------------------------------
    # OVERLAPPING / TOUCHING EVENTS
    # --------------------------------------------------------

    if len(c) > 1:

        starts = c.start_time.iloc[1:].to_numpy()
        previous_ends = c.end_time.iloc[:-1].to_numpy()

        bad = (
            starts
            <= previous_ends
        )

        if bad.any():

            rows = (
                np.flatnonzero(bad) + 1
            )[:5]

            P.append(
                "overlapping/touching intervals "
                f"at rows {rows.tolist()}"
            )

    # --------------------------------------------------------
    # DATA QUALITY FLAGS
    # --------------------------------------------------------

    flags = (
        c.data_quality_flag
        .fillna("")
        .astype(str)
    )

    edge = flags.str.contains(
        "end_truncated_by_gap"
        "|open_end"
        "|start_at_data_edge"
        "|onset_truncated"
        "|onset_truncated_by_gap",
        regex=True,
    )

    # --------------------------------------------------------
    # TIME ORDER
    # --------------------------------------------------------

    if (
        c.start_time
        > c.peak_time
    ).any():

        P.append(
            "start > peak"
        )

    if (
        c.peak_time
        > c.end_time
    ).any():

        P.append(
            "peak > end"
        )

    # --------------------------------------------------------
    # START == PEAK
    # --------------------------------------------------------
    #
    # A storm can legitimately begin at its minimum.
    # This is explicitly marked as start_at_peak.
    #
    # Other legitimate boundary flags are also accepted.
    # --------------------------------------------------------

    eq_s = (
        (c.start_time == c.peak_time)
        & ~flags.str.contains(
            "start_at_data_edge"
            "|onset_truncated"
            "|onset_truncated_by_gap"
            "|split_without_recovery"
            "|start_at_peak",
            regex=True,
        )
    )

    if eq_s.any():

        P.append(
            "start == peak without "
            "justifying flag: "
            f"{c.storm_id[eq_s].tolist()[:5]}"
        )

    # --------------------------------------------------------
    # PEAK == END
    # --------------------------------------------------------

    eq_e = (
        (c.peak_time == c.end_time)
        & ~edge
    )

    if eq_e.any():

        P.append(
            "peak == end without "
            "justifying flag: "
            f"{c.storm_id[eq_e].tolist()[:5]}"
        )

    # --------------------------------------------------------
    # DURATION VALIDATION
    # --------------------------------------------------------

    h = pd.Timedelta(hours=1)

    duration_checks = [
        (
            "total_duration_hours",
            "end_time",
            "start_time",
        ),
        (
            "main_phase_hours",
            "peak_time",
            "start_time",
        ),
        (
            "recovery_phase_hours",
            "end_time",
            "peak_time",
        ),
    ]

    for col, a, b in duration_checks:

        if (
            c[col] < 0
        ).any():

            P.append(
                f"negative {col}"
            )

        expected = (
            c[a] - c[b]
        ) / h

        if (
            abs(
                c[col] - expected
            )
            > TOL
        ).any():

            P.append(
                f"{col} != {a}-{b}"
            )

    if (
        abs(
            c.total_duration_hours
            - c.main_phase_hours
            - c.recovery_phase_hours
        )
        > TOL
    ).any():

        P.append(
            "total != main + recovery"
        )

    # --------------------------------------------------------
    # Dst / TRIGGER VALIDATION
    # --------------------------------------------------------

    if (
        c.min_dst_nt
        > params.trigger_nt
    ).any():

        P.append(
            "min_dst above trigger"
        )

    # --------------------------------------------------------
    # INTENSITY TIER VALIDATION
    # --------------------------------------------------------

    exp_tier = c.min_dst_nt.map(
        lambda v:
        classify_intensity(
            v,
            params.tiers,
        )
    )

    tier_mismatch = (
        exp_tier
        != c.intensity_tier
    )

    if tier_mismatch.any():

        P.append(
            "intensity tier mismatch: "
            f"{c.storm_id[tier_mismatch].tolist()[:5]}"
        )

    # ========================================================
    # RECOVERY-TO-150 VALIDATION
    # ========================================================

    r = c.recovery_to_150_hours
    lvl = params.crossing_level_nt

    # --------------------------------------------------------
    # Case A:
    # Minimum never reached -150.
    #
    # Therefore recovery_to_150 MUST be NaN.
    # --------------------------------------------------------

    invalid_above = (
        r.notna()
        & (
            c.min_dst_nt
            > lvl
        )
    )

    if invalid_above.any():

        P.append(
            "recovery_to_150 set although "
            "min_dst never reached the level"
        )

    # --------------------------------------------------------
    # Case B:
    # Minimum is BELOW -150.
    #
    # A valid observed recovery must be:
    #
    #     > 0
    #     <= recovery_phase_hours
    #
    # --------------------------------------------------------

    below = (
        c.min_dst_nt
        < lvl
    )

    invalid_below = (
        r.notna()
        & below
        & (
            (r <= 0)
            | (
                r
                > c.recovery_phase_hours
                + TOL
            )
        )
    )

    if invalid_below.any():

        P.append(
            "recovery_to_150 not within "
            "(0, recovery_phase] for "
            "storm below recovery level"
        )

    # --------------------------------------------------------
    # Case C:
    # Minimum is EXACTLY -150.
    #
    # By definition:
    #
    #     recovery_to_150 = 0
    # --------------------------------------------------------

    exact = np.isclose(
        c.min_dst_nt,
        lvl,
        atol=TOL,
    )

    invalid_exact = (
        exact
        & (
            r.isna()
            | (
                abs(r)
                > TOL
            )
        )
    )

    if invalid_exact.any():

        P.append(
            "recovery_to_150 must be 0 "
            "when minimum Dst equals "
            "recovery level"
        )

    # --------------------------------------------------------
    # Case D:
    # Minimum BELOW -150 but no recovery observed.
    #
    # For a completed event this is an error.
    #
    # For an event truncated by data edge/gap/open end,
    # NaN is acceptable.
    # --------------------------------------------------------

    miss = (
        r.isna()
        & below
        & ~edge
    )

    if miss.any():

        P.append(
            "recovery_to_150 NaN for "
            "completed storm that reached "
            "below the level: "
            f"{c.storm_id[miss].tolist()[:5]}"
        )

    # ========================================================
    # CHECKS AGAINST THE ORIGINAL Dst SERIES
    # ========================================================

    if series is not None:

        s = to_regular_grid(
            series.astype(float)
        )
        if s.index.tz is None:
            s.index = s.index.tz_localize("UTC")
        elif str(s.index.tz) != "UTC":
            s.index = s.index.tz_convert("UTC")

        x = s.to_numpy()
        idx = s.index

        for row in c.itertuples():

            try:

                i0 = idx.get_loc(
                    row.start_time
                )

                ip = idx.get_loc(
                    row.peak_time
                )

                i1 = idx.get_loc(
                    row.end_time
                )

            except KeyError:

                P.append(
                    f"{row.storm_id}: "
                    "timestamp not in series"
                )

                continue

            # ------------------------------------------------
            # Event segment
            # ------------------------------------------------

            seg = x[
                i0:i1 + 1
            ]

            if np.isnan(seg).any():

                P.append(
                    f"{row.storm_id}: "
                    "missing data inside "
                    "event interval"
                )

                continue

            # ------------------------------------------------
            # Peak location
            # ------------------------------------------------

            if not (
                i0
                <= ip
                <= i1
            ):

                P.append(
                    f"{row.storm_id}: "
                    "peak outside interval"
                )

            # ------------------------------------------------
            # Minimum Dst consistency
            # ------------------------------------------------

            if (
                abs(
                    np.nanmin(seg)
                    - row.min_dst_nt
                )
                > TOL
                or
                abs(
                    x[ip]
                    - row.min_dst_nt
                )
                > TOL
            ):

                P.append(
                    f"{row.storm_id}: "
                    "min_dst inconsistent "
                    "with series"
                )

            # ------------------------------------------------
            # Peak must be first minimum
            # ------------------------------------------------

            if (
                i0
                + int(
                    np.argmin(seg)
                )
                != ip
            ):

                P.append(
                    f"{row.storm_id}: "
                    "peak is not the "
                    "(first) minimum "
                    "of the interval"
                )

            # ------------------------------------------------
            # Recompute recovery-to-150
            # ------------------------------------------------

            rec = recovery_to_level_hours(
                x,
                ip,
                i1,
                lvl,
            )

            stored_rec = (
                row.recovery_to_150_hours
            )

            # Both NaN -> correct.
            if (
                np.isnan(rec)
                and np.isnan(stored_rec)
            ):
                pass

            # One NaN but not the other -> wrong.
            elif (
                np.isnan(rec)
                != np.isnan(stored_rec)
            ):

                P.append(
                    f"{row.storm_id}: "
                    "recovery_to_150 differs "
                    "from recomputation "
                    f"({stored_rec} vs {rec})"
                )

            # Both numeric -> compare.
            elif (
                abs(
                    rec
                    - stored_rec
                )
                >= 1e-2
            ):

                P.append(
                    f"{row.storm_id}: "
                    "recovery_to_150 differs "
                    "from recomputation "
                    f"({stored_rec} vs {rec})"
                )

    return P


def assert_valid(
    cat,
    series=None,
    params: Params = Params(),
):

    problems = validate_catalogue(
        cat,
        series,
        params,
    )

    if problems:

        raise CatalogueValidationError(
            f"{len(problems)} problem(s):\n  "
            + "\n  ".join(
                problems[:30]
            )
        )

    return True