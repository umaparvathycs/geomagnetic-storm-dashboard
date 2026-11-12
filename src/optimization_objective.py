"""Objective functions for geomagnetic-storm parameter optimization.

Reference events must be independently documented; they must not be derived
from the detector's own catalogue or from the baseline storm count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReferenceEvent:
    """An independently documented historical storm reference interval."""

    name: str
    date_from: pd.Timestamp
    date_to: pd.Timestamp
    min_dst_nt: float | None = None

    def __post_init__(self) -> None:
        start = pd.Timestamp(self.date_from)
        end = pd.Timestamp(self.date_to)

        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")

        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")

        if end <= start:
            raise ValueError(
                f"Reference event {self.name!r} has date_to <= date_from."
            )

        object.__setattr__(self, "date_from", start)
        object.__setattr__(self, "date_to", end)

        if self.min_dst_nt is not None:
            object.__setattr__(
                self,
                "min_dst_nt",
                float(self.min_dst_nt),
            )


def _overlap_hours(
    start_a: pd.Timestamp,
    end_a: pd.Timestamp,
    start_b: pd.Timestamp,
    end_b: pd.Timestamp,
) -> float:
    """Return the positive temporal overlap in hours."""
    start = max(pd.Timestamp(start_a), pd.Timestamp(start_b))
    end = min(pd.Timestamp(end_a), pd.Timestamp(end_b))

    if end <= start:
        return 0.0

    return (end - start).total_seconds() / 3600.0


def _reference_overlap(
    catalogue: pd.DataFrame,
    reference: ReferenceEvent,
) -> pd.DataFrame:
    """Return catalogue storms that overlap a reference interval."""
    if catalogue.empty:
        return pd.DataFrame(
            columns=[
                "catalogue_index",
                "overlap_hours",
                "min_dst_nt",
            ]
        )

    overlap: list[dict] = []

    for idx, row in catalogue.iterrows():
        start = pd.Timestamp(row["start_time"])
        end = pd.Timestamp(row["end_time"])

        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        else:
            start = start.tz_convert("UTC")

        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        else:
            end = end.tz_convert("UTC")

        hours = _overlap_hours(
            start,
            end,
            reference.date_from,
            reference.date_to,
        )

        if hours > 0:
            overlap.append(
                {
                    "catalogue_index": idx,
                    "overlap_hours": hours,
                    "min_dst_nt": float(row["min_dst_nt"]),
                }
            )

    return pd.DataFrame(
        overlap,
        columns=[
            "catalogue_index",
            "overlap_hours",
            "min_dst_nt",
        ],
    )


def evaluate_reference_events(
    catalogue: pd.DataFrame,
    reference_events: Iterable[ReferenceEvent],
    min_dst_tolerance_nt: float = 10.0,
) -> dict:
    """Score event detection, minimum-Dst agreement, and fragmentation."""
    reference_events = list(reference_events)

    if not reference_events:
        raise ValueError(
            "No reference events supplied. Add independently documented "
            "events to the project's known_events configuration before "
            "optimization."
        )

    detected = 0
    min_matches = 0
    fragmented = 0
    details: list[dict] = []

    for ref in reference_events:
        matches = _reference_overlap(catalogue, ref)

        if matches.empty:
            details.append(
                {
                    "name": ref.name,
                    "detected": False,
                    "n_matches": 0,
                    "fragmented": False,
                    "best_overlap_hours": 0.0,
                    "catalogue_min_dst_nt": None,
                    "reference_min_dst_nt": ref.min_dst_nt,
                    "min_dst_match": False,
                }
            )
            continue

        detected += 1

        is_fragmented = len(matches) > 1
        fragmented += int(is_fragmented)

        # Greatest overlap wins; ties go to the stronger (more negative) Dst.
        best = matches.sort_values(
            ["overlap_hours", "min_dst_nt"],
            ascending=[False, True],
        ).iloc[0]

        min_match = False

        if ref.min_dst_nt is not None:
            min_match = (
                abs(
                    float(best["min_dst_nt"])
                    - ref.min_dst_nt
                )
                <= min_dst_tolerance_nt
            )

            min_matches += int(min_match)

        details.append(
            {
                "name": ref.name,
                "detected": True,
                "n_matches": int(len(matches)),
                "fragmented": bool(is_fragmented),
                "best_overlap_hours": float(
                    best["overlap_hours"]
                ),
                "catalogue_min_dst_nt": float(
                    best["min_dst_nt"]
                ),
                "reference_min_dst_nt": ref.min_dst_nt,
                "min_dst_match": bool(min_match),
            }
        )

    n = len(reference_events)

    detection_rate = detected / n
    min_match_rate = min_matches / n
    fragmentation_rate = fragmented / n

    # Reference score:
    # 60% event detection
    # 20% minimum-Dst agreement
    # 20% penalty for fragmented reference events
    score = (
        0.60 * detection_rate
        + 0.20 * min_match_rate
        + 0.20 * (1.0 - fragmentation_rate)
    )

    return {
        "reference_score": float(score),
        "detection_rate": float(detection_rate),
        "min_dst_match_rate": float(min_match_rate),
        "fragmentation_rate": float(fragmentation_rate),
        "reference_events": int(n),
        "reference_detected": int(detected),
        "reference_min_matches": int(min_matches),
        "reference_fragmented": int(fragmented),
        "reference_details": details,
    }


def structural_score(catalogue: pd.DataFrame) -> dict:
    """Score basic catalogue validity; 1.0 means no detected problems."""
    if catalogue.empty:
        return {
            "structural_score": 0.0,
            "duplicate_ids": 0,
            "invalid_time_order": 0,
            "negative_duration": 0,
            "overlaps": 0,
            "missing_required_values": 0,
            "structural_problems": 1,
        }

    required = [
        "storm_id",
        "start_time",
        "peak_time",
        "end_time",
        "min_dst_nt",
        "total_duration_hours",
    ]

    missing_columns = [
        col for col in required
        if col not in catalogue.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Catalogue is missing required columns: {missing_columns}"
        )

    df = catalogue.copy()

    for col in ("start_time", "peak_time", "end_time"):
        df[col] = pd.to_datetime(
            df[col],
            utc=True,
            errors="coerce",
        )

    duplicate_ids = int(
        df["storm_id"].duplicated().sum()
    )

    invalid_time_order = int(
        (
            (df["start_time"] > df["peak_time"])
            | (df["peak_time"] > df["end_time"])
            | (df["start_time"] == df["end_time"])
        )
        .fillna(True)
        .sum()
    )

    durations = pd.to_numeric(
        df["total_duration_hours"],
        errors="coerce",
    )

    negative_duration = int(
        (durations < 0).fillna(True).sum()
    )

    missing_required_values = int(
        df[required].isna().any(axis=1).sum()
    )

    # Count overlapping catalogue intervals.
    ordered = (
        df.sort_values(["start_time", "end_time"])
        .reset_index(drop=True)
    )

    overlaps = 0
    previous_end = None

    for _, row in ordered.iterrows():
        current_start = row["start_time"]
        current_end = row["end_time"]

        if pd.isna(current_start) or pd.isna(current_end):
            continue

        if (
            previous_end is not None
            and current_start < previous_end
        ):
            overlaps += 1

        previous_end = (
            current_end
            if previous_end is None
            else max(previous_end, current_end)
        )

    problems = (
        duplicate_ids
        + invalid_time_order
        + negative_duration
        + overlaps
        + missing_required_values
    )

    score = max(
        0.0,
        1.0 - problems / max(len(df), 1),
    )

    return {
        "structural_score": float(score),
        "duplicate_ids": duplicate_ids,
        "invalid_time_order": invalid_time_order,
        "negative_duration": negative_duration,
        "overlaps": int(overlaps),
        "missing_required_values": missing_required_values,
        "structural_problems": int(problems),
    }


def robustness_score(
    neighboring_catalogue_sizes: Iterable[int] | None,
    center_count: int,
) -> dict:
    """Measure count stability against one-step neighboring parameter sets."""
    if neighboring_catalogue_sizes is None:
        return {
            "robustness_score": 0.0,
            "neighbor_count_mean_relative_deviation": None,
            "neighbor_count_max_relative_deviation": None,
            "neighbor_count_n": 0,
        }

    neighbors = list(neighboring_catalogue_sizes)

    if not neighbors or center_count <= 0:
        return {
            "robustness_score": 0.0,
            "neighbor_count_mean_relative_deviation": None,
            "neighbor_count_max_relative_deviation": None,
            "neighbor_count_n": len(neighbors),
        }

    deviations = [
        abs(float(count) - float(center_count))
        / float(center_count)
        for count in neighbors
    ]

    mean_dev = float(np.mean(deviations))
    max_dev = float(np.max(deviations))

    return {
        "robustness_score": float(
            max(0.0, 1.0 - mean_dev)
        ),
        "neighbor_count_mean_relative_deviation": mean_dev,
        "neighbor_count_max_relative_deviation": max_dev,
        "neighbor_count_n": len(neighbors),
    }


def calculate_objective(
    catalogue: pd.DataFrame,
    reference_events: Iterable[ReferenceEvent],
    neighboring_catalogue_sizes: Iterable[int] | None = None,
) -> tuple[float, dict]:
    """Combine reference performance (70%), structure (20%), robustness (10%)."""
    reference_metrics = evaluate_reference_events(
        catalogue,
        reference_events,
    )

    structural_metrics = structural_score(catalogue)

    robustness_metrics = robustness_score(
        neighboring_catalogue_sizes,
        len(catalogue),
    )

    objective = (
        0.70 * reference_metrics["reference_score"]
        + 0.20 * structural_metrics["structural_score"]
        + 0.10 * robustness_metrics["robustness_score"]
    )

    metrics = {
        "objective": float(objective),
        **reference_metrics,
        **structural_metrics,
        **robustness_metrics,
        "storm_count": int(len(catalogue)),
    }

    return float(objective), metrics


def objective_to_dict(
    objective_result: tuple[float, dict],
) -> dict:
    """Return calculate_objective() as a flat dictionary."""
    objective, metrics = objective_result

    result = dict(metrics)
    result["objective"] = float(objective)

    return result