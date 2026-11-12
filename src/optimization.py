"""Grid-search optimization for geomagnetic storm detection parameters.

The existing storm detector is used unchanged. Optimization uses independently
specified reference events, catalogue validity, and local parameter robustness;
it does not optimize toward the baseline catalogue's storm count or durations.
"""
from __future__ import annotations

import json
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from .common import load_config, resolve
from .storm_detection import Params, detect_storms
from .optimization_objective import (
    ReferenceEvent,
    calculate_objective,
    objective_to_dict,
)


TRIGGER_VALUES = [-40.0, -35.0, -30.0, -25.0]
RELEASE_VALUES = [-25.0, -20.0, -15.0, -10.0]
MERGE_GAP_VALUES = [3, 6, 9, 12]
MIN_CORE_VALUES = [1, 2, 3]

PARAMETER_NAMES = (
    "trigger_nt",
    "release_nt",
    "merge_gap_hours",
    "min_core_hours",
)


def load_dst_series(cfg) -> pd.Series:
    """Load the cleaned, validated hourly Dst series."""
    path = resolve(cfg, "clean_series")

    if not path.exists():
        raise FileNotFoundError(
            f"Clean Dst series not found: {path}\n"
            "Run preprocessing first."
        )

    df = pd.read_csv(path, parse_dates=["timestamp_utc"])

    required = ["timestamp_utc", "dst_nt"]
    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Clean Dst series is missing columns: {', '.join(missing)}"
        )

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    if df["timestamp_utc"].isna().any():
        raise ValueError(
            "Invalid or missing timestamp_utc values in cleaned Dst series."
        )

    if df["timestamp_utc"].duplicated().any():
        raise ValueError(
            "Duplicate timestamps found in cleaned Dst series."
        )

    if df["dst_nt"].isna().any():
        raise ValueError(
            "Missing Dst values found in cleaned Dst series."
        )

    timestamps = pd.DatetimeIndex(
        pd.to_datetime(df["timestamp_utc"], utc=True)
    )

    if len(timestamps) > 1:
        gaps = timestamps.to_series().diff().dropna()

        if not (gaps == pd.Timedelta(hours=1)).all():
            raise ValueError(
                "Cleaned Dst series is not continuous hourly data."
            )

    return pd.Series(
        df["dst_nt"].to_numpy(dtype=float),
        index=timestamps,
        name="dst_nt",
    )


def build_base_params(cfg) -> Params:
    """Construct detector parameters from the current project configuration."""
    detection_cfg = cfg["detection"]

    tiers = {
        key: tuple(value)
        for key, value in detection_cfg["tiers"].items()
    }

    return Params(
        trigger_nt=float(detection_cfg["trigger_nt"]),
        release_nt=float(detection_cfg["release_nt"]),
        merge_gap_hours=int(detection_cfg["merge_gap_hours"]),
        min_core_hours=int(detection_cfg["min_core_hours"]),
        max_lookback_hours=int(
            detection_cfg["max_lookback_hours"]
        ),
        crossing_level_nt=float(
            detection_cfg["crossing_level_nt"]
        ),
        tiers=tiers,
    )


def _first_present(item: dict, keys: tuple[str, ...]):
    """Return the first non-empty value from the supplied keys."""
    for key in keys:
        if item.get(key) not in (None, ""):
            return item[key]

    return None


def _as_reference_event(item, index: int) -> ReferenceEvent:
    """Convert a configured event dictionary into a ReferenceEvent."""
    if isinstance(item, ReferenceEvent):
        return item

    if not isinstance(item, dict):
        raise TypeError(
            f"Reference event #{index} must be a mapping/dictionary; "
            f"got {type(item).__name__}."
        )

    name = str(
        item.get("name")
        or item.get("event_name")
        or f"Reference event {index}"
    )

    start = _first_present(
        item,
        (
            "date_from",
            "start_time",
            "start_date",
            "start",
            "date_start",
        ),
    )

    end = _first_present(
        item,
        (
            "date_to",
            "end_time",
            "end_date",
            "end",
            "date_end",
        ),
    )

    minimum = _first_present(
        item,
        (
            "min_dst_nt",
            "minimum_dst_nt",
            "min_dst",
            "dst_min",
        ),
    )

    if start is None or end is None:
        raise ValueError(
            f"Reference event {name!r} must contain a start date "
            "(date_from/start_time) and an end date "
            "(date_to/end_time)."
        )

    return ReferenceEvent(
        name=name,
        date_from=pd.Timestamp(start),
        date_to=pd.Timestamp(end),
        min_dst_nt=(
            None if minimum is None else float(minimum)
        ),
    )


def load_known_events(cfg) -> list[ReferenceEvent]:
    """Load and validate reference events from project configuration."""
    events = cfg.get("known_events", [])

    if events is None:
        events = []

    if isinstance(events, dict):
        # Support either {"events": [...]} or a mapping of event names
        # to event-field dictionaries.
        if isinstance(events.get("events"), list):
            events = events["events"]
        else:
            converted = []

            for name, value in events.items():
                if not isinstance(value, dict):
                    raise TypeError(
                        "known_events must be a list of event dictionaries."
                    )

                item = dict(value)
                item.setdefault("name", name)
                converted.append(item)

            events = converted

    if not isinstance(events, (list, tuple)):
        raise TypeError(
            "Configuration field 'known_events' must be a list "
            "of event dictionaries."
        )

    parsed = [
        _as_reference_event(event, i + 1)
        for i, event in enumerate(events)
    ]

    if not parsed:
        raise ValueError(
            "No independently documented reference events were found "
            "in config['known_events']. Add reference events before "
            "running optimization; the script will not invent them."
        )

    return parsed


def generate_parameter_combinations() -> list[dict]:
    """Generate parameter combinations with release threshold above trigger."""
    combinations = []

    for trigger, release, merge_gap, min_core in product(
        TRIGGER_VALUES,
        RELEASE_VALUES,
        MERGE_GAP_VALUES,
        MIN_CORE_VALUES,
    ):
        if release <= trigger:
            continue

        combinations.append(
            {
                "trigger_nt": float(trigger),
                "release_nt": float(release),
                "merge_gap_hours": int(merge_gap),
                "min_core_hours": int(min_core),
            }
        )

    return combinations


def _catalogue_from_records(records) -> pd.DataFrame:
    """Create a catalogue DataFrame with required columns."""
    catalogue = pd.DataFrame(records)

    if catalogue.empty:
        return pd.DataFrame(
            columns=[
                "storm_id",
                "start_time",
                "peak_time",
                "end_time",
                "min_dst_nt",
                "total_duration_hours",
            ]
        )

    return catalogue


def evaluate_candidate(
    series,
    base_params,
    known_events,
    candidate,
    neighboring_catalogue_sizes=None,
):
    """Run the existing detector and score one parameter combination."""
    params = replace(
        base_params,
        trigger_nt=float(candidate["trigger_nt"]),
        release_nt=float(candidate["release_nt"]),
        merge_gap_hours=int(candidate["merge_gap_hours"]),
        min_core_hours=int(candidate["min_core_hours"]),
    )

    catalogue = _catalogue_from_records(
        detect_storms(series, params)
    )

    result = objective_to_dict(
        calculate_objective(
            catalogue,
            known_events,
            neighboring_catalogue_sizes,
        )
    )

    result.update(
        {
            key: candidate[key]
            for key in PARAMETER_NAMES
        }
    )

    # Compatibility aliases for readable output.
    result["n_storms"] = int(len(catalogue))
    result["total_score"] = float(result["objective"])

    return result, catalogue


def _neighbor_indices(
    index: int,
    combinations: list[dict],
) -> list[int]:
    """Return candidates one grid step away on exactly one parameter axis."""
    grids = {
        "trigger_nt": TRIGGER_VALUES,
        "release_nt": RELEASE_VALUES,
        "merge_gap_hours": MERGE_GAP_VALUES,
        "min_core_hours": MIN_CORE_VALUES,
    }

    center = combinations[index]
    neighbors = []

    for other_index, other in enumerate(combinations):
        if other_index == index:
            continue

        distances = []

        for key in PARAMETER_NAMES:
            distances.append(
                abs(
                    grids[key].index(center[key])
                    - grids[key].index(other[key])
                )
            )

        if sum(distances) == 1:
            neighbors.append(other_index)

    return neighbors


def _json_event(event: ReferenceEvent) -> dict:
    """Convert a reference event to JSON-serializable values."""
    return {
        "name": event.name,
        "date_from": event.date_from.isoformat(),
        "date_to": event.date_to.isoformat(),
        "min_dst_nt": event.min_dst_nt,
    }


def main(argv=None) -> None:
    print("=" * 70)
    print("GEOMAGNETIC STORM PROJECT - PARAMETER OPTIMIZATION")
    print("=" * 70)

    cfg = load_config(None)

    # Load the cleaned hourly Dst series.
    series = load_dst_series(cfg)

    print(f"\nObservations : {len(series):,}")
    print(f"Start        : {series.index.min()}")
    print(f"End          : {series.index.max()}")

    # Build the existing detector's base configuration.
    base_params = build_base_params(cfg)

    print("\nCurrent detector parameters:")
    print(f"  trigger_nt       = {base_params.trigger_nt}")
    print(f"  release_nt       = {base_params.release_nt}")
    print(f"  merge_gap_hours  = {base_params.merge_gap_hours}")
    print(f"  min_core_hours   = {base_params.min_core_hours}")
    print(f"  max_lookback     = {base_params.max_lookback_hours}")
    print(f"  crossing_level   = {base_params.crossing_level_nt}")

    # Load independently documented reference events.
    known_events = load_known_events(cfg)

    print(f"\nKnown reference events: {len(known_events)}")

    for event in known_events:
        print(
            f"  - {event.name}: "
            f"{event.date_from.date()} to {event.date_to.date()}"
        )

    if len(known_events) < 5:
        print(
            "WARNING: Few reference events are configured; "
            "the selected parameters may not generalize well."
        )

    # Generate the grid.
    combinations = generate_parameter_combinations()

    print(f"\nParameter combinations: {len(combinations)}")
    print("Running grid search (detector runs once per combination)...")
    print("-" * 70)

    # First pass: evaluate each combination once.
    successful: dict[int, dict] = {}
    results: list[dict] = []

    for i, candidate in enumerate(combinations):
        label = (
            f"[{i + 1:03d}/{len(combinations):03d}] "
            f"trigger={candidate['trigger_nt']:6.1f}, "
            f"release={candidate['release_nt']:6.1f}, "
            f"gap={candidate['merge_gap_hours']:2d}, "
            f"core={candidate['min_core_hours']}"
        )

        print(label)

        try:
            result, _ = evaluate_candidate(
                series,
                base_params,
                known_events,
                candidate,
            )

            successful[i] = result
            results.append(result)

            print(
                f"        storms={result['storm_count']:5d}, "
                f"preliminary objective={result['objective']:.6f}"
            )

        except Exception as exc:
            print(f"        ERROR: {exc}")

            error_row = dict(candidate)

            error_row.update(
                {
                    "objective": -1.0,
                    "total_score": -1.0,
                    "storm_count": 0,
                    "n_storms": 0,
                    "reference_score": 0.0,
                    "structural_score": 0.0,
                    "robustness_score": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

            results.append(error_row)

    if not successful:
        raise RuntimeError(
            "Every parameter combination failed. Check the printed errors. "
            "No best combination will be reported."
        )

    # Second pass: calculate local robustness using successful neighbors.
    successful_results = []

    for i, base_result in successful.items():
        neighbor_indices = _neighbor_indices(i, combinations)

        neighbor_counts = [
            successful[j]["storm_count"]
            for j in neighbor_indices
            if j in successful
        ]

        if neighbor_counts and base_result["storm_count"] > 0:
            deviations = [
                abs(float(count) - base_result["storm_count"])
                / base_result["storm_count"]
                for count in neighbor_counts
            ]

            mean_dev = float(np.mean(deviations))
            max_dev = float(np.max(deviations))
            robustness_score_value = max(0.0, 1.0 - mean_dev)

        else:
            mean_dev = None
            max_dev = None
            robustness_score_value = 0.0

        base_result["robustness_score"] = float(
            robustness_score_value
        )
        base_result["neighbor_count_mean_relative_deviation"] = mean_dev
        base_result["neighbor_count_max_relative_deviation"] = max_dev
        base_result["neighbor_count_n"] = len(neighbor_counts)

        base_result["objective"] = float(
            0.70 * base_result["reference_score"]
            + 0.20 * base_result["structural_score"]
            + 0.10 * base_result["robustness_score"]
        )

        base_result["total_score"] = base_result["objective"]
        successful_results.append(base_result)

    # Replace preliminary successful rows with final scores.
    # Keep failed rows in the output for auditing.
    final_by_candidate = {
        tuple((key, row[key]) for key in PARAMETER_NAMES): row
        for row in successful_results
    }

    final_results = []

    for row in results:
        key = tuple(
            (name, row[name])
            for name in PARAMETER_NAMES
        )

        final_results.append(
            final_by_candidate.get(key, row)
        )

    results_df = pd.DataFrame(final_results)

    results_df["_failed"] = (
        results_df.get(
            "error",
            pd.Series(index=results_df.index, dtype=object),
        ).notna()
    )

    results_df = (
        results_df.sort_values(
            ["_failed", "objective"],
            ascending=[True, False],
        )
        .drop(columns="_failed")
        .reset_index(drop=True)
    )

    results_df.insert(
        0,
        "rank",
        range(1, len(results_df) + 1),
    )

    # Prepare the output directory.
    metrics_dir = resolve(cfg, "metrics_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Save all candidate results.
    results_path = metrics_dir / "optimization_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n[OK] {results_path}")

    # Only successful candidates may be selected as the best.
    valid_df = results_df[
        results_df.get(
            "error",
            pd.Series(index=results_df.index, dtype=object),
        ).isna()
    ]

    if valid_df.empty:
        raise RuntimeError(
            "No successful candidate exists; refusing to report "
            "a failed row as the best result."
        )

    best = valid_df.iloc[0]

    best_parameters = {
        key: (
            int(best[key])
            if key.endswith("hours")
            else float(best[key])
        )
        for key in PARAMETER_NAMES
    }

    best_objective = {
        "objective": float(best["objective"]),
        "reference_score": float(best["reference_score"]),
        "structural_score": float(best["structural_score"]),
        "robustness_score": float(best["robustness_score"]),
        "storm_count": int(best["storm_count"]),
        "reference_events": int(best["reference_events"]),
        "reference_detected": int(best["reference_detected"]),
        "reference_min_matches": int(best["reference_min_matches"]),
        "reference_fragmented": int(best["reference_fragmented"]),
        "structural_problems": int(best["structural_problems"]),
        "neighbor_count_n": int(best["neighbor_count_n"]),
    }

    summary = {
        "method": "grid_search_with_one_step_neighbor_robustness",
        "parameter_space": {
            "trigger_nt": TRIGGER_VALUES,
            "release_nt": RELEASE_VALUES,
            "merge_gap_hours": MERGE_GAP_VALUES,
            "min_core_hours": MIN_CORE_VALUES,
        },
        "number_of_combinations": len(combinations),
        "successful_combinations": len(successful),
        "failed_combinations": len(combinations) - len(successful),
        "objective_weights": {
            "reference_performance": 0.70,
            "structural_validity": 0.20,
            "local_robustness": 0.10,
        },
        "baseline_parameters": {
            "trigger_nt": float(base_params.trigger_nt),
            "release_nt": float(base_params.release_nt),
            "merge_gap_hours": int(base_params.merge_gap_hours),
            "min_core_hours": int(base_params.min_core_hours),
            "max_lookback_hours": int(base_params.max_lookback_hours),
            "crossing_level_nt": float(base_params.crossing_level_nt),
        },
        "best_parameters": best_parameters,
        "best_objective": best_objective,
        "known_events": [
            _json_event(event)
            for event in known_events
        ],
        "scientific_caveat": (
            "Optimization is only as representative as the independent "
            "reference events. A small reference set can identify parameters "
            "that fit those cases but does not establish general performance. "
            "No baseline catalogue count/duration targets or CNN performance "
            "are used as ground truth."
        ),
    }

    summary_path = metrics_dir / "optimization_summary.json"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            allow_nan=False,
        )

    print(f"[OK] {summary_path}")

    # Print the best successful candidate.
    print("\n" + "=" * 70)
    print("BEST SUCCESSFUL PARAMETER COMBINATION")
    print("=" * 70)

    for key, value in best_parameters.items():
        unit = (
            " nT" if key.endswith("_nt")
            else " h" if key.endswith("hours")
            else ""
        )

        print(f"{key:24s}: {value}{unit}")

    print(f"Objective score         : {best_objective['objective']:.6f}")
    print(f"Reference score         : {best_objective['reference_score']:.6f}")
    print(f"Structural score        : {best_objective['structural_score']:.6f}")
    print(f"Robustness score        : {best_objective['robustness_score']:.6f}")
    print(f"Storms detected         : {best_objective['storm_count']:,}")

    print(
        "Reference events found  : "
        f"{best_objective['reference_detected']}/"
        f"{best_objective['reference_events']}"
    )

    print("\nNOTE: This is an optimization result, not independent validation.")
    print("=" * 70)


if __name__ == "__main__":
    main()