"""
Baseline geomagnetic storm detection.

Runs the existing deterministic storm detector using the baseline
parameters from config.yaml and records baseline statistics.
"""

from __future__ import annotations

import json

import pandas as pd

from .catalogue_validation import assert_valid
from .common import load_config, resolve
from .storm_detection import Params, detect_storms


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run baseline geomagnetic storm detection."
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to config.yaml",
    )

    args = parser.parse_args(argv)

    print("=" * 70)
    print("GEOMAGNETIC STORM PROJECT - BASELINE DETECTION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------

    cfg = load_config(args.config)
    params = Params.from_config(cfg)

    print("\nBaseline detection parameters:")
    print(f"  Trigger threshold : {params.trigger_nt} nT")
    print(f"  Release threshold : {params.release_nt} nT")
    print(f"  Merge gap         : {params.merge_gap_hours} hours")
    print(f"  Minimum core      : {params.min_core_hours} hours")
    print(f"  Maximum lookback  : {params.max_lookback_hours} hours")
    print(f"  Recovery level    : {params.crossing_level_nt} nT")

    # ------------------------------------------------------------
    # Load cleaned Dst data
    # ------------------------------------------------------------

    clean_path = resolve(cfg, "clean_series")

    if not clean_path.exists():
        raise FileNotFoundError(
            f"Clean Dst file not found:\n{clean_path}\n\n"
            "Run data cleaning first."
        )

    print("\nLoading cleaned Dst data...")

    clean = pd.read_csv(
        clean_path,
        parse_dates=["timestamp_utc"],
    )

    required = {"timestamp_utc", "dst_nt"}
    missing = required - set(clean.columns)

    if missing:
        raise ValueError(
            "Clean dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

    clean = clean.set_index("timestamp_utc").sort_index()

    product = (
        clean["product"]
        if "product" in clean.columns
        else None
    )

    print(f"  Observations: {len(clean):,}")
    print(f"  Start       : {clean.index.min()}")
    print(f"  End         : {clean.index.max()}")

    # ------------------------------------------------------------
    # Run baseline detector
    # ------------------------------------------------------------

    print("\nRunning baseline storm detector...")

    catalogue = detect_storms(
        clean["dst_nt"],
        params,
        product,
    )

    print(f"  Storms detected: {len(catalogue):,}")

    # ------------------------------------------------------------
    # Validate catalogue
    # ------------------------------------------------------------

    print("\nValidating baseline catalogue...")

    assert_valid(
        catalogue,
        clean["dst_nt"],
        params,
    )

    print("  [OK] Catalogue validation passed.")

    # ------------------------------------------------------------
    # Calculate baseline statistics
    # ------------------------------------------------------------

    if catalogue.empty:
        raise RuntimeError(
            "Baseline detector produced zero storms."
        )

    duration = catalogue["total_duration_hours"]
    minimum = catalogue["min_dst_nt"]

    metrics = {
        "method": "deterministic_baseline",
        "parameters": {
            "trigger_nt": params.trigger_nt,
            "release_nt": params.release_nt,
            "merge_gap_hours": params.merge_gap_hours,
            "min_core_hours": params.min_core_hours,
            "max_lookback_hours": params.max_lookback_hours,
            "crossing_level_nt": params.crossing_level_nt,
        },
        "dataset": {
            "observations": int(len(clean)),
            "start": str(clean.index.min()),
            "end": str(clean.index.max()),
        },
        "n_storms": int(len(catalogue)),
        "duration_hours": {
            "mean": float(duration.mean()),
            "median": float(duration.median()),
            "min": float(duration.min()),
            "max": float(duration.max()),
            "p95": float(duration.quantile(0.95)),
        },
        "min_dst_nt": {
            "mean": float(minimum.mean()),
            "median": float(minimum.median()),
            "min": float(minimum.min()),
            "max": float(minimum.max()),
        },
        "intensity_distribution": {
            str(k): int(v)
            for k, v in (
                catalogue["intensity_tier"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "quality_flags": {
            str(k): int(v)
            for k, v in (
                catalogue["data_quality_flag"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
    }

    # ------------------------------------------------------------
    # Save baseline catalogue
    # ------------------------------------------------------------

    catalogue_dir = resolve(cfg, "catalogue").parent
    catalogue_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_catalogue = (
        catalogue_dir / "baseline_storm_catalogue.csv"
    )

    catalogue.to_csv(
        output_catalogue,
        index=False,
    )

    # ------------------------------------------------------------
    # Save baseline metrics
    # ------------------------------------------------------------

    metrics_dir = resolve(cfg, "metrics_dir")
    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_metrics = (
        metrics_dir / "baseline_detection_metrics.json"
    )

    with open(
        output_metrics,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------

    print("\n" + "-" * 70)
    print("BASELINE DETECTION RESULTS")
    print("-" * 70)

    print(f"Storms detected       : {len(catalogue):,}")
    print(f"Median duration       : {duration.median():.2f} h")
    print(f"Mean duration         : {duration.mean():.2f} h")
    print(f"95th percentile       : {duration.quantile(0.95):.2f} h")
    print(f"Longest storm         : {duration.max():.2f} h")
    print(f"Median minimum Dst    : {minimum.median():.2f} nT")
    print(f"Strongest storm       : {minimum.min():.2f} nT")

    print("\nIntensity distribution:")
    print(
        catalogue["intensity_tier"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nData-quality flags:")
    print(
        catalogue["data_quality_flag"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nOutputs:")
    print(f"  Catalogue: {output_catalogue}")
    print(f"  Metrics  : {output_metrics}")

    print("\n" + "=" * 70)
    print("BASELINE DETECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()