"""
Storm-level feature analysis.

Reads the validated baseline storm catalogue and produces:
1. A storm-level feature summary CSV
2. A JSON summary of the feature distributions
3. Separate figures showing key storm-feature distributions

This module does NOT modify the storm detector or catalogue.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd

from .common import load_config, resolve


REQUIRED_COLUMNS = [
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


def load_catalogue(cfg):
    """
    Load the validated baseline storm catalogue.
    """

    catalogue_path = (
        resolve(cfg, "catalogue").parent
        / "baseline_storm_catalogue.csv"
    )

    if not catalogue_path.exists():
        raise FileNotFoundError(
            f"Baseline catalogue not found:\n"
            f"{catalogue_path}\n\n"
            f"Run baseline detection first."
        )

    df = pd.read_csv(
        catalogue_path,
        parse_dates=[
            "start_time",
            "peak_time",
            "end_time",
        ],
    )

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            "Baseline catalogue is missing required columns: "
            + ", ".join(sorted(missing))
        )

    return df, catalogue_path


def validate_features(df):
    """
    Perform structural checks on the storm-level features.

    A storm is valid when:

        start_time < peak_time < end_time

    OR when onset occurs exactly at the peak:

        start_time == peak_time < end_time

    The second case is allowed because the existing storm detector
    explicitly supports the legitimate 'start_at_peak' condition.
    """

    checks = {}

    checks["rows"] = int(len(df))

    # ------------------------------------------------------------
    # Storm ID validation
    # ------------------------------------------------------------

    checks["duplicate_storm_ids"] = int(
        df["storm_id"].duplicated().sum()
    )

    # ------------------------------------------------------------
    # Time-order validation
    #
    # Valid:
    #   start < peak < end
    #   start == peak < end
    #
    # Invalid:
    #   start > peak
    #   peak > end
    #   start == end
    # ------------------------------------------------------------

    checks["invalid_time_order"] = int(
        (
            (df["start_time"] > df["peak_time"])
            | (df["peak_time"] > df["end_time"])
            | (df["start_time"] == df["end_time"])
        ).sum()
    )

    # ------------------------------------------------------------
    # Duration validation
    # ------------------------------------------------------------

    checks["negative_total_duration"] = int(
        (df["total_duration_hours"] < 0).sum()
    )

    checks["negative_main_phase"] = int(
        (df["main_phase_hours"] < 0).sum()
    )

    checks["negative_recovery_phase"] = int(
        (df["recovery_phase_hours"] < 0).sum()
    )

    checks["negative_descent_rate"] = int(
        (df["descent_rate_nt_hr"] < 0).sum()
    )

    # ------------------------------------------------------------
    # Duration consistency
    # ------------------------------------------------------------

    duration_difference = (
        df["total_duration_hours"]
        - (
            df["main_phase_hours"]
            + df["recovery_phase_hours"]
        )
    )

    checks["duration_consistency_errors"] = int(
        (duration_difference.abs() > 1e-6).sum()
    )

    # ------------------------------------------------------------
    # Dst validation
    # ------------------------------------------------------------

    checks["invalid_min_dst"] = int(
        (
            df["min_dst_nt"]
            > df["onset_dst_nt"]
        ).sum()
    )

    checks["missing_min_dst"] = int(
        df["min_dst_nt"].isna().sum()
    )

    # ------------------------------------------------------------
    # Timestamp validation
    # ------------------------------------------------------------

    checks["missing_start_time"] = int(
        df["start_time"].isna().sum()
    )

    checks["missing_peak_time"] = int(
        df["peak_time"].isna().sum()
    )

    checks["missing_end_time"] = int(
        df["end_time"].isna().sum()
    )

    return checks


def build_summary(df):
    """
    Build numerical summary statistics for storm-level features.
    """

    numerical_columns = [
        "onset_dst_nt",
        "min_dst_nt",
        "total_duration_hours",
        "main_phase_hours",
        "recovery_phase_hours",
        "descent_rate_nt_hr",
        "recovery_to_150_hours",
    ]

    summary_rows = []

    for column in numerical_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        summary_rows.append(
            {
                "feature": column,
                "count": int(series.count()),
                "missing": int(series.isna().sum()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q25": float(series.quantile(0.25)),
                "q75": float(series.quantile(0.75)),
                "q95": float(series.quantile(0.95)),
                "max": float(series.max()),
            }
        )

    return pd.DataFrame(summary_rows)


def build_json_summary(df, checks):
    """
    Build a compact JSON summary.
    """

    return {
        "n_storms": int(len(df)),
        "validation": checks,
        "intensity_distribution": {
            str(k): int(v)
            for k, v in (
                df["intensity_tier"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "quality_flag_distribution": {
            str(k): int(v)
            for k, v in (
                df["data_quality_flag"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "data_product_distribution": {
            str(k): int(v)
            for k, v in (
                df["data_product"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
    }


def create_feature_figure(df, output_path):
    """
    Create separate distributions of the main storm-level features.
    """

    features = [
        (
            "min_dst_nt",
            "Minimum Dst (nT)",
            "Distribution of Storm Minimum Dst",
        ),
        (
            "total_duration_hours",
            "Total duration (hours)",
            "Distribution of Storm Duration",
        ),
        (
            "main_phase_hours",
            "Main phase (hours)",
            "Distribution of Main-Phase Duration",
        ),
        (
            "recovery_phase_hours",
            "Recovery phase (hours)",
            "Distribution of Recovery-Phase Duration",
        ),
    ]

    for column, xlabel, title in features:

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        ax.hist(
            values,
            bins=50,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Number of storms")

        ax.grid(
            True,
            alpha=0.25,
        )

        fig.tight_layout()

        output = (
            output_path.parent
            / (
                output_path.stem
                + "_"
                + column
                + output_path.suffix
            )
        )

        fig.savefig(
            output,
            dpi=180,
        )

        plt.close(fig)

        print(f"[OK] {output}")


def main(argv=None):

    print("=" * 70)
    print("GEOMAGNETIC STORM PROJECT - STORM-LEVEL FEATURES")
    print("=" * 70)

    cfg = load_config(None)

    # ------------------------------------------------------------
    # Load baseline catalogue
    # ------------------------------------------------------------

    print("\nLoading baseline storm catalogue...")

    df, catalogue_path = load_catalogue(cfg)

    print(f"Catalogue: {catalogue_path}")
    print(f"Storms   : {len(df):,}")

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    print("\nValidating storm-level features...")

    checks = validate_features(df)

    for key, value in checks.items():
        print(f"{key}: {value}")

    failed_checks = {
        key: value
        for key, value in checks.items()
        if key != "rows" and value != 0
    }

    if failed_checks:
        raise ValueError(
            "Storm-level feature validation failed:\n"
            + json.dumps(
                failed_checks,
                indent=2,
            )
        )

    print("[OK] Feature validation passed.")

    # ------------------------------------------------------------
    # Build numerical summary
    # ------------------------------------------------------------

    print("\nCalculating feature statistics...")

    summary = build_summary(df)

    # ------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------

    metrics_dir = resolve(
        cfg,
        "metrics_dir",
    )

    figures_dir = resolve(
        cfg,
        "figures_dir",
    )

    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Save CSV summary
    # ------------------------------------------------------------

    summary_csv = (
        metrics_dir
        / "storm_feature_summary.csv"
    )

    summary.to_csv(
        summary_csv,
        index=False,
    )

    print(f"[OK] {summary_csv}")

    # ------------------------------------------------------------
    # Save JSON summary
    # ------------------------------------------------------------

    json_summary = build_json_summary(
        df,
        checks,
    )

    summary_json = (
        metrics_dir
        / "storm_feature_summary.json"
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_summary,
            f,
            indent=2,
        )

    print(f"[OK] {summary_json}")

    # ------------------------------------------------------------
    # Create feature distributions
    # ------------------------------------------------------------

    print("\nCreating feature distributions...")

    figure_base = (
        figures_dir
        / "storm_feature_distribution.png"
    )

    create_feature_figure(
        df,
        figure_base,
    )

    # ------------------------------------------------------------
    # Print feature summary
    # ------------------------------------------------------------

    print("\n" + "-" * 70)
    print("STORM-LEVEL FEATURE SUMMARY")
    print("-" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\n" + "=" * 70)
    print("STORM-LEVEL FEATURES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()