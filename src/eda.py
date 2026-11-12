from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = ROOT / "data" / "processed" / "dst_hourly_clean.csv"
FIGURE_DIR = ROOT / "outputs" / "figures"
METRIC_DIR = ROOT / "outputs" / "metrics"



def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Clean Dst file not found: {DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE,
        parse_dates=["timestamp_utc"]
    )

    required = {"timestamp_utc", "dst_nt", "product"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Restrict the analysis to the Kyoto WDC final Dst period.
    start = pd.Timestamp("1957-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2020-12-31 23:00:00", tz="UTC")

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"], utc=True, errors="coerce"
    )
    df["dst_nt"] = pd.to_numeric(
        df["dst_nt"], errors="coerce"
    )

    df = df[
        (df["timestamp_utc"] >= start) &
        (df["timestamp_utc"] <= end)
    ].copy()

    # Only accept records identified as final-product data.
    df = df[df["product"] == "dst_final"].copy()

    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    expected = pd.date_range(start=start, end=end, freq="h")
    actual = pd.DatetimeIndex(df["timestamp_utc"])

    if df["timestamp_utc"].isna().any():
        raise ValueError("Invalid timestamps found in final-period data.")

    if df["dst_nt"].isna().any():
        raise ValueError("Invalid Dst values found in final-period data.")

    if df["timestamp_utc"].duplicated().any():
        raise ValueError("Duplicate timestamps found in final-period data.")

    if not actual.equals(expected):
        missing_hours = len(expected.difference(actual))
        unexpected_hours = len(actual.difference(expected))
        raise ValueError(
            "Final-period hourly coverage check failed: "
            f"{missing_hours} missing hours, "
            f"{unexpected_hours} unexpected timestamps."
        )

    if len(df) != 561024:
        raise ValueError(
            f"Expected 561024 final hourly rows, got {len(df)}."
        )

    return df


def validate_data(df):
    timestamps = df["timestamp_utc"]
    dst = pd.to_numeric(df["dst_nt"], errors="coerce")

    diffs = timestamps.diff().dropna()

    return {
        "rows": int(len(df)),
        "start": str(timestamps.min()),
        "end": str(timestamps.max()),
        "duplicate_timestamps": int(
            timestamps.duplicated().sum()
        ),
        "missing_dst_values": int(dst.isna().sum()),
        "timestamp_gaps_gt_1h": int(
            (diffs > pd.Timedelta(hours=1)).sum()
        ),
        "dst_min_nt": float(dst.min()),
        "dst_max_nt": float(dst.max()),
        "dst_mean_nt": float(dst.mean()),
        "dst_median_nt": float(dst.median()),
        "dst_std_nt": float(dst.std()),
        "dst_q01_nt": float(dst.quantile(0.01)),
        "dst_q05_nt": float(dst.quantile(0.05)),
        "dst_q25_nt": float(dst.quantile(0.25)),
        "dst_q75_nt": float(dst.quantile(0.75)),
        "dst_q95_nt": float(dst.quantile(0.95)),
        "dst_q99_nt": float(dst.quantile(0.99)),
    }


def plot_time_series(df):
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(
        df["timestamp_utc"],
        df["dst_nt"],
        linewidth=0.5
    )

    ax.axhline(
        -30,
        linestyle="--",
        linewidth=1.2,
        label="-30 nT storm threshold"
    )

    ax.axhline(
        -150,
        linestyle=":",
        linewidth=1.2,
        label="-150 nT reference"
    )

    ax.set_title("Kyoto WDC Dst Index")
    ax.set_xlabel("UTC time")
    ax.set_ylabel("Dst (nT)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    output = FIGURE_DIR / "dst_time_series.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"[OK] {output}")


def plot_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        df["dst_nt"].dropna(),
        bins=100
    )

    ax.axvline(
        -30,
        linestyle="--",
        linewidth=1.2,
        label="-30 nT"
    )

    ax.axvline(
        -150,
        linestyle=":",
        linewidth=1.2,
        label="-150 nT"
    )

    ax.set_title("Distribution of Hourly Dst Values")
    ax.set_xlabel("Dst (nT)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    output = FIGURE_DIR / "dst_distribution.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"[OK] {output}")


def monthly_analysis(df):
    temp = df.copy()

    temp["month"] = (
        temp["timestamp_utc"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        temp.groupby("month")["dst_nt"]
        .agg(
            mean_dst="mean",
            median_dst="median",
            min_dst="min",
            max_dst="max",
            std_dst="std",
            observations="count"
        )
        .reset_index()
    )

    monthly.to_csv(
        METRIC_DIR / "monthly_dst_statistics.csv",
        index=False
    )

    dates = pd.to_datetime(
        monthly["month"],
        format="%Y-%m"
    )

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(
        dates,
        monthly["min_dst"],
        linewidth=0.8,
        label="Monthly minimum Dst"
    )

    ax.axhline(
        -30,
        linestyle="--",
        linewidth=1.2,
        label="-30 nT"
    )

    ax.axhline(
        -150,
        linestyle=":",
        linewidth=1.2,
        label="-150 nT"
    )

    ax.set_title("Monthly Minimum Dst")
    ax.set_xlabel("Month")
    ax.set_ylabel("Dst (nT)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    output = FIGURE_DIR / "dst_monthly_statistics.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"[OK] {output}")


def annual_analysis(df):
    temp = df.copy()
    temp["year"] = temp["timestamp_utc"].dt.year

    annual = (
        temp.groupby("year")["dst_nt"]
        .agg(
            mean_dst="mean",
            median_dst="median",
            min_dst="min",
            max_dst="max",
            std_dst="std",
            observations="count"
        )
        .reset_index()
    )

    annual.to_csv(
        METRIC_DIR / "annual_dst_statistics.csv",
        index=False
    )

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.plot(
        annual["year"],
        annual["min_dst"],
        marker=".",
        linewidth=1,
        label="Annual minimum Dst"
    )

    ax.axhline(
        -30,
        linestyle="--",
        linewidth=1.2,
        label="-30 nT"
    )

    ax.axhline(
        -150,
        linestyle=":",
        linewidth=1.2,
        label="-150 nT"
    )

    ax.set_title("Annual Minimum Dst")
    ax.set_xlabel("Year")
    ax.set_ylabel("Dst (nT)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()

    output = FIGURE_DIR / "dst_annual_statistics.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"[OK] {output}")


def main():
    print("=" * 70)
    print("GEOMAGNETIC STORM PROJECT - EDA")
    print("=" * 70)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading cleaned Dst data...")

    df = load_data()

    print(f"Rows : {len(df):,}")
    print(f"Start: {df['timestamp_utc'].min()}")
    print(f"End  : {df['timestamp_utc'].max()}")

    print("\nValidating data...")

    summary = validate_data(df)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nCreating figures...")

    plot_time_series(df)
    plot_distribution(df)
    monthly_analysis(df)
    annual_analysis(df)

    summary_file = METRIC_DIR / "eda_summary.json"

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print(f"[OK] {summary_file}")

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
