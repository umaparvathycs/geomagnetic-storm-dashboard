"""
Clean and validate Kyoto WDC Dst hourly data.

Handles Kyoto HTML rows where negative values may be concatenated, e.g.:

    10  -32 ... -50-112-176-201-220-243-303
    11  -309-333-296-283-276...

The parser extracts signed integers rather than relying on whitespace.
Supports WDC fixed-width format (DSTyyMM...) and HTML table data.

Outputs:
    data/processed/dst_hourly_clean.csv
    data/processed/data_quality_report.json
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .common import load_config as common_load_config


PRODUCT_RANK = {
    "dst_final": 0,
    "dst_provisional": 1,
    "dst_realtime": 2,
}


def load_config() -> dict:
    """Load project configuration."""
    try:
        return common_load_config()
    except Exception:
        config_path = Path("config.yaml")
        if not config_path.exists():
            raise FileNotFoundError("config.yaml was not found.")
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


def extract_pre_data(text: str) -> str:
    """
    Extract the contents of <pre class="data">...</pre>.
    """
    match = re.search(
        r'<pre\s+class=["\']data["\'][^>]*>(.*?)</pre>',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return text

    return html.unescape(match.group(1))


def parse_year_month_from_filename(path: Path) -> tuple[int, int]:
    """
    Extract YYYYMM from filenames such as 195701.html.
    """
    match = re.search(r"(\d{6})", path.stem)

    if not match:
        raise ValueError(f"Cannot determine YYYYMM from {path.name}")

    tag = match.group(1)
    year = int(tag[:4])
    month = int(tag[4:])

    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month in {path.name}")

    return year, month


def parse_wdc_text(
    source: Path | str,
    product: str = "dst_final",
    source_name: str | None = None,
) -> pd.DataFrame:
    """
    Parse one Kyoto WDC monthly HTML file, raw WDC text, or text snippet.
    """
    source_file = source_name or "text"
    year: int | None = None
    month: int | None = None

    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).is_file()
    ):
        path = Path(source)
        source_file = path.name
        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        )
        try:
            year, month = parse_year_month_from_filename(path)
        except Exception:
            year, month = None, None
    else:
        text = str(source)

    pre = extract_pre_data(text)

    rows = []
    integer_pattern = re.compile(r"[+-]?\d+")

    for line in pre.splitlines():
        line = line.strip()

        if not line:
            continue

        # Check for fixed-width WDC format: "DSTyyMM*ddRRX0ccBASE..."
        if line.startswith("DST") and len(line) >= 20:
            try:
                yy = int(line[3:5])
                mm = int(line[5:7])
                dd = int(line[8:10])
                century = int(line[14:16])
                row_year = century * 100 + yy
                hourly_values = [
                    int(line[20 + 4 * i : 24 + 4 * i]) for i in range(24)
                ]
                date = pd.Timestamp(
                    year=row_year,
                    month=mm,
                    day=dd,
                    tz="UTC",
                )
                for hour, dst in enumerate(hourly_values):
                    timestamp = date + pd.Timedelta(hours=hour)
                    rows.append(
                        {
                            "timestamp_utc": timestamp,
                            "dst_raw": float(dst),
                            "dst_nt": float(dst),
                            "product": product,
                            "source_file": source_file,
                            "source": source_file,
                        }
                    )
                continue
            except Exception:
                pass

        numbers = integer_pattern.findall(line)

        # Standard HTML day row: day + 24 hourly values
        if len(numbers) < 25:
            continue

        try:
            day = int(numbers[0])
        except ValueError:
            continue

        if not 1 <= day <= 31:
            continue

        hourly_values = []
        for value in numbers[1:25]:
            try:
                hourly_values.append(int(value))
            except ValueError:
                hourly_values = []
                break

        if len(hourly_values) != 24:
            continue

        if year is None or month is None:
            continue

        try:
            date = pd.Timestamp(
                year=year,
                month=month,
                day=day,
                tz="UTC",
            )
        except ValueError:
            continue

        for hour, dst in enumerate(hourly_values):
            timestamp = date + pd.Timedelta(hours=hour)
            rows.append(
                {
                    "timestamp_utc": timestamp,
                    "dst_raw": float(dst),
                    "dst_nt": float(dst),
                    "product": product,
                    "source_file": source_file,
                    "source": source_file,
                }
            )

    if not rows:
        raise ValueError(
            f"{product}/{source_file}: no WDC Dst day rows were found."
        )

    return pd.DataFrame(rows)


def read_csv_series(
    csv_path: str | Path,
    product: str = "dst_provisional",
    missing_val: int = 9999,
) -> pd.DataFrame:
    """
    Read an external or newly provided CSV series into standard raw schema.
    """
    path = Path(csv_path)
    df = pd.read_csv(path)

    if "timestamp_utc" not in df.columns:
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "timestamp_utc"})

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    val_col = (
        "dst_raw"
        if "dst_raw" in df.columns
        else ("dst_nt" if "dst_nt" in df.columns else df.columns[1])
    )

    df["dst_raw"] = (
        pd.to_numeric(df[val_col], errors="coerce")
        .fillna(missing_val)
        .astype(int)
    )
    df["dst_nt"] = df["dst_raw"].where(df["dst_raw"] != missing_val, np.nan)
    df["product"] = product
    df["source"] = path.name
    df["source_file"] = path.name

    return df[
        [
            "timestamp_utc",
            "dst_raw",
            "dst_nt",
            "product",
            "source",
            "source_file",
        ]
    ]


def build_raw(cfg: dict) -> pd.DataFrame:
    """
    Parse all configured monthly HTML files.
    """
    all_frames = []

    # Check data section or top-level products
    products = (
        cfg.get("data", {}).get("products", [])
        if "data" in cfg
        else cfg.get("products", {})
    )

    if isinstance(products, list):
        # Master config style: list of product dicts
        raw_root = Path(cfg.get("paths", {}).get("raw_dir", "data/raw"))
        for p in products:
            p_name = p.get("name")
            p_dir = raw_root / p_name
            if not p_dir.exists():
                continue
            files = sorted(p_dir.glob("*.html"))
            for path in files:
                try:
                    df = parse_wdc_text(path, p_name)
                    all_frames.append(df)
                except Exception as exc:
                    print(f"  [ERROR] {path.name}: {exc}")
    elif isinstance(products, dict):
        for product, product_cfg in products.items():
            raw_dir = Path(product_cfg.get("raw_dir", ""))
            if not raw_dir.exists():
                continue
            files = sorted(raw_dir.glob("*.html"))
            for path in files:
                try:
                    df = parse_wdc_text(path, product)
                    all_frames.append(df)
                except Exception as exc:
                    print(f"  [ERROR] {path.name}: {exc}")

    if not all_frames:
        raise RuntimeError("No Dst data could be parsed from data/raw.")

    full = pd.concat(all_frames, ignore_index=True)
    full["timestamp_utc"] = pd.to_datetime(
        full["timestamp_utc"], utc=True, errors="coerce"
    )
    full["dst_raw"] = pd.to_numeric(full["dst_raw"], errors="coerce")
    full["dst_nt"] = pd.to_numeric(full["dst_nt"], errors="coerce")
    full = full.dropna(subset=["timestamp_utc"])

    return full


def clean_series(
    raw: pd.DataFrame,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Select the highest-priority product when multiple products
    contain the same timestamp, and reindex onto a complete continuous grid.
    """
    df = raw.copy()

    if "product" not in df.columns:
        df["product"] = "dst_final"

    df["product_rank"] = (
        df["product"].map(PRODUCT_RANK).fillna(999).astype(int)
    )

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    # Sort by timestamp and product rank
    df = df.sort_values(["timestamp_utc", "product_rank"])

    duplicate_count = int(
        df.duplicated(subset=["timestamp_utc"], keep=False).sum()
    )

    # Keep highest-quality product first per timestamp
    df = df.drop_duplicates(subset=["timestamp_utc"], keep="first").reset_index(
        drop=True
    )

    start = df["timestamp_utc"].min()
    end = df["timestamp_utc"].max()

    if pd.isna(start) or pd.isna(end):
        raise RuntimeError("No valid timestamps remain after cleaning.")

    expected = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    df = df.set_index("timestamp_utc").reindex(expected)
    df.index.name = "timestamp_utc"

    # Value column handling
    missing_val = 9999
    plausible_max = 1000.0

    if cfg:
        d_cfg = cfg.get("data", cfg)
        missing_val = d_cfg.get("missing_value", 9999)
        plausible_max = float(d_cfg.get("plausible_abs_max_nt", 1000.0))

    val_series = df["dst_raw"] if "dst_raw" in df.columns else df["dst_nt"]

    # Valid mask: finite, not missing marker, within physical range
    valid_mask = (
        val_series.notna()
        & (val_series != missing_val)
        & (val_series.abs() <= plausible_max)
    )

    df["dst_nt"] = val_series.where(valid_mask, np.nan)

    if "product" in df.columns:
        df["product"] = df["product"].ffill().bfill().fillna("dst_final")
    else:
        df["product"] = "dst_final"

    src_col = (
        "source_file"
        if "source_file" in df.columns
        else ("source" if "source" in df.columns else None)
    )
    if src_col:
        df["source_file"] = df[src_col].ffill().bfill().fillna("clean")
    else:
        df["source_file"] = "clean"
    df["source"] = df["source_file"]

    clean = df.reset_index()
    if "dst_raw" in df.columns:
        clean["dst_raw"] = df["dst_raw"].values

    missing_hours = int(clean["dst_nt"].isna().sum())
    is_na = clean["dst_nt"].isna()
    n_gaps = int((is_na & ~is_na.shift(fill_value=False)).sum())

    hours_by_prod = clean.groupby("product")["dst_nt"].count().to_dict()

    report = {
        "first_timestamp": str(start),
        "last_timestamp": str(end),
        "start_utc": str(start),
        "end_utc": str(end),
        "observed_hours": int(len(clean) - missing_hours),
        "expected_hours": int(len(clean)),
        "missing_hours": missing_hours,
        "n_gaps": n_gaps,
        "hours_by_product": hours_by_prod,
        "interpolated_hours": 0,
        "duplicate_timestamps": duplicate_count,
        "minimum_dst_nt": (
            float(clean["dst_nt"].min())
            if clean["dst_nt"].notna().any()
            else np.nan
        ),
        "maximum_dst_nt": (
            float(clean["dst_nt"].max())
            if clean["dst_nt"].notna().any()
            else np.nan
        ),
        "mean_dst_nt": (
            float(clean["dst_nt"].mean())
            if clean["dst_nt"].notna().any()
            else np.nan
        ),
        "median_dst_nt": (
            float(clean["dst_nt"].median())
            if clean["dst_nt"].notna().any()
            else np.nan
        ),
        "products": sorted(clean["product"].dropna().unique().tolist()),
    }

    return clean, report


def validate_clean(
    df: pd.DataFrame,
    cfg: dict | None = None,
) -> list[str]:
    """
    Validate the cleaned dataset. Returns a list of problem strings.
    """
    problems: list[str] = []

    required = {"timestamp_utc", "dst_nt", "product"}
    missing_columns = required.difference(df.columns)

    if missing_columns:
        problems.append(f"Missing required columns: {sorted(missing_columns)}")

    if df.empty:
        problems.append("Clean dataset is empty.")

    if "timestamp_utc" in df.columns:
        if df["timestamp_utc"].isna().any():
            problems.append("Clean dataset contains invalid timestamps.")
        if not df["timestamp_utc"].is_monotonic_increasing:
            problems.append("Timestamps are not sorted.")
        if df["timestamp_utc"].duplicated().any():
            problems.append("Duplicate timestamps remain.")

    return problems


def check_known_file() -> None:
    """
    Parser sanity check using January 1957.
    """
    path = Path("data/raw/dst_final/195701.html")

    if not path.exists():
        raise FileNotFoundError(f"Known test file not found: {path}")

    df = parse_wdc_text(path, "dst_final")

    print("Known-file parser check: PASS")
    print(f"Rows parsed: {len(df)}")
    print(f"Start: {df['timestamp_utc'].min()}")
    print(f"End:   {df['timestamp_utc'].max()}")
    print(f"Minimum Dst: {df['dst_nt'].min():g} nT")
    print(f"Maximum Dst: {df['dst_nt'].max():g} nT")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-known",
        action="store_true",
        help="Run parser test on January 1957.",
    )

    args = parser.parse_args()

    if args.check_known:
        check_known_file()
        return

    cfg = load_config()
    raw = build_raw(cfg)

    print(f"\nTotal parsed records: {len(raw)}")
    clean, report = clean_series(raw, cfg)

    print(
        f"Duplicate timestamp records before product selection: "
        f"{report['duplicate_timestamps']}"
    )
    print(f"Clean records: {len(clean)}")

    problems = validate_clean(clean, cfg)
    if problems:
        raise ValueError("Clean data validation failed: " + "; ".join(problems))

    print("\nValidation: PASS")
    print(f"Rows: {len(clean)}")
    print(f"Start: {clean['timestamp_utc'].min()}")
    print(f"End:   {clean['timestamp_utc'].max()}")
    print(f"Minimum Dst: {clean['dst_nt'].min():g} nT")
    print(f"Maximum Dst: {clean['dst_nt'].max():g} nT")

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "dst_hourly_clean.csv"
    report_path = output_dir / "data_quality_report.json"

    clean.to_csv(csv_path, index=False)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {report_path}")
    print("\nCleaning complete.")


if __name__ == "__main__":
    main()
