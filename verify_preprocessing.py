import pandas as pd
from pathlib import Path
from src.data_cleaning import parse_wdc_text

tests = [
    ("1957-01", "dst_final"),
    ("1989-03", "dst_final"),
    ("2000-01", "dst_final"),
    ("2020-12", "dst_final"),
    ("2025-01", "dst_provisional"),
]

clean = pd.read_csv(
    "data/processed/dst_hourly_clean.csv",
    parse_dates=["timestamp_utc"],
)

clean["timestamp_utc"] = pd.to_datetime(
    clean["timestamp_utc"],
    utc=True,
)

print()
print(
    f"{'MONTH':<10} {'PRODUCT':<17} {'RAW':>5} "
    f"{'MATCH':>7} {'MISMATCH':>10} {'MISSING':>9} {'RESULT':>8}"
)
print("-" * 75)

results = []

for month, product in tests:

    filename = month.replace("-", "") + ".html"
    path = Path("data/raw") / product / filename

    if not path.exists():
        print(
            f"{month:<10} {product:<17} "
            f"{'FILE NOT FOUND':>5}"
        )
        results.append(False)
        continue

    raw = parse_wdc_text(path, product)

    matched = raw.merge(
        clean,
        on="timestamp_utc",
        suffixes=("_raw", "_clean"),
    )

    mismatches = int(
        (matched["dst_nt_raw"] != matched["dst_nt_clean"]).sum()
    )

    missing = len(raw) - len(matched)

    passed = (
        len(raw) == len(matched)
        and mismatches == 0
        and missing == 0
    )

    results.append(passed)

    print(
        f"{month:<10} {product:<17} "
        f"{len(raw):>5} "
        f"{len(matched):>7} "
        f"{mismatches:>10} "
        f"{missing:>9} "
        f"{'PASS' if passed else 'FAIL':>8}"
    )

print("-" * 75)
print(
    "OVERALL:",
    "PASS" if all(results) else "FAIL"
)
print()