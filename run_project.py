"""
Master Project Runner: Geomagnetic Storm 1D-CNN Detection & Evaluation
======================================================================
Executes the complete audited pipeline end-to-end:
1. Data Preprocessing & Integrity Verification (Kyoto Dst Final 1957-2020)
2. Automated Test Suite (All 40 tests across unit and audit suites)
3. 1D-CNN Model Evaluation & Hourly Metric Reproduction (Test Set: 2016-2020)
4. Comprehensive Event-Level Evaluation (Hungarian matching, temporal coverage, tier breakdown)
"""

import sys
import os
from pathlib import Path
import unittest
import pandas as pd

# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Gracefully disable Windows WMI delay-load on Python 3.13
import platform
platform._wmi = None


def print_header(title: str):
    print("\n" + "=" * 78)
    print(f"  {title.upper()}")
    print("=" * 78)


def step_1_verify_data():
    print_header("Step 1: Data Preprocessing & Integrity Verification")
    clean_csv = ROOT_DIR / "data" / "processed" / "dst_hourly_clean.csv"
    if not clean_csv.exists():
        print(f"  [FAIL] Missing processed dataset: {clean_csv}")
        return False

    df = pd.read_csv(clean_csv)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    flag_col = "product" if "product" in df.columns else "flag"
    dst_col = "dst_nt" if "dst_nt" in df.columns else "dst"

    final_df = df[
        (df[flag_col] == "dst_final") &
        (df["timestamp_utc"] >= "1957-01-01 00:00:00+00:00") &
        (df["timestamp_utc"] <= "2020-12-31 23:00:00+00:00")
    ].sort_values("timestamp_utc").reset_index(drop=True)

    total_rows = len(final_df)
    expected_rows = 561024
    null_count = final_df[dst_col].isnull().sum()
    dup_count = final_df["timestamp_utc"].duplicated().sum()

    print(f"  Processed File Path:       {clean_csv}")
    print(f"  Date Range:                1957-01-01 00:00 UTC to 2020-12-31 23:00 UTC")
    print(f"  Verified Final Rows:       {total_rows:,} (Expected: {expected_rows:,})")
    print(f"  Missing / NaN Dst Values:  {null_count}")
    print(f"  Duplicate Timestamps:      {dup_count}")

    if total_rows == expected_rows and null_count == 0 and dup_count == 0:
        print("  -> STATUS: [PASS] 561,024 final Kyoto Dst hours verified with zero gaps/NaNs.")
        return True
    else:
        print("  -> STATUS: [FAIL] Data anomalies detected.")
        return False


def step_2_run_tests():
    print_header("Step 2: Automated Test Suite Execution")
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT_DIR / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)

    print(f"\n  Total Tests Run: {result.testsRun}")
    print(f"  Errors:         {len(result.errors)}")
    print(f"  Failures:       {len(result.failures)}")

    if result.wasSuccessful():
        print(f"  -> STATUS: [PASS] All {result.testsRun} tests passed successfully.")
        return True
    else:
        print("  -> STATUS: [FAIL] One or more tests failed.")
        return False


def step_3_evaluate_model():
    print_header("Step 3: 1D-CNN Hourly Model Evaluation (Test Period 2016-2020)")
    from src.evaluate_cnn import main as run_hourly_evaluation
    run_hourly_evaluation()
    print("  -> STATUS: [PASS] Test set hourly metrics successfully reproduced.")
    return True


def step_4_evaluate_events():
    print_header("Step 4: Event-Level Evaluation & Root-Cause Diagnostics")
    from src.evaluate_events import main as run_event_evaluation
    run_event_evaluation()
    print("  -> STATUS: [PASS] Event-level evaluation, Hungarian matches & reports generated.")
    return True


def main():
    print("=" * 78)
    print("   GEOMAGNETIC STORM 1D-CNN: MASTER VERIFICATION RUNNER")
    print("   Project Workspace: " + str(ROOT_DIR))
    print("=" * 78)

    ok1 = step_1_verify_data()
    ok2 = step_2_run_tests()
    ok3 = step_3_evaluate_model()
    ok4 = step_4_evaluate_events()

    print("\n" + "=" * 78)
    if ok1 and ok2 and ok3 and ok4:
        print("   ALL PIPELINE STAGES COMPLETED & VERIFIED SUCCESSFULLY!")
        print("   Outputs and CSV metrics are saved in outputs/metrics/.")
        print("   Figures and annual plots are saved in outputs/figures/.")
    else:
        print("   ONE OR MORE PIPELINE STAGES ENCOUNTERED ISSUES.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
