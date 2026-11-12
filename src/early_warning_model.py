"""
Early Warning Storm Intensity Forecaster (6-Hour Onset Predictor)
=================================================================
Objective:
Given the initial 6 hours of Dst observations after storm onset (t = 0 to 6h),
predict the final peak storm severity category:
  - Class 0: Weak Storm      (-30 nT >= Min Dst > -50 nT)
  - Class 1: Moderate Storm  (-50 nT >= Min Dst > -100 nT)
  - Class 2: Intense+ Storm  (Min Dst <= -100 nT)

Evaluates:
1. Multi-class prediction performance (Accuracy, Macro-F1, Per-class Precision/Recall)
2. Empirical lead-time distribution: delta_t = t_peak - (t_onset + 6h)
"""

from pathlib import Path
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# Ensure root is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def map_intensity_class(min_dst: float) -> int:
    """Map minimum Dst to 3-tier severity class."""
    if min_dst > -50.0:
        return 0  # Weak
    elif min_dst > -100.0:
        return 1  # Moderate
    else:
        return 2  # Intense+ (Severe / Extreme)


CLASS_NAMES = ["Weak (-30 to -50 nT)", "Moderate (-50 to -100 nT)", "Intense+ (<= -100 nT)"]


def extract_6h_features(clean_dst_path: Path, catalogue_path: Path):
    """Extract first 6-hour Dst trajectory, lead time, & physical features for every storm."""
    print("Loading clean Dst series and storm catalogue...")
    dst_df = pd.read_csv(clean_dst_path, parse_dates=["timestamp_utc"])
    dst_df["timestamp_utc"] = pd.to_datetime(dst_df["timestamp_utc"], utc=True)
    dst_series = dst_df.sort_values("timestamp_utc").set_index("timestamp_utc")["dst_nt"]

    cat_df = pd.read_csv(catalogue_path, parse_dates=["start_time", "peak_time", "end_time"])
    cat_df["start_time"] = pd.to_datetime(cat_df["start_time"], utc=True)
    cat_df["peak_time"] = pd.to_datetime(cat_df["peak_time"], utc=True)
    cat_df["end_time"] = pd.to_datetime(cat_df["end_time"], utc=True)

    features = []
    labels = []
    storm_metadata = []

    print(f"Extracting 6-hour onset trajectories for {len(cat_df)} storms...")
    for idx, row in cat_df.iterrows():
        t0 = row["start_time"]
        t6 = t0 + pd.Timedelta(hours=6)
        t_peak = row["peak_time"]

        # Calculate empirical lead time from decision point (t0 + 6h) to peak
        lead_time_hours = (t_peak - t6).total_seconds() / 3600.0

        # Slice 7 hourly points: t0, t0+1h, ..., t0+6h
        try:
            window = dst_series.loc[t0:t6]
        except KeyError:
            continue

        if len(window) < 3:
            continue

        # If storm ended before 6 hours, pad with last available value
        vals = window.values.astype(float)
        if len(vals) < 7:
            vals = np.pad(vals, (0, 7 - len(vals)), mode="edge")
        else:
            vals = vals[:7]

        # Engineer 6-hour physical features
        dst_0 = vals[0]
        dst_3 = vals[3]
        dst_6 = vals[6]
        min_6h = np.min(vals)
        drop_6h = dst_0 - min_6h
        slope_6h = (dst_6 - dst_0) / 6.0
        accel_6h = (dst_6 - dst_3) - (dst_3 - dst_0)
        volatility_6h = np.std(vals)
        mean_6h = np.mean(vals)

        feature_vector = [
            vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6],
            min_6h, drop_6h, slope_6h, accel_6h, volatility_6h, mean_6h
        ]

        label = map_intensity_class(row["min_dst_nt"])

        features.append(feature_vector)
        labels.append(label)
        storm_metadata.append({
            "storm_id": row["storm_id"],
            "start_time": row["start_time"],
            "peak_time": row["peak_time"],
            "min_dst_nt": row["min_dst_nt"],
            "main_phase_hours": row["main_phase_hours"],
            "lead_time_hours": lead_time_hours
        })

    feature_names = [
        "dst_0h", "dst_1h", "dst_2h", "dst_3h", "dst_4h", "dst_5h", "dst_6h",
        "min_6h", "drop_6h", "slope_6h", "accel_6h", "volatility_6h", "mean_6h"
    ]

    X = np.array(features)
    y = np.array(labels)
    meta_df = pd.DataFrame(storm_metadata)

    return X, y, meta_df, feature_names


def train_and_evaluate_early_warning():
    clean_csv = ROOT_DIR / "data" / "processed" / "dst_hourly_clean.csv"
    cat_csv = ROOT_DIR / "data" / "catalogue" / "storm_catalogue.csv"
    output_dir = ROOT_DIR / "outputs" / "models"
    metrics_dir = ROOT_DIR / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    X, y, meta_df, feature_names = extract_6h_features(clean_csv, cat_csv)

    # Chronological splits matching project specs:
    # Train: 1957 - 2009
    # Val:   2010 - 2015
    # Test:  2016 - 2020
    train_mask = meta_df["start_time"] < "2010-01-01 00:00:00+00:00"
    val_mask = (meta_df["start_time"] >= "2010-01-01 00:00:00+00:00") & (meta_df["start_time"] < "2016-01-01 00:00:00+00:00")
    test_mask = meta_df["start_time"] >= "2016-01-01 00:00:00+00:00"

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    meta_test = meta_df[test_mask].reset_index(drop=True)

    print("\nDataset Split Summary:")
    print(f"  Training Storms:   {len(y_train):,} (1957 - 2009)")
    print(f"  Validation Storms: {len(y_val):,} (2010 - 2015)")
    print(f"  Test Storms:       {len(y_test):,} (2016 - 2020)")

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Early Warning Gradient Boosting Classifier
    print("\nTraining Early Warning Forecaster (Gradient Boosting Ensemble)...")
    model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Predict on unseen test storms (2016-2020)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)

    # Calculate empirical lead time statistics
    test_lead_times = meta_test["lead_time_hours"].values
    pos_lead_times = test_lead_times[test_lead_times > 0]
    lead_stats = {
        "mean_lead_time_hours": float(np.mean(test_lead_times)),
        "median_lead_time_hours": float(np.median(test_lead_times)),
        "p25_lead_time_hours": float(np.percentile(test_lead_times, 25)),
        "p75_lead_time_hours": float(np.percentile(test_lead_times, 75)),
        "pct_storms_with_advance_lead_time": float((test_lead_times > 0).mean() * 100),
        "mean_positive_lead_time_hours": float(np.mean(pos_lead_times)) if len(pos_lead_times) > 0 else 0.0,
        "median_positive_lead_time_hours": float(np.median(pos_lead_times)) if len(pos_lead_times) > 0 else 0.0
    }

    # Intense+ specific lead time
    intense_mask = (y_test == 2)
    intense_lead_times = test_lead_times[intense_mask]
    if len(intense_lead_times) > 0:
        lead_stats["intense_storms_count"] = int(np.sum(intense_mask))
        lead_stats["intense_mean_lead_time_hours"] = float(np.mean(intense_lead_times))
        lead_stats["intense_median_lead_time_hours"] = float(np.median(intense_lead_times))
        lead_stats["intense_individual_lead_times"] = [float(x) for x in intense_lead_times]

    print("\n" + "=" * 65)
    print("  EARLY WARNING TEST RESULTS (6-Hour Onset -> Peak Intensity Class)")
    print("  Test Evaluation Period: 2016 - 2020 (Unseen Test Storms)")
    print("=" * 65)
    print(f"  Overall Accuracy:    {acc*100:.2f}%")
    print(f"  Macro F1-Score:      {f1_macro*100:.2f}%")
    print(f"  Weighted F1-Score:   {f1_weighted*100:.2f}%")

    print("\n  Confusion Matrix (Rows=True Class, Columns=Predicted Class):")
    print("                 Pred: Weak | Pred: Moderate | Pred: Intense+")
    for i, row in enumerate(cm):
        print(f"  True {CLASS_NAMES[i]:<20} {row[0]:>8} {row[1]:>14} {row[2]:>14}")

    print("\n  Detailed Classification Report:")
    report_dict = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    print("\n  Empirical Lead-Time Analysis (from Hour 6 to Peak Minimum):")
    print(f"    - Median Lead Time (all storms):     {lead_stats['median_lead_time_hours']:.1f} hours")
    print(f"    - Mean Lead Time (positive storms):  {lead_stats['mean_positive_lead_time_hours']:.1f} hours")
    print(f"    - Storms with >0h advance warning:   {lead_stats['pct_storms_with_advance_lead_time']:.1f}%")
    if "intense_mean_lead_time_hours" in lead_stats:
        print(f"    - Intense+ Storms Average Lead Time: {lead_stats['intense_mean_lead_time_hours']:.1f} hours")

    # Feature Importance Analysis
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\n  Top Predictive Features (Physics Interpretation):")
    for rank, idx in enumerate(sorted_idx[:5], 1):
        print(f"    {rank}. {feature_names[idx]:<15} (Importance: {importances[idx]*100:.1f}%)")

    # Save model and artifacts
    model_save_path = output_dir / "early_warning_model.joblib"
    scaler_save_path = output_dir / "early_warning_scaler.joblib"
    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"\nSaved trained model to:  {model_save_path}")
    print(f"Saved feature scaler to: {scaler_save_path}")

    # Save metrics JSON
    summary = {
        "task": "Storm Peak Severity Prediction from Initial 6 Hours",
        "evaluation_period": "2016-2020",
        "test_storms": len(y_test),
        "accuracy": acc,
        "macro_f1": f1_macro,
        "weighted_f1": f1_weighted,
        "confusion_matrix": cm.tolist(),
        "class_report": report_dict,
        "lead_time_statistics": lead_stats,
        "top_features": [
            {"feature": feature_names[i], "importance": float(importances[i])}
            for i in sorted_idx
        ]
    }
    with open(metrics_dir / "early_warning_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return model, scaler, summary


if __name__ == "__main__":
    train_and_evaluate_early_warning()
