"""
Evaluate CNN predictions at the geomagnetic storm-event level.

Provides:
1. Strict 1-to-1 bipartite Hungarian matching (primary baseline for isolated storm identification).
2. Reference storm recognition / coverage analysis (measuring physical storm detection rate).
3. Detailed breakdown by intensity tier (Weak, Moderate, Intense).
4. Detailed breakdown by year (2016-2020).
5. Timing error (onset delay) and duration error analysis.
6. Diagnostic merge-gap sensitivity analysis (0 to 12 hours).
7. Physical analysis of false detections (sub-threshold depressions vs non-storm noise).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .common import load_config, resolve
from .cnn_dataset import build_dataset
from .cnn_model import StormCNN


def compatible_time(value, timezone):
    """Convert a timestamp to the data's timezone."""
    ts = pd.Timestamp(value)
    if timezone is not None:
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone)
        else:
            ts = ts.tz_convert(timezone)
    return ts


def make_predicted_events(times, predictions, dst, merge_gap_hours):
    """
    Merge positive hourly predictions into predicted events.

    Events separated by up to merge_gap_hours non-storm hours
    are merged, unless missing Dst observations separate them.
    """
    positive_times = pd.DatetimeIndex(
        times[np.asarray(predictions).astype(bool)]
    )

    if len(positive_times) == 0:
        return pd.DataFrame(
            columns=["predicted_event_id", "start_time", "end_time"]
        )

    events = []
    start = previous = positive_times[0]

    for current in positive_times[1:]:
        delta = (current - previous).total_seconds() / 3600.0
        gap = delta - 1.0

        # Do not merge across missing Dst observations.
        interval = dst.loc[previous:current]
        missing_between = interval.isna().any()

        if (
            delta <= merge_gap_hours + 1.0
            and gap >= 0
            and not missing_between
        ):
            previous = current
            continue

        events.append((start, previous))
        start = previous = current

    events.append((start, previous))

    return pd.DataFrame(
        [
            {
                "predicted_event_id": f"CNN_{i:05d}",
                "start_time": start_time,
                "end_time": end_time,
            }
            for i, (start_time, end_time) in enumerate(events, 1)
        ]
    )


def compute_overlap_matrix(reference: pd.DataFrame, predicted: pd.DataFrame) -> np.ndarray:
    """Compute pairwise temporal overlap in hours between reference and predicted intervals."""
    overlap = np.zeros((len(reference), len(predicted)), dtype=np.int64)
    if len(reference) == 0 or len(predicted) == 0:
        return overlap

    for i, ref in reference.iterrows():
        for j, pred in predicted.iterrows():
            left = max(ref["start_time"], pred["start_time"])
            right = min(ref["end_time"], pred["end_time"])
            if left <= right:
                overlap[i, j] = int((right - left).total_seconds() // 3600) + 1
    return overlap


def run_merge_gap_sensitivity(times, predictions, dst, reference: pd.DataFrame, gaps=(0, 1, 2, 3, 4, 5, 6, 8, 12)):
    """Run sensitivity analysis across different event merge gaps."""
    records = []
    for g in gaps:
        pred_df = make_predicted_events(times, predictions, dst, g)
        if len(pred_df) == 0:
            records.append({
                "merge_gap_hours": g, "predicted_events": 0, "matched_1to1": 0,
                "precision_1to1": 0.0, "recall_1to1": 0.0, "f1_1to1": 0.0,
                "ref_coverage_rate": 0.0, "multi_ref_predictions": 0,
            })
            continue

        ov = compute_overlap_matrix(reference, pred_df)
        r_idx, p_idx = linear_sum_assignment(-ov)
        tp = sum(1 for i, j in zip(r_idx, p_idx) if ov[i, j] > 0)
        p = tp / len(pred_df) if len(pred_df) else 0.0
        r = tp / len(reference) if len(reference) else 0.0
        f = (2 * p * r / (p + r)) if (p + r) else 0.0

        ref_cov = (ov.sum(axis=1) > 0).sum() / len(reference) if len(reference) else 0.0
        multi_ref = (np.count_nonzero(ov > 0, axis=0) > 1).sum()

        records.append({
            "merge_gap_hours": g,
            "predicted_events": len(pred_df),
            "matched_1to1": tp,
            "precision_1to1": round(p, 4),
            "recall_1to1": round(r, 4),
            "f1_1to1": round(f, 4),
            "ref_coverage_rate": round(ref_cov, 4),
            "multi_ref_predictions": int(multi_ref),
        })
    return pd.DataFrame(records)


def main():
    cfg = load_config()
    c = cfg["cnn"]

    # Use final Kyoto Dst data only, January 1957–December 2020.
    raw = pd.read_csv(
        resolve(cfg, "clean_series"),
        parse_dates=["timestamp_utc"],
    )

    raw = raw[
        (raw["product"] == "dst_final")
        & (raw["timestamp_utc"] >= "1957-01-01")
        & (raw["timestamp_utc"] < "2021-01-01")
    ].copy()

    series = (
        raw.sort_values("timestamp_utc")
        .set_index("timestamp_utc")["dst_nt"]
    )
    series.index = pd.DatetimeIndex(series.index)

    catalogue = pd.read_csv(
        resolve(cfg, "catalogue"),
        parse_dates=["start_time", "peak_time", "end_time"],
    )

    # Build the same dataset and split used for hourly evaluation.
    data, dst, _ = build_dataset(series, catalogue, cfg)
    test = data["test"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        resolve(cfg, "models_dir") / "storm_cnn.pt",
        map_location=device,
        weights_only=False,
    )

    model = StormCNN(window=int(checkpoint["window"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    probabilities = []
    X = torch.from_numpy(test["X"])

    with torch.no_grad():
        for start in range(0, len(X), 8192):
            batch = X[start:start + 8192].to(device)
            probabilities.extend(
                torch.sigmoid(model(batch)).cpu().numpy().ravel()
            )

    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    test_times = pd.DatetimeIndex(test["end_time"])

    # Hourly metrics
    true_labels = test.get("y")
    y_true = np.asarray(true_labels).astype(int) if true_labels is not None else None

    hourly_metrics = {}
    if y_true is not None:
        hourly_metrics = {
            "test_windows": int(len(y_true)),
            "accuracy": float(accuracy_score(y_true, predictions)),
            "precision": float(precision_score(y_true, predictions, zero_division=0)),
            "recall": float(recall_score(y_true, predictions, zero_division=0)),
            "f1": float(f1_score(y_true, predictions, zero_division=0)),
            "pr_auc": float(average_precision_score(y_true, probabilities)),
            "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        }

    # Save hourly model outputs for diagnosis and reproducibility.
    metrics_dir = resolve(cfg, "metrics_dir")
    figures_dir = resolve(cfg, "figures_dir")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    hourly = pd.DataFrame({
        "timestamp_utc": test_times,
        "probability": probabilities,
        "prediction": predictions,
    })
    if y_true is not None:
        hourly["true_label"] = y_true

    hourly_path = metrics_dir / "cnn_hourly_predictions.csv"
    hourly.to_csv(hourly_path, index=False)
    print(f"Saved hourly predictions to: {hourly_path}")

    merge_gap = int(c.get("event_merge_gap_hours", 6))

    predicted = make_predicted_events(
        test_times,
        predictions,
        dst,
        merge_gap,
    )

    # Evaluation coverage is limited to timestamps represented by valid test windows.
    coverage_start = test_times.min()
    coverage_end = test_times.max()
    timezone = dst.index.tz

    # Select catalogue events intersecting the evaluation coverage.
    reference_rows = []
    for i, row in enumerate(catalogue.itertuples(index=False), 1):
        start = compatible_time(row.start_time, timezone)
        end = compatible_time(row.end_time, timezone)

        if end < coverage_start or start > coverage_end:
            continue

        reference_rows.append({
            "reference_event_id": f"REF_{i:05d}",
            "start_time": max(start, coverage_start),
            "end_time": min(end, coverage_end),
            "intensity_tier": getattr(row, "intensity_tier", "Unknown"),
            "min_dst_nt": float(getattr(row, "min_dst_nt", np.nan)),
            "data_quality_flag": getattr(row, "data_quality_flag", "ok"),
        })

    reference = pd.DataFrame(reference_rows)

    # Pairwise overlap matrix
    overlap = compute_overlap_matrix(reference, predicted)

    # 1. Strict 1-to-1 Hungarian matching
    matched_pairs = []
    if overlap.size:
        ref_indices, pred_indices = linear_sum_assignment(-overlap)
        for i, j in zip(ref_indices, pred_indices):
            if overlap[i, j] > 0:
                matched_pairs.append((i, j, int(overlap[i, j])))

    matched_refs = {i for i, _, _ in matched_pairs}
    matched_preds = {j for _, j, _ in matched_pairs}

    tp = len(matched_pairs)
    missed = len(reference) - tp
    false_detections = len(predicted) - tp

    precision = tp / len(predicted) if len(predicted) else 0.0
    recall = tp / len(reference) if len(reference) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    # Timing errors and overlap stats for 1-to-1 matches
    match_rows = []
    onset_delays = []
    duration_errors = []
    overlap_durations = []

    for i, j, hours in matched_pairs:
        r_start = reference.iloc[i]["start_time"]
        r_end = reference.iloc[i]["end_time"]
        p_start = predicted.iloc[j]["start_time"]
        p_end = predicted.iloc[j]["end_time"]

        onset_delay = (p_start - r_start).total_seconds() / 3600.0
        r_dur = (r_end - r_start).total_seconds() / 3600.0 + 1.0
        p_dur = (p_end - p_start).total_seconds() / 3600.0 + 1.0
        dur_err = p_dur - r_dur
        cov_frac = hours / r_dur

        onset_delays.append(onset_delay)
        duration_errors.append(dur_err)
        overlap_durations.append(hours)

        match_rows.append({
            "reference_event_id": reference.iloc[i]["reference_event_id"],
            "reference_start": r_start,
            "reference_end": r_end,
            "reference_tier": reference.iloc[i]["intensity_tier"],
            "reference_min_dst": reference.iloc[i]["min_dst_nt"],
            "predicted_event_id": predicted.iloc[j]["predicted_event_id"],
            "predicted_start": p_start,
            "predicted_end": p_end,
            "overlap_hours": hours,
            "onset_error_hours": round(onset_delay, 2),
            "duration_error_hours": round(dur_err, 2),
            "coverage_fraction": round(cov_frac, 4),
            "status": "matched",
        })

    for i, row in reference.iterrows():
        if i not in matched_refs:
            r_dur = (row["end_time"] - row["start_time"]).total_seconds() / 3600.0 + 1.0
            # Check if overlapped by any prediction (even if unassigned in 1-to-1)
            row_overlaps = overlap[i, :]
            has_ov = row_overlaps.sum() > 0
            max_ov = int(row_overlaps.max()) if has_ov else 0
            cov_frac = max_ov / r_dur if has_ov else 0.0

            match_rows.append({
                "reference_event_id": row["reference_event_id"],
                "reference_start": row["start_time"],
                "reference_end": row["end_time"],
                "reference_tier": row["intensity_tier"],
                "reference_min_dst": row["min_dst_nt"],
                "predicted_event_id": None,
                "predicted_start": None,
                "predicted_end": None,
                "overlap_hours": max_ov,
                "onset_error_hours": np.nan,
                "duration_error_hours": np.nan,
                "coverage_fraction": round(cov_frac, 4),
                "status": "missed_storm",
            })

    for j, row in predicted.iterrows():
        if j not in matched_preds:
            p_start = row["start_time"]
            p_end = row["end_time"]
            seg_dst = dst.loc[p_start:p_end]
            min_dst_in_pred = float(seg_dst.min()) if len(seg_dst) else np.nan

            match_rows.append({
                "reference_event_id": None,
                "reference_start": None,
                "reference_end": None,
                "reference_tier": None,
                "reference_min_dst": min_dst_in_pred,
                "predicted_event_id": row["predicted_event_id"],
                "predicted_start": p_start,
                "predicted_end": p_end,
                "overlap_hours": 0,
                "onset_error_hours": np.nan,
                "duration_error_hours": np.nan,
                "coverage_fraction": 0.0,
                "status": "false_detection",
            })

    # Save main event matches
    matches_df = pd.DataFrame(match_rows)
    matches_df.to_csv(metrics_dir / "cnn_event_matches.csv", index=False)
    predicted.to_csv(metrics_dir / "cnn_predicted_events.csv", index=False)
    reference.to_csv(metrics_dir / "reference_test_events.csv", index=False)

    # 2. Multi-overlap Coverage Analysis
    ref_overlap_counts = (overlap > 0).sum(axis=1)
    ref_with_overlap = int((ref_overlap_counts > 0).sum())
    pred_overlap_counts = (overlap > 0).sum(axis=0)
    pred_with_overlap = int((pred_overlap_counts > 0).sum())
    multi_ref_predictions = int((pred_overlap_counts > 1).sum())

    # Coverage thresholds
    cov_50 = 0
    cov_80 = 0
    cov_100 = 0
    for i, row in reference.iterrows():
        r_dur = (row["end_time"] - row["start_time"]).total_seconds() / 3600.0 + 1.0
        max_ov = overlap[i, :].max() if overlap.shape[1] else 0
        if max_ov / r_dur >= 0.5:
            cov_50 += 1
        if max_ov / r_dur >= 0.8:
            cov_80 += 1
        if max_ov / r_dur >= 0.999:
            cov_100 += 1

    # False detection min Dst breakdown
    false_preds_df = matches_df[matches_df["status"] == "false_detection"]
    fa_min_dsts = false_preds_df["reference_min_dst"].dropna().values
    fa_sub_30 = int((fa_min_dsts <= -30.0).sum())
    fa_near_threshold = int(((fa_min_dsts > -30.0) & (fa_min_dsts <= -25.0)).sum())
    fa_above_25 = int((fa_min_dsts > -25.0).sum())

    # 3. Breakdown by Year
    yearly_records = []
    for yr in range(coverage_start.year, coverage_end.year + 1):
        yr_refs = reference[reference["start_time"].dt.year == yr]
        yr_preds = predicted[predicted["start_time"].dt.year == yr]
        yr_matched = matches_df[(matches_df["status"] == "matched") & (matches_df["reference_start"].dt.year == yr)]

        n_ref = len(yr_refs)
        n_pred = len(yr_preds)
        n_tp = len(yr_matched)
        n_missed = n_ref - n_tp
        p = n_tp / n_pred if n_pred else 0.0
        r = n_tp / n_ref if n_ref else 0.0
        f = (2 * p * r / (p + r)) if (p + r) else 0.0

        # Coverage rate in this year
        yr_ref_indices = yr_refs.index.values
        if len(yr_ref_indices) and overlap.shape[1]:
            yr_cov = int((overlap[yr_ref_indices, :].sum(axis=1) > 0).sum()) / n_ref
        else:
            yr_cov = 0.0

        yearly_records.append({
            "year": yr,
            "reference_storms": n_ref,
            "predicted_events": n_pred,
            "matched_1to1": n_tp,
            "missed_1to1": n_missed,
            "precision_1to1": round(p, 4),
            "recall_1to1": round(r, 4),
            "f1_1to1": round(f, 4),
            "reference_coverage_rate": round(yr_cov, 4),
        })

    yearly_df = pd.DataFrame(yearly_records)
    yearly_df.to_csv(metrics_dir / "event_metrics_by_year.csv", index=False)

    # 4. Breakdown by Intensity Tier
    tier_records = []
    for tier in ["Weak", "Moderate", "Intense", "Severe", "Extreme"]:
        tier_refs = reference[reference["intensity_tier"] == tier]
        n_ref = len(tier_refs)
        if n_ref == 0:
            continue
        tier_indices = tier_refs.index.values
        tier_matched = sum(1 for i, _, _ in matched_pairs if i in set(tier_indices))
        tier_cov = int((overlap[tier_indices, :].sum(axis=1) > 0).sum())

        tier_records.append({
            "intensity_tier": tier,
            "reference_storms": n_ref,
            "matched_1to1": tier_matched,
            "missed_1to1": n_ref - tier_matched,
            "recall_1to1": round(tier_matched / n_ref, 4),
            "storms_with_overlap": tier_cov,
            "coverage_rate": round(tier_cov / n_ref, 4),
        })

    tier_df = pd.DataFrame(tier_records)
    tier_df.to_csv(metrics_dir / "event_metrics_by_tier.csv", index=False)

    # 5. Gap sensitivity analysis
    sensitivity_df = run_merge_gap_sensitivity(test_times, predictions, dst, reference)
    sensitivity_df.to_csv(metrics_dir / "merge_gap_sensitivity.csv", index=False)

    # Master summary JSON
    summary = {
        "evaluation": "one-to-one event matching by temporal overlap",
        "test_start": str(coverage_start),
        "test_end": str(coverage_end),
        "hourly_probability_threshold": 0.5,
        "event_merge_gap_hours": merge_gap,
        "reference_events": int(len(reference)),
        "predicted_events": int(len(predicted)),
        "detected_storms": int(tp),
        "missed_storms": int(missed),
        "false_detections": int(false_detections),
        "event_precision": float(precision),
        "event_recall": float(recall),
        "event_f1": float(f1),
        "hourly_classification": hourly_metrics,
        "timing_and_duration": {
            "onset_delay_hours": {
                "mean": round(float(np.mean(onset_delays)), 2) if onset_delays else None,
                "median": round(float(np.median(onset_delays)), 2) if onset_delays else None,
                "std": round(float(np.std(onset_delays)), 2) if onset_delays else None,
            },
            "duration_error_hours": {
                "mean": round(float(np.mean(duration_errors)), 2) if duration_errors else None,
                "median": round(float(np.median(duration_errors)), 2) if duration_errors else None,
                "std": round(float(np.std(duration_errors)), 2) if duration_errors else None,
            },
            "overlap_hours": {
                "mean": round(float(np.mean(overlap_durations)), 2) if overlap_durations else None,
                "median": round(float(np.median(overlap_durations)), 2) if overlap_durations else None,
                "min": int(np.min(overlap_durations)) if overlap_durations else 0,
                "max": int(np.max(overlap_durations)) if overlap_durations else 0,
            },
        },
        "coverage_analysis": {
            "reference_storms_with_overlap": ref_with_overlap,
            "reference_storms_without_overlap": int(len(reference) - ref_with_overlap),
            "reference_detection_rate": round(ref_with_overlap / len(reference), 4) if len(reference) else 0.0,
            "reference_storms_coverage_ge_50pct": cov_50,
            "reference_storms_coverage_ge_80pct": cov_80,
            "reference_storms_coverage_100pct": cov_100,
            "predictions_overlapping_multiple_reference_storms": multi_ref_predictions,
            "predicted_events_with_overlap": pred_with_overlap,
            "predicted_events_without_overlap": int(len(predicted) - pred_with_overlap),
        },
        "false_detection_analysis": {
            "total_false_detections": len(false_preds_df),
            "reached_catalog_threshold_le_minus30": fa_sub_30,
            "near_threshold_depression_minus25_to_minus30": fa_near_threshold,
            "weak_or_quiet_gt_minus25": fa_above_25,
            "median_duration_hours": float(
                (pd.to_datetime(false_preds_df["predicted_end"]) - pd.to_datetime(false_preds_df["predicted_start"])).dt.total_seconds().median() / 3600.0 + 1.0
            ) if len(false_preds_df) else 0.0,
        },
    }

    with (metrics_dir / "event_evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot the Dst series year by year for readable inspection.
    plot_data = dst.loc[coverage_start:coverage_end]

    for year in range(coverage_start.year, coverage_end.year + 1):
        yearly = plot_data[plot_data.index.year == year]
        if yearly.empty:
            continue

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(yearly.index, yearly.to_numpy(), linewidth=0.7, label="Dst (nT)")

        first_ref_label = True
        for row in reference.itertuples(index=False):
            left = max(row.start_time, yearly.index.min())
            right = min(row.end_time, yearly.index.max())
            if left <= right:
                ax.axvspan(
                    left, right, alpha=0.18,
                    label="Catalogue storm" if first_ref_label else None,
                )
                first_ref_label = False

        first_pred_label = True
        for row in predicted.itertuples(index=False):
            left = max(row.start_time, yearly.index.min())
            right = min(row.end_time, yearly.index.max())
            if left <= right:
                ax.axvspan(
                    left, right, alpha=0.28,
                    label="CNN-predicted event" if first_pred_label else None,
                )
                first_pred_label = False

        ax.axhline(-30, linestyle="--", linewidth=0.8, label="Catalogue trigger (−30 nT)")
        ax.set_title(f"CNN Event Evaluation — {year}")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Dst (nT)")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figures_dir / f"cnn_event_evaluation_{year}.png", dpi=160)
        plt.close(fig)

    print("\nEVENT-LEVEL EVALUATION")
    print("=" * 50)
    print(f"Test coverage:      {coverage_start} to {coverage_end}")
    print(f"Reference storms:   {len(reference):,}")
    print(f"CNN predicted:      {len(predicted):,}")
    print(f"Detected storms:    {tp:,}")
    print(f"Missed storms:      {missed:,}")
    print(f"False detections:   {false_detections:,}")
    print(f"Event precision:    {precision:.4f} ({precision*100:.2f}%)")
    print(f"Event recall:       {recall:.4f} ({recall*100:.2f}%)")
    print(f"Event F1-score:     {f1:.4f} ({f1*100:.2f}%)")
    print("\nPHYSICAL RECOGNITION (COVERAGE)")
    print(f"Reference storms with overlap:  {ref_with_overlap}/{len(reference)} ({ref_with_overlap/len(reference)*100:.2f}%)")
    print(f"Reference storms >= 50% coverage: {cov_50}/{len(reference)} ({cov_50/len(reference)*100:.2f}%)")
    print(f"Predictions spanning multi-storms: {multi_ref_predictions}")
    print(f"\nSaved metrics to: {metrics_dir}")
    print(f"Saved yearly plots to: {figures_dir}")


if __name__ == "__main__":
    main()
