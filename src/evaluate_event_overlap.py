
"""Supplementary overlap-based evaluation of CNN storm events."""

import json
import pandas as pd
from pathlib import Path

metrics = Path("outputs/metrics")

predicted = pd.read_csv(
    metrics / "cnn_predicted_events.csv",
    parse_dates=["start_time", "end_time"],
)

reference = pd.read_csv(
    metrics / "reference_test_events.csv",
    parse_dates=["start_time", "end_time"],
)

# Make timestamps consistently UTC.
for frame, columns in [
    (predicted, ["start_time", "end_time"]),
    (reference, ["start_time", "end_time"]),
]:
    for col in columns:
        frame[col] = pd.to_datetime(frame[col], utc=True)

# Record every reference/predicted event pair that overlaps.
pairs = []

for i, ref in reference.iterrows():
    for j, pred in predicted.iterrows():
        start = max(ref["start_time"], pred["start_time"])
        end = min(ref["end_time"], pred["end_time"])

        if start <= end:
            overlap_hours = int(
                (end - start).total_seconds() // 3600
            ) + 1

            pairs.append({
                "reference_event_id": ref["reference_event_id"],
                "reference_start": ref["start_time"],
                "reference_end": ref["end_time"],
                "predicted_event_id": pred["predicted_event_id"],
                "predicted_start": pred["start_time"],
                "predicted_end": pred["end_time"],
                "overlap_hours": overlap_hours,
            })

pairs_df = pd.DataFrame(pairs, columns=[
    "reference_event_id", "reference_start", "reference_end",
    "predicted_event_id", "predicted_start", "predicted_end",
    "overlap_hours",
])

pairs_df.to_csv(
    metrics / "cnn_event_overlap_pairs.csv", index=False
)

# A reference storm is detected if any predicted event overlaps it.
overlapped_refs = set(pairs_df["reference_event_id"]) if len(pairs_df) else set()
overlapped_preds = set(pairs_df["predicted_event_id"]) if len(pairs_df) else set()

missed = reference[
    ~reference["reference_event_id"].isin(overlapped_refs)
]

false_detections = predicted[
    ~predicted["predicted_event_id"].isin(overlapped_preds)
]

# Count predicted events that overlap more than one reference storm.
ref_counts = pairs_df.groupby("predicted_event_id")[
    "reference_event_id"
].nunique() if len(pairs_df) else pd.Series(dtype=int)

merged_predictions = int((ref_counts > 1).sum())

summary = {
    "method": "any temporal overlap; supplementary, many-to-many",
    "minimum_overlap_hours": 1,
    "reference_storms": int(len(reference)),
    "predicted_events": int(len(predicted)),
    "reference_storms_with_overlap": int(len(overlapped_refs)),
    "reference_storms_without_overlap": int(len(missed)),
    "predicted_events_with_overlap": int(len(overlapped_preds)),
    "predicted_events_without_overlap": int(len(false_detections)),
    "overlapping_event_pairs": int(len(pairs_df)),
    "predictions_overlapping_multiple_reference_storms": merged_predictions,
    "reference_detection_rate": (
        len(overlapped_refs) / len(reference) if len(reference) else 0.0
    ),
    "prediction_overlap_rate": (
        len(overlapped_preds) / len(predicted) if len(predicted) else 0.0
    ),
}

with (metrics / "event_overlap_evaluation_summary.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\nSUPPLEMENTARY OVERLAP EVALUATION")
print("=" * 45)
for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print(f"\nSaved results to: {metrics}")
