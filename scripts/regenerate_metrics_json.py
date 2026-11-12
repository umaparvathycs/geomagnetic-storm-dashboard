"""
Regenerate outputs/metrics/metrics.json from the verified checkpoint.

Ensures the saved metrics file matches the actual test set (43,801 windows,
2016-2020) and uses both the fixed threshold=0.5 and the validation-selected
threshold. Replaces the stale metrics.json that was from an earlier run with
a different test set size (92,713 windows).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import platform
platform._wmi = None

from src.common import load_config, resolve
from src.cnn_dataset import build_dataset
from src.cnn_model import StormCNN


def compute_metrics(y_true, probabilities, threshold):
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix[[TN,FP],[FN,TP]]": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n_samples": int(len(y_true)),
    }


def main():
    cfg = load_config()

    # Load data with same filters as evaluate_cnn.py
    dst = pd.read_csv(resolve(cfg, "clean_series"), parse_dates=["timestamp_utc"])
    dst = dst[
        (dst["product"] == "dst_final")
        & (dst["timestamp_utc"] >= "1957-01-01")
        & (dst["timestamp_utc"] < "2021-01-01")
    ].copy()
    series = dst.sort_values("timestamp_utc").set_index("timestamp_utc")["dst_nt"]

    catalogue = pd.read_csv(
        resolve(cfg, "catalogue"),
        parse_dates=["start_time", "peak_time", "end_time"],
    )

    data, _, _ = build_dataset(series, catalogue, cfg)

    device = torch.device("cpu")
    checkpoint = torch.load(
        resolve(cfg, "models_dir") / "storm_cnn.pt",
        map_location=device,
        weights_only=False,
    )
    model = StormCNN(window=int(checkpoint["window"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # --- Evaluate all three splits ---
    results = {}
    for split_name in ["test", "val", "train"]:
        split_key = split_name if split_name != "val" else "val"
        split_data = data.get(split_key) or data.get(split_name)
        if split_data is None:
            continue

        X = torch.from_numpy(split_data["X"])
        y_true = split_data["y"].astype(int)

        probabilities = []
        with torch.no_grad():
            for start in range(0, len(X), 8192):
                batch = X[start : start + 8192].to(device)
                probs = torch.sigmoid(model(batch))
                probabilities.extend(probs.cpu().numpy().ravel())
        probabilities = np.asarray(probabilities)

        # At threshold 0.5 (standard)
        results[f"{split_name}_at_0.5"] = compute_metrics(y_true, probabilities, 0.5)

        # Find best-F1 threshold on validation set
        if split_name == "val":
            best_thr, best_f1 = 0.5, 0.0
            for thr in np.arange(0.3, 0.8, 0.005):
                f1_val = f1_score(y_true, (probabilities >= thr).astype(int), zero_division=0)
                if f1_val > best_f1:
                    best_thr, best_f1 = thr, f1_val
            results["selected_threshold_from_validation"] = float(best_thr)
            results["validation_at_selected_threshold"] = compute_metrics(
                y_true, probabilities, best_thr
            )

    # Apply validation-selected threshold to test set
    if "selected_threshold_from_validation" in results:
        sel_thr = results["selected_threshold_from_validation"]
        test_data = data["test"]
        X_test = torch.from_numpy(test_data["X"])
        y_test = test_data["y"].astype(int)
        test_probs = []
        with torch.no_grad():
            for start in range(0, len(X_test), 8192):
                batch = X_test[start : start + 8192].to(device)
                probs = torch.sigmoid(model(batch))
                test_probs.extend(probs.cpu().numpy().ravel())
        test_probs = np.asarray(test_probs)
        results["test_at_selected_threshold"] = compute_metrics(y_test, test_probs, sel_thr)

    # Save
    out_path = resolve(cfg, "metrics_dir") / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Regenerated {out_path}")
    print(f"  Test samples (at 0.5): {results['test_at_0.5']['n_samples']:,}")
    print(f"  Test accuracy (at 0.5): {results['test_at_0.5']['accuracy']:.4f}")
    if "test_at_selected_threshold" in results:
        sel = results["test_at_selected_threshold"]
        print(f"  Test accuracy (at {results['selected_threshold_from_validation']:.4f}): {sel['accuracy']:.4f}")


if __name__ == "__main__":
    main()
