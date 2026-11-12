
"""Evaluate the saved CNN on the held-out test set."""

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .common import load_config, resolve
from .cnn_dataset import build_dataset
from .cnn_model import StormCNN



def main():
    cfg = load_config()
    c = cfg["cnn"]

    # Load final Dst data only, 1957–2020
    dst = pd.read_csv(
        resolve(cfg, "clean_series"),
        parse_dates=["timestamp_utc"],
    )

    dst = dst[
        (dst["product"] == "dst_final")
        & (dst["timestamp_utc"] >= "1957-01-01")
        & (dst["timestamp_utc"] < "2021-01-01")
    ].copy()

    series = (
        dst.sort_values("timestamp_utc")
        .set_index("timestamp_utc")["dst_nt"]
    )

    catalogue = pd.read_csv(
        resolve(cfg, "catalogue"),
        parse_dates=["start_time", "peak_time", "end_time"],
    )

    data, _, _ = build_dataset(series, catalogue, cfg)
    test = data["test"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        resolve(cfg, "models_dir") / "storm_cnn.pt",
        map_location=device,
        weights_only=False,
    )

    model = StormCNN(
        window=int(checkpoint["window"])
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    X = torch.from_numpy(test["X"])
    y_true = test["y"].astype(int)

    probabilities = []

    with torch.no_grad():
        for start in range(0, len(X), 8192):
            batch = X[start:start + 8192].to(device)
            probs = torch.sigmoid(model(batch))
            probabilities.extend(probs.cpu().numpy().ravel())

    probabilities = np.asarray(probabilities)
    y_pred = (probabilities >= 0.5).astype(int)

    print("\nTEST SET RESULTS")
    print("=" * 50)
    print(f"Test windows: {len(y_true):,}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"PR-AUC:    {average_precision_score(y_true, probabilities):.4f}")

    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Non-storm", "Storm"],
        zero_division=0,
    ))


if __name__ == "__main__":
    main()
