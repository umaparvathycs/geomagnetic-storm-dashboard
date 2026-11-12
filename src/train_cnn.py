
"""Train the 1D CNN for hourly geomagnetic-storm recognition.

Usage:
    python -m src.train_cnn
"""

from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from .common import load_config, resolve
from .cnn_dataset import build_dataset, class_distribution


def seed_everything(seed: int):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(argv=None):
    import argparse
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from .cnn_model import StormCNN

    # ------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config",
        default=None,
        help="Path to config.yaml",
    )

    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    c = cfg["cnn"]

    seed_everything(int(c["seed"]))

    # ------------------------------------------------------------
    # Load final Dst data: January 1957 through December 2020
    # ------------------------------------------------------------

    dst_data = pd.read_csv(
        resolve(cfg, "clean_series"),
        parse_dates=["timestamp_utc"],
    )

    dst_data = dst_data[
        (dst_data["product"] == "dst_final")
        & (dst_data["timestamp_utc"] >= "1957-01-01")
        & (dst_data["timestamp_utc"] < "2021-01-01")
    ].copy()

    if dst_data.empty:
        raise ValueError(
            "No dst_final data found for January 1957–December 2020."
        )

    dst_data = dst_data.sort_values("timestamp_utc")

    series = dst_data.set_index("timestamp_utc")["dst_nt"]

    print("\nINPUT DATA")
    print("=" * 60)
    print("Product: dst_final")
    print(f"Start: {series.index.min()}")
    print(f"End:   {series.index.max()}")
    print(f"Hourly rows: {len(series):,}")
    print("=" * 60)

    # ------------------------------------------------------------
    # Load storm catalogue
    # ------------------------------------------------------------

    catalogue = pd.read_csv(
        resolve(cfg, "catalogue"),
        parse_dates=[
            "start_time",
            "peak_time",
            "end_time",
        ],
    )

    # ------------------------------------------------------------
    # Build CNN dataset
    # ------------------------------------------------------------

    data, _, _ = build_dataset(
        series,
        catalogue,
        cfg,
    )

    distribution = class_distribution(data)

    print("\nCNN DATASET")
    print("=" * 60)
    print(distribution.to_string(index=False))
    print("=" * 60)

    # ------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------

    metrics_dir = resolve(cfg, "metrics_dir")
    figures_dir = resolve(cfg, "figures_dir")
    models_dir = resolve(cfg, "models_dir")

    for directory in (
        metrics_dir,
        figures_dir,
        models_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    distribution.to_csv(
        metrics_dir / "class_distribution.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA GPU not available; training on CPU.")

    # ------------------------------------------------------------
    # Training / validation data
    # ------------------------------------------------------------

    train = data["train"]
    val = data["val"]

    X_train = torch.from_numpy(train["X"])
    y_train = torch.from_numpy(train["y"])

    X_val = torch.from_numpy(val["X"]).to(device)
    y_val = torch.from_numpy(val["y"]).to(device)

    # ------------------------------------------------------------
    # Class weighting
    # ------------------------------------------------------------

    positive = float(train["y"].sum())
    negative = float(
        len(train["y"]) - positive
    )

    pos_weight_value = (
        negative / max(positive, 1.0)
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    print(
        f"\nTraining positive windows: {int(positive):,}"
    )

    print(
        f"Training negative windows: {int(negative):,}"
    )

    print(
        f"Positive class weight: {pos_weight_value:.4f}"
    )

    # ------------------------------------------------------------
    # Model
    # ------------------------------------------------------------

    model = StormCNN(
        window=int(c["window_hours"])
    ).to(device)

    print("\nMODEL")
    print("=" * 60)
    print(model)
    print("=" * 60)

    # ------------------------------------------------------------
    # Loss and optimizer
    # ------------------------------------------------------------

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(c["lr"]),
    )

    # ------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        int(c["seed"])
    )

    loader = DataLoader(
        TensorDataset(
            X_train,
            y_train,
        ),
        batch_size=int(c["batch_size"]),
        shuffle=True,
        generator=generator,
    )

    # ------------------------------------------------------------
    # Prediction helper
    # ------------------------------------------------------------

    def predict(X):
        model.eval()

        predictions = []

        with torch.no_grad():
            for start in range(
                0,
                len(X),
                8192,
            ):
                batch = X[
                    start:start + 8192
                ]

                probabilities = torch.sigmoid(
                    model(batch)
                )

                predictions.append(
                    probabilities.cpu()
                )

        return torch.cat(
            predictions
        ).numpy()

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    history = []

    best_pr_auc = -np.inf
    bad_epochs = 0

    epochs = int(c["epochs"])
    patience = int(c["patience"])

    print("\nTRAINING")
    print("=" * 60)

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = loss_fn(
                logits,
                y_batch,
            )

            loss.backward()
            optimizer.step()

            batch_size = len(X_batch)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        train_loss = (
            total_loss / total_samples
        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        model.eval()

        with torch.no_grad():
            validation_loss = loss_fn(
                model(X_val),
                y_val,
            ).item()

        validation_probabilities = predict(
            X_val
        )

        validation_labels = val["y"]

        validation_predictions = (
            validation_probabilities >= 0.5
        )

        pr_auc = float(
            average_precision_score(
                validation_labels,
                validation_probabilities,
            )
        )

        f1 = float(
            f1_score(
                validation_labels,
                validation_predictions,
                zero_division=0,
            )
        )

        precision = float(
            precision_score(
                validation_labels,
                validation_predictions,
                zero_division=0,
            )
        )

        recall = float(
            recall_score(
                validation_labels,
                validation_predictions,
                zero_division=0,
            )
        )

        record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(validation_loss),
            "val_pr_auc": pr_auc,
            "val_f1@0.5": f1,
            "val_precision@0.5": precision,
            "val_recall@0.5": recall,
        }

        history.append(record)

        print(record)

        # --------------------------------------------------------
        # Save best model according to PR-AUC
        # --------------------------------------------------------

        if pr_auc > best_pr_auc + 1e-5:
            best_pr_auc = pr_auc
            bad_epochs = 0

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "window": int(
                        c["window_hours"]
                    ),
                    "scale_nt": float(
                        c["scale_nt"]
                    ),
                    "label_position": c[
                        "label_position"
                    ],
                    "seed": int(c["seed"]),
                    "epoch": epoch,
                    "val_pr_auc": pr_auc,
                },
                models_dir / "storm_cnn.pt",
            )

            print(
                f"  -> saved best model "
                f"(validation PR-AUC={pr_auc:.4f})"
            )

        else:
            bad_epochs += 1

            if bad_epochs >= patience:
                print(
                    f"\nEarly stopping after "
                    f"{epoch} epochs."
                )
                break

    # ------------------------------------------------------------
    # Save training history
    # ------------------------------------------------------------

    with open(
        metrics_dir / "training_history.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
        )

    # ------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------

    history_df = pd.DataFrame(history)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4),
    )

    axes[0].plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="train",
    )

    axes[0].plot(
        history_df["epoch"],
        history_df["val_loss"],
        label="validation",
    )

    axes[0].set_title("CNN Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Class-weighted BCE")
    axes[0].legend()
    axes[0].grid(alpha=0.4)

    axes[1].plot(
        history_df["epoch"],
        history_df["val_pr_auc"],
        label="PR-AUC",
    )

    axes[1].plot(
        history_df["epoch"],
        history_df["val_f1@0.5"],
        label="F1 @ 0.5",
    )

    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(alpha=0.4)

    fig.tight_layout()

    fig.savefig(
        figures_dir / "training_curves.png",
        dpi=150,
    )

    plt.close(fig)

    print("\nTRAINING COMPLETE")
    print("=" * 60)
    print(
        f"Best validation PR-AUC: "
        f"{best_pr_auc:.4f}"
    )
    print(
        f"Model saved to: "
        f"{models_dir / 'storm_cnn.pt'}"
    )
    print(
        f"History saved to: "
        f"{metrics_dir / 'training_history.json'}"
    )
    print(
        f"Curves saved to: "
        f"{figures_dir / 'training_curves.png'}"
    )


if __name__ == "__main__":
    main()
