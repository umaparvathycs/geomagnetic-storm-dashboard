"""1D CNN for hourly Dst windows (PyTorch). Input (B, W, 1) -> logit (B,).

Design rationale: the input is a short univariate series (W=48), so a small network is enough and less likely to
memorise. Two Conv1d blocks (k=5) learn local shapes (sudden drop, ramp, recovery curvature); a third (k=3)
refines them; GLOBAL average + max pooling make the head independent of where in the window the pattern sits and
add a "how deep did it go" feature; Dense -> Dropout -> 1 logit (sigmoid applied in the loss / at inference)."""
from __future__ import annotations
import torch
from torch import nn


class StormCNN(nn.Module):
    def __init__(self, window: int = 48, channels=(16, 32, 64), dropout: float = 0.3):
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, 5, padding=2), nn.BatchNorm1d(c1), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, 5, padding=2), nn.BatchNorm1d(c2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(c2, c3, 3, padding=1), nn.BatchNorm1d(c3), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(2 * c3, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        self.window = window

    def forward(self, x):                       # x: (B, W, 1)
        h = self.features(x.permute(0, 2, 1))   # (B, C, W/4)
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(h).squeeze(1)
