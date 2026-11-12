"""Shared helpers: config loading, paths, intensity classification."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    """Load project configuration."""

    config_path = Path(path) if path else ROOT / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ValueError("config.yaml did not load as a dictionary.")

    cfg["_root"] = str(config_path.resolve().parent)

    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a configured path relative to the project root."""

    if "paths" not in cfg:
        raise KeyError(
            "Missing 'paths' section in config.yaml."
        )

    if key not in cfg["paths"]:
        raise KeyError(
            f"Missing paths.{key} in config.yaml."
        )

    path = Path(cfg["paths"][key])

    if path.is_absolute():
        return path

    return Path(cfg["_root"]) / path


def classify_intensity(
    min_dst: float,
    tiers: dict | None = None
) -> str:
    """
    Classify storm intensity.

    Tier convention:
        (lower, upper]

    Therefore:

        -30 < Dst <= -30  -> boundary handled by trigger rule
        -50 < Dst <= -30   -> Weak
        -100 < Dst <= -50  -> Moderate
        -250 < Dst <= -100 -> Intense
        -350 < Dst <= -250 -> Severe
        Dst <= -350         -> Extreme

    Values above -30 are returned as "None".
    """

    if min_dst != min_dst:
        return "Unknown"

    if tiers is None:
        tiers = {
            "Weak": [-50, -30],
            "Moderate": [-100, -50],
            "Intense": [-250, -100],
            "Severe": [-350, -250],
            "Extreme": [None, -350],
        }

    for name, bounds in tiers.items():

        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(
                f"Invalid intensity tier '{name}': {bounds}. "
                "Expected [lower, upper]."
            )

        lo, hi = bounds

        if lo is None:
            if min_dst <= hi:
                return name

        elif min_dst > lo and min_dst <= hi:
            return name

    return "None"