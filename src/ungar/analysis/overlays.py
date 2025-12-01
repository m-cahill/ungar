"""Overlay loading and aggregation tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
from ungar.enums import RANK_COUNT, SUIT_COUNT
from ungar.xai import CardOverlay, overlay_from_dict, overlay_to_dict


def load_overlays(run_dir: str | Path) -> List[CardOverlay]:
    """Load all overlays from a run directory."""
    run_dir = Path(run_dir)
    overlay_dir = run_dir / "overlays"

    if not overlay_dir.exists():
        return []

    overlays = []
    # Load all JSON files, sorting by name might help order if timestamped
    # But usually we just want the aggregate set.
    for f in sorted(overlay_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data_list = json.load(fp)
            # Support both single list of overlays per file or multiple files
            # M19: We might have single dict now too.
            if isinstance(data_list, list):
                for item in data_list:
                    overlays.append(overlay_from_dict(item))
            elif isinstance(data_list, dict):
                # Single overlay
                overlays.append(overlay_from_dict(data_list))
            else:
                # Should be list based on OverlayExporter
                pass

    return overlays


def compute_mean_overlay(overlays: List[CardOverlay], label: str = "mean") -> CardOverlay:
    """Compute the mean of a list of overlays."""
    if not overlays:
        raise ValueError("Cannot compute mean of empty overlay list")

    # Check consistency (naive check)
    first = overlays[0]
    for o in overlays[1:]:
        if o.importance.shape != first.importance.shape:
            raise ValueError("Overlay shapes mismatch")

    stack = np.stack([o.importance for o in overlays])
    mean_imp = np.mean(stack, axis=0)

    return CardOverlay(
        run_id=first.run_id,  # Assume same run, or use "aggregated"
        label=label,
        agg="mean",
        step=-1,  # sentinel
        importance=mean_imp,
        meta={"aggregated": True, "count": len(overlays)},
    )


def compute_max_overlay(overlays: List[CardOverlay], label: str = "max") -> CardOverlay:
    """Compute the max of a list of overlays."""
    if not overlays:
        raise ValueError("Cannot compute max of empty overlay list")

    stack = np.stack([o.importance for o in overlays])
    max_imp = np.max(stack, axis=0)

    return CardOverlay(
        run_id=overlays[0].run_id,
        label=label,
        agg="max",
        step=-1,
        importance=max_imp,
        meta={"aggregated": True, "count": len(overlays)},
    )


def aggregate_overlays(overlays: List[CardOverlay], method: str = "mean") -> np.ndarray:
    """Compute aggregated importance map."""
    if not overlays:
        return np.zeros((SUIT_COUNT, RANK_COUNT))

    stack = np.stack([o.importance for o in overlays])

    if method == "mean":
        return np.mean(stack, axis=0)
    elif method == "max":
        return np.max(stack, axis=0)
    elif method == "std":
        return np.std(stack, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def save_aggregation(
    importance: np.ndarray, out_path: str | Path, label: str = "aggregated"
) -> None:
    """Save aggregated map as a synthetic overlay JSON."""
    # Create a dummy overlay to reuse serialization logic
    overlay = CardOverlay(
        run_id="aggregated",
        label=label,
        agg="mean",
        step=0,
        importance=importance,
        meta={"aggregated": True},
    )

    data = overlay_to_dict(overlay)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
