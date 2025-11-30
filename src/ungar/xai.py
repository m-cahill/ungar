"""Explainable AI (XAI) data structures for UNGAR.

This module defines standard formats for card overlays, allowing attribution
and saliency maps to be represented consistently across different games.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypedDict, cast

import numpy as np

from .enums import RANK_COUNT, SUIT_COUNT


@dataclass
class CardOverlay:
    """Represents an importance overlay for a 4x14 card grid.

    Attributes:
        run_id: The ID of the training run.
        label: A string label describing what this overlay represents (e.g., 'saliency').
        agg: Aggregation method used (e.g., 'mean', 'max', 'none').
        step: The step or episode index this overlay corresponds to.
        importance: A (4, 14) float array where values indicate relevance/attribution.
        meta: Additional metadata dictionary.
    """

    run_id: str
    label: str
    agg: str
    step: int
    importance: np.ndarray  # shape (4, 14)
    meta: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate shape of importance array."""
        expected_shape = (SUIT_COUNT, RANK_COUNT)
        if self.importance.shape != expected_shape:
            raise ValueError(
                f"importance must have shape {expected_shape}, got {self.importance.shape!r}"
            )

    def to_payload(self) -> dict[str, Any]:
        """Convert to a schema-compliant dictionary for serialization.

        Returns:
            A dict matching the M17 overlay schema.
        """
        # numpy tolist() returns nested lists for 2D arrays
        importance_list = self.importance.tolist()
        return {
            "run_id": self.run_id,
            "label": self.label,
            "agg": self.agg,
            "step": self.step,
            "importance": importance_list,
            "meta": dict(self.meta),
        }


def zero_overlay(label: str, meta: Mapping[str, Any] | None = None) -> CardOverlay:
    """Create a CardOverlay initialized with zeros.

    Args:
        label: Label for the overlay.
        meta: Optional metadata dict.

    Returns:
        A new CardOverlay instance with zero importance.
    """
    return CardOverlay(
        run_id="placeholder",
        label=label,
        agg="none",
        step=0,
        importance=np.zeros((SUIT_COUNT, RANK_COUNT), dtype=float),
        meta=meta or {},
    )


class CardOverlayDict(TypedDict):
    """Dictionary representation of a CardOverlay for serialization."""

    run_id: str
    label: str
    agg: str
    step: int
    importance: list[list[float]]
    meta: dict[str, Any]


def overlay_to_dict(overlay: CardOverlay) -> CardOverlayDict:
    """Convert CardOverlay to a JSON-serializable dictionary.

    Args:
        overlay: The overlay to serialize.

    Returns:
        A dictionary with 'importance' as a nested list.
    """
    return cast(CardOverlayDict, overlay.to_payload())


def overlay_from_dict(data: CardOverlayDict) -> CardOverlay:
    """Reconstruct CardOverlay from a dictionary.

    Args:
        data: Dictionary created by overlay_to_dict.

    Returns:
        Reconstructed CardOverlay instance.
    """
    arr = np.array(data["importance"], dtype=float)
    return CardOverlay(
        run_id=data.get("run_id", "unknown"),  # Backwards compat if needed, though M17 strictly requires it
        label=data["label"],
        agg=data.get("agg", "none"),
        step=data.get("step", 0),
        importance=arr,
        meta=data["meta"],
    )
