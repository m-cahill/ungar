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
        importance: A (4, 14) float array where values indicate relevance/attribution.
        label: A string label describing what this overlay represents (e.g., 'saliency').
        meta: Additional metadata dictionary.
    """

    importance: np.ndarray  # shape (4, 14)
    label: str
    meta: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate shape of importance array."""
        expected_shape = (SUIT_COUNT, RANK_COUNT)
        if self.importance.shape != expected_shape:
            raise ValueError(
                f"importance must have shape {expected_shape}, got {self.importance.shape!r}"
            )


def zero_overlay(label: str, meta: Mapping[str, Any] | None = None) -> CardOverlay:
    """Create a CardOverlay initialized with zeros.

    Args:
        label: Label for the overlay.
        meta: Optional metadata dict.

    Returns:
        A new CardOverlay instance with zero importance.
    """
    return CardOverlay(
        importance=np.zeros((SUIT_COUNT, RANK_COUNT), dtype=float),
        label=label,
        meta=meta or {},
    )


class CardOverlayDict(TypedDict):
    """Dictionary representation of a CardOverlay for serialization."""

    importance: list[list[float]]
    label: str
    meta: dict[str, Any]


def overlay_to_dict(overlay: CardOverlay) -> CardOverlayDict:
    """Convert CardOverlay to a JSON-serializable dictionary.

    Args:
        overlay: The overlay to serialize.

    Returns:
        A dictionary with 'importance' as a nested list.
    """
    # numpy tolist() returns nested lists for 2D arrays
    importance_list = overlay.importance.tolist()
    return {
        "importance": cast(list[list[float]], importance_list),
        "label": overlay.label,
        "meta": dict(overlay.meta),
    }


def overlay_from_dict(data: CardOverlayDict) -> CardOverlay:
    """Reconstruct CardOverlay from a dictionary.

    Args:
        data: Dictionary created by overlay_to_dict.

    Returns:
        Reconstructed CardOverlay instance.
    """
    arr = np.array(data["importance"], dtype=float)
    return CardOverlay(importance=arr, label=data["label"], meta=data["meta"])
