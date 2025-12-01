"""Overlay methods for XAI."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ungar.enums import RANK_COUNT, SUIT_COUNT
from ungar.xai import CardOverlay, zero_overlay


@runtime_checkable
class OverlayMethod(Protocol):
    """Protocol for XAI overlay generation methods."""

    label: str

    def compute(
        self,
        obs: np.ndarray,
        action: int,
        *,
        step: int,
        run_id: str,
        meta: dict | None = None,
    ) -> CardOverlay:
        """Compute an overlay for a given observation and action."""
        ...


class RandomOverlayMethod:
    """Generates random importance values."""

    label = "random"

    def compute(
        self,
        obs: np.ndarray,
        action: int,
        *,
        step: int,
        run_id: str,
        meta: dict | None = None,
    ) -> CardOverlay:
        """Generate a random 4x14 overlay."""
        importance = np.random.rand(SUIT_COUNT, RANK_COUNT)
        # Normalize to sum to 1
        importance = importance / importance.sum()

        return CardOverlay(
            run_id=run_id,
            label=self.label,
            agg="none",
            step=step,
            importance=importance,
            meta=meta or {},
        )


class HandHighlightMethod:
    """Highlights cards in the agent's hand (heuristic)."""

    label = "heuristic"

    def compute(
        self,
        obs: np.ndarray,
        action: int,
        *,
        step: int,
        run_id: str,
        meta: dict | None = None,
    ) -> CardOverlay:
        """Generate an overlay highlighting held cards."""
        # Assumption: obs is flat, need to reshape or know structure.
        # UNGAR Tensor structure: (4, 14, N).
        # Plane 0 is typically "my_hand".
        # We need to know the shape to reshape properly if obs is flattened.
        # But wait, obs passed here might be flat from training loop?
        # Yes, training loop passes flattened obs.
        # We need the tensor shape.
        # Let's assume standard 4x14xN.

        # This method is tightly coupled to tensor layout.
        # For now, we try to infer or assume N based on size.

        size = obs.size
        # 4 * 14 = 56
        tensor_plane_size = SUIT_COUNT * RANK_COUNT

        if size % tensor_plane_size != 0:
            # Fallback if unknown shape
            return zero_overlay(self.label, meta)

        n_planes = size // tensor_plane_size
        # Reshape to (4, 14, n)
        # NOTE: Check flatten order. Usually 'C' (row-major).
        tensor = obs.reshape((SUIT_COUNT, RANK_COUNT, n_planes))

        # Plane 0 = My Hand (convention in high_card_duel and others)
        hand_plane = tensor[:, :, 0]

        # Normalize
        count = np.sum(hand_plane)
        if count > 0:
            importance = hand_plane.astype(float) / count
        else:
            importance = np.zeros((SUIT_COUNT, RANK_COUNT), dtype=float)

        return CardOverlay(
            run_id=run_id,
            label=self.label,
            agg="none",
            step=step,
            importance=importance,
            meta=meta or {},
        )
