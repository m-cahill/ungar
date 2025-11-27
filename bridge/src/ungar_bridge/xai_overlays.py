"""XAI overlay generators for UNGAR bridge.

This module provides functions to generate explanation overlays for specific games,
starting with simple heuristics for High Card Duel.
"""

from __future__ import annotations

from ungar.enums import RANK_COUNT
from ungar.tensor import CardTensor
from ungar.xai import CardOverlay, zero_overlay


def make_high_card_overlay(state_tensor: CardTensor, label: str = "high_card_importance") -> CardOverlay:
    """Generate a heuristic importance overlay for High Card Duel.
    
    This simple overlay highlights the cards in 'my_hand' with importance 1.0,
    representing that the agent's decision is primarily based on their own card.
    
    Args:
        state_tensor: The current game state tensor.
        label: Label for the overlay.

    Returns:
        A CardOverlay with 'my_hand' cards highlighted.
    """
    overlay = zero_overlay(label=label, meta={"game": "high_card_duel"})
    importance = overlay.importance

    # Check if 'my_hand' plane exists to be safe
    if "my_hand" in state_tensor.spec.plane_names:
        my_hand_cards = state_tensor.cards_in_plane("my_hand")
        for card in my_hand_cards:
            idx = card.to_index()
            suit_index, rank_index = divmod(idx, RANK_COUNT)
            importance[suit_index, rank_index] = 1.0

    return overlay

