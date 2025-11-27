"""Tests for High Card Duel XAI overlays."""

import numpy as np
from ungar.cards import Card
from ungar.enums import Rank, Suit
from ungar.tensor import CardTensor
from ungar_bridge.xai_overlays import make_high_card_overlay


def test_high_card_overlay_heuristic() -> None:
    """Test that the heuristic overlay highlights 'my_hand' cards."""
    # Setup a state tensor with known cards
    my_card = Card(Suit.SPADES, Rank.ACE)  # (0, 0)
    op_card = Card(Suit.HEARTS, Rank.KING)  # (1, 12)

    # Create tensor
    tensor = CardTensor.from_plane_card_map(
        {
            "my_hand": [my_card],
            "opponent_hand": [op_card],
            "unseen": [],  # Others don't matter for this test
        }
    )

    # Generate overlay
    overlay = make_high_card_overlay(tensor)

    # Verify metadata
    assert overlay.label == "high_card_importance"
    assert overlay.meta["game"] == "high_card_duel"

    # Verify importance
    # Expect 1.0 at my_card position (0, 0)
    assert overlay.importance[0, 0] == 1.0

    # Expect 0.0 at op_card position (1, 12)
    assert overlay.importance[1, 12] == 0.0

    # Expect only 1 non-zero element (since I have 1 card)
    assert np.count_nonzero(overlay.importance) == 1


def test_high_card_overlay_empty_hand() -> None:
    """Test overlay with no cards in hand (e.g. after play or if missing)."""
    tensor = CardTensor.from_plane_card_map(
        {
            "my_hand": [],
            "opponent_hand": [],
        }
    )

    overlay = make_high_card_overlay(tensor)
    assert np.all(overlay.importance == 0.0)
