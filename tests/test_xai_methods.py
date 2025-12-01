"""Unit tests for XAI overlay methods."""

import numpy as np
from ungar.enums import RANK_COUNT, SUIT_COUNT
from ungar.xai_methods import HandHighlightMethod, RandomOverlayMethod


def test_random_method() -> None:
    """Test RandomOverlayMethod."""
    method = RandomOverlayMethod()
    obs = np.zeros((SUIT_COUNT, RANK_COUNT, 3)).flatten()

    overlay = method.compute(obs, 0, step=1, run_id="test")

    assert overlay.label == "random"
    assert overlay.step == 1
    assert overlay.run_id == "test"
    assert overlay.importance.shape == (SUIT_COUNT, RANK_COUNT)
    # Check normalization
    assert np.isclose(overlay.importance.sum(), 1.0)


def test_heuristic_method() -> None:
    """Test HandHighlightMethod."""
    method = HandHighlightMethod()

    # Create fake tensor with 1 card in hand (plane 0)
    # 4x14x1
    tensor = np.zeros((SUIT_COUNT, RANK_COUNT, 1))
    tensor[0, 0, 0] = 1  # Ace of Spades in hand
    obs = tensor.flatten()

    overlay = method.compute(obs, 0, step=1, run_id="test")

    assert overlay.label == "heuristic"
    assert overlay.importance[0, 0] == 1.0
    assert overlay.importance.sum() == 1.0

    # Test with 2 cards
    tensor[3, 13, 0] = 1  # Joker in hand
    obs = tensor.flatten()
    overlay = method.compute(obs, 0, step=1, run_id="test")
    assert overlay.importance[0, 0] == 0.5
    assert overlay.importance[3, 13] == 0.5
    assert np.isclose(overlay.importance.sum(), 1.0)


def test_heuristic_empty_hand() -> None:
    """Test heuristic with no cards."""
    method = HandHighlightMethod()
    obs = np.zeros((4 * 14 * 1))
    overlay = method.compute(obs, 0, step=1, run_id="test")
    assert overlay.importance.sum() == 0.0
