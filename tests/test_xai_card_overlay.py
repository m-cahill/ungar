"""Tests for core XAI data structures."""

import numpy as np
import pytest
from ungar.xai import CardOverlay, zero_overlay


def test_card_overlay_shape_validation() -> None:
    """Test that CardOverlay validates the importance matrix shape."""
    # Correct shape (4, 14)
    valid_importance = np.zeros((4, 14))
    overlay = CardOverlay(importance=valid_importance, label="test", meta={})
    assert overlay.importance.shape == (4, 14)

    # Incorrect shape
    invalid_importance = np.zeros((4, 13))
    with pytest.raises(ValueError, match="shape"):
        CardOverlay(importance=invalid_importance, label="test", meta={})


def test_zero_overlay() -> None:
    """Test the zero_overlay helper."""
    overlay = zero_overlay("test_zero")
    assert overlay.label == "test_zero"
    assert overlay.importance.shape == (4, 14)
    assert np.all(overlay.importance == 0.0)
    assert overlay.meta == {}

    # With meta
    meta = {"foo": "bar"}
    overlay_meta = zero_overlay("test_meta", meta=meta)
    assert overlay_meta.meta == meta
