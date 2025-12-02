"""Tests for core XAI data structures."""

import numpy as np
import pytest
from ungar.xai import (
    CardOverlay,
    overlay_from_dict,
    overlay_to_dict,
    zero_overlay,
)


def test_card_overlay_shape_validation() -> None:
    """Test that CardOverlay validates the importance matrix shape."""
    # Correct shape (4, 14)
    valid_importance = np.zeros((4, 14))
    overlay = CardOverlay(
        run_id="test",
        label="test",
        agg="none",
        step=0,
        importance=valid_importance,
        meta={},
    )
    assert overlay.importance.shape == (4, 14)

    # Incorrect shape
    invalid_importance = np.zeros((4, 13))
    with pytest.raises(ValueError, match="shape"):
        CardOverlay(
            run_id="test",
            label="test",
            agg="none",
            step=0,
            importance=invalid_importance,
            meta={},
        )


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


def test_overlay_serialization_roundtrip() -> None:
    """Test overlay_to_dict and overlay_from_dict roundtrip."""
    # Create an overlay with some data
    importance = np.zeros((4, 14))
    importance[0, 0] = 1.0  # Ace of Spades
    importance[3, 13] = 0.5  # Joker

    meta = {"game": "test", "round": 1}
    original = CardOverlay(
        run_id="test_run",
        label="test_roundtrip",
        agg="mean",
        step=10,
        importance=importance,
        meta=meta,
    )

    # Serialize
    data = overlay_to_dict(original)

    # Verify structure
    assert data["label"] == "test_roundtrip"
    assert data["meta"] == meta
    assert isinstance(data["importance"], list)
    assert len(data["importance"]) == 4
    assert len(data["importance"][0]) == 14
    assert data["importance"][0][0] == 1.0

    # Deserialize
    reconstructed = overlay_from_dict(data)

    # Verify
    assert reconstructed.label == original.label
    assert reconstructed.meta == original.meta
    assert np.array_equal(reconstructed.importance, original.importance)
    assert reconstructed.importance.dtype == float
