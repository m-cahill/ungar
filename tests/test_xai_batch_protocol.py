"""Tests for batch overlay protocol (M22-A)."""

import numpy as np
from ungar.xai import CardOverlay
from ungar.xai_methods import HandHighlightMethod, RandomOverlayMethod


def test_default_compute_batch_fallback() -> None:
    """Test that non-gradient methods use default compute_batch fallback."""
    method = HandHighlightMethod()

    # Create batch of 3 items
    batch = [
        {"obs": np.ones(56, dtype=np.float32), "action": 0, "step": 1, "run_id": "test"},
        {"obs": np.ones(56, dtype=np.float32), "action": 1, "step": 2, "run_id": "test"},
        {"obs": np.ones(56, dtype=np.float32), "action": 2, "step": 3, "run_id": "test"},
    ]

    # Call compute_batch
    overlays = method.compute_batch(batch)

    # Should return 3 overlays
    assert len(overlays) == 3

    # Each should be a valid CardOverlay
    for i, overlay in enumerate(overlays):
        assert isinstance(overlay, CardOverlay)
        assert overlay.step == i + 1
        assert overlay.label == "heuristic"
        assert overlay.importance.shape == (4, 14)


def test_random_method_batch_fallback() -> None:
    """Test that RandomOverlayMethod works with compute_batch."""
    method = RandomOverlayMethod()

    batch = [
        {"obs": np.zeros(56, dtype=np.float32), "action": 0, "step": 10, "run_id": "test"},
        {"obs": np.zeros(56, dtype=np.float32), "action": 0, "step": 20, "run_id": "test"},
    ]

    overlays = method.compute_batch(batch)

    assert len(overlays) == 2
    assert overlays[0].step == 10
    assert overlays[1].step == 20

    # Random overlays should be normalized
    assert np.isclose(overlays[0].importance.sum(), 1.0, atol=1e-6)
    assert np.isclose(overlays[1].importance.sum(), 1.0, atol=1e-6)


def test_empty_batch() -> None:
    """Test that empty batch returns empty list."""
    method = HandHighlightMethod()

    overlays = method.compute_batch([])

    assert overlays == []


def test_single_item_batch() -> None:
    """Test that batch_size=1 works correctly."""
    method = HandHighlightMethod()

    batch = [
        {"obs": np.ones(56, dtype=np.float32), "action": 0, "step": 5, "run_id": "test"},
    ]

    overlays = method.compute_batch(batch)

    assert len(overlays) == 1
    assert overlays[0].step == 5


def test_batch_preserves_meta() -> None:
    """Test that metadata is preserved through batch processing."""
    method = HandHighlightMethod()

    custom_meta = {"episode": 42, "player": 1}
    batch = [
        {
            "obs": np.ones(56, dtype=np.float32),
            "action": 0,
            "step": 1,
            "run_id": "test",
            "meta": custom_meta,
        },
    ]

    overlays = method.compute_batch(batch)

    assert overlays[0].meta == custom_meta


def test_batch_handles_missing_meta() -> None:
    """Test that missing meta field is handled gracefully."""
    method = HandHighlightMethod()

    batch = [
        {"obs": np.ones(56, dtype=np.float32), "action": 0, "step": 1, "run_id": "test"},
    ]

    overlays = method.compute_batch(batch)

    # Should not crash, meta should be None or empty dict
    assert overlays[0].meta is None or overlays[0].meta == {}
