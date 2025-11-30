"""Tests for overlay aggregation logic."""

import numpy as np
import pytest
from ungar.analysis.overlays import compute_mean_overlay
from ungar.xai import CardOverlay


def test_compute_mean_overlay() -> None:
    """Test averaging overlays."""
    # Overlay 1: all 1s
    o1 = CardOverlay(
        run_id="r1",
        label="test",
        agg="none",
        step=1,
        importance=np.ones((4, 14)),
        meta={}
    )
    # Overlay 2: all 3s
    o2 = CardOverlay(
        run_id="r1",
        label="test",
        agg="none",
        step=2,
        importance=np.full((4, 14), 3.0),
        meta={}
    )
    
    mean_o = compute_mean_overlay([o1, o2])
    
    assert mean_o.agg == "mean"
    assert mean_o.importance.shape == (4, 14)
    # Mean of 1 and 3 is 2
    assert np.allclose(mean_o.importance, 2.0)
    assert mean_o.meta["count"] == 2


def test_compute_mean_empty() -> None:
    with pytest.raises(ValueError):
        compute_mean_overlay([])

