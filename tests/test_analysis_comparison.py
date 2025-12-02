"""Tests for overlay comparison logic."""

import numpy as np
from ungar.analysis.overlays import compare_overlays
from ungar.xai import CardOverlay


def test_compare_overlays() -> None:
    """Test A - B comparison."""
    # Overlay A: all 1s
    o_a = CardOverlay(
        run_id="r1", label="A", agg="none", step=1, importance=np.ones((4, 14)), meta={}
    )
    # Overlay B: all 0.5s
    o_b = CardOverlay(
        run_id="r1",
        label="B",
        agg="none",
        step=1,
        importance=np.full((4, 14), 0.5),
        meta={},
    )

    diff = compare_overlays([o_a], [o_b])

    assert diff.agg == "diff"
    # A - B = 1 - 0.5 = 0.5
    # Normalized by max(|0.5|) = 0.5 -> 1.0
    assert np.allclose(diff.importance, 1.0)
    assert diff.meta["comparison"] is True


def test_compare_overlays_sign() -> None:
    """Test signed difference."""
    # A has 1.0 at (0,0), B has 0.0
    # A has 0.0 at (0,1), B has 1.0

    imp_a = np.zeros((4, 14))
    imp_a[0, 0] = 1.0

    imp_b = np.zeros((4, 14))
    imp_b[0, 1] = 1.0

    o_a = CardOverlay(run_id="r1", label="A", agg="none", step=1, importance=imp_a, meta={})
    o_b = CardOverlay(run_id="r1", label="B", agg="none", step=1, importance=imp_b, meta={})

    diff = compare_overlays([o_a], [o_b])

    # Diff at (0,0) = 1 - 0 = 1
    # Diff at (0,1) = 0 - 1 = -1
    # Max abs = 1
    # Normalized diffs: 1 and -1

    assert np.isclose(diff.importance[0, 0], 1.0)
    assert np.isclose(diff.importance[0, 1], -1.0)
