"""Tests for analysis tools (metrics and plots)."""

import csv
from pathlib import Path

import numpy as np
import pytest
from ungar.analysis.metrics import load_metrics
from ungar.analysis.overlays import aggregate_overlays, load_overlays


@pytest.fixture
def sample_run_dir(tmp_path: Path) -> Path:
    """Create a fake run directory with metrics and overlays."""
    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()

    # Create metrics.csv
    metrics_file = run_dir / "metrics.csv"
    with open(metrics_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "episode_reward", "loss"])
        writer.writeheader()
        for i in range(10):
            writer.writerow({"step": i, "episode_reward": i * 1.0, "loss": 0.5})

    # Create overlays
    overlay_dir = run_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    import json

    from ungar.xai import overlay_to_dict, zero_overlay

    # We serialize as LIST of overlays per file because load_overlays expects that
    # OR we need to see how load_overlays is implemented.
    # M17 docs said: One file per run + method.
    # M19 plan says: runs/<run_id>/overlays/<label>_<step>.json (single overlay per file)
    # Let's check load_overlays implementation in src/ungar/analysis/overlays.py

    # Overlay 1: uniform 0.5
    o1 = zero_overlay("test")
    o1.importance.fill(0.5)
    with open(overlay_dir / "o1.json", "w") as f:
        # If load_overlays supports single dict:
        json.dump(overlay_to_dict(o1), f)

    # Overlay 2: uniform 1.0
    o2 = zero_overlay("test")
    o2.importance.fill(1.0)
    with open(overlay_dir / "o2.json", "w") as f:
        json.dump(overlay_to_dict(o2), f)

    return run_dir


def test_load_metrics(sample_run_dir: Path) -> None:
    """Test loading metrics from CSV."""
    df = load_metrics(sample_run_dir)
    assert len(df.steps) == 10
    assert df.rewards[-1] == 9.0
    assert "loss" in df.other
    assert len(df.other["loss"]) == 10


def test_load_and_aggregate_overlays(sample_run_dir: Path) -> None:
    """Test loading and aggregating overlays."""
    overlays = load_overlays(sample_run_dir)
    assert len(overlays) == 2

    mean_map = aggregate_overlays(overlays, method="mean")
    assert mean_map.shape == (4, 14)
    assert np.allclose(mean_map, 0.75)  # (0.5 + 1.0) / 2

    max_map = aggregate_overlays(overlays, method="max")
    assert np.allclose(max_map, 1.0)


def test_plot_smoke(sample_run_dir: Path, tmp_path: Path) -> None:
    """Smoke test for plotting functions (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        pytest.skip("matplotlib not installed")

    from ungar.analysis.plots import plot_learning_curve, plot_overlay_heatmap

    # Learning curve
    out_curve = tmp_path / "curve.png"
    plot_learning_curve([sample_run_dir], out_path=out_curve)
    assert out_curve.exists()

    # Heatmap
    out_map = tmp_path / "map.png"
    heatmap = np.zeros((4, 14))
    plot_overlay_heatmap(heatmap, out_path=out_map)
    assert out_map.exists()
