"""Visualization tools for training metrics and overlays."""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from ungar.analysis.metrics import load_metrics
from ungar.enums import RANK_COUNT, SUIT_COUNT


def plot_learning_curve(
    run_dirs: List[str | Path],
    out_path: str | Path | None = None,
    smooth_window: int = 10,
    metric: str = "rewards",
) -> None:
    """Plot learning curves for one or more runs."""
    plt.figure(figsize=(10, 6))

    for run_dir in run_dirs:
        try:
            metrics = load_metrics(run_dir)
        except FileNotFoundError:
            continue

        data = getattr(metrics, metric, [])
        if not data:
            continue

        steps = metrics.steps

        # Smoothing
        if smooth_window > 1 and len(data) > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(data, kernel, mode="valid")
            valid_steps = steps[smooth_window - 1 :]
            plt.plot(valid_steps, smoothed, label=Path(run_dir).name)
        else:
            plt.plot(steps, data, label=Path(run_dir).name)

    plt.xlabel("Steps")
    plt.ylabel(metric.capitalize())
    plt.title(f"Learning Curve: {metric}")
    plt.legend()
    plt.grid(True)

    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()

    plt.close()


def plot_overlay_heatmap(
    importance: np.ndarray,
    out_path: str | Path | None = None,
    title: str = "Card Importance Heatmap",
) -> None:
    """Plot a 4x14 heatmap of card importance."""
    plt.figure(figsize=(12, 5))  # Avoid magic number 4

    # Setup grid
    plt.imshow(importance, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(label="Importance")

    # Labels
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "Joker"]
    suits = ["Spades", "Hearts", "Diamonds", "Clubs"]

    plt.xticks(range(RANK_COUNT), ranks)
    plt.yticks(range(SUIT_COUNT), suits)

    plt.title(title)

    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()

    plt.close()
