"""Metrics loading and analysis utilities."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class MetricsFrame:
    """Lightweight container for training metrics."""

    steps: List[int]
    rewards: List[float]
    lengths: List[float]
    other: Dict[str, List[float]]

    @classmethod
    def from_dict_list(cls, data: List[Dict[str, Any]]) -> MetricsFrame:
        if not data:
            return cls([], [], [], {})

        steps = []
        rewards = []
        lengths = []
        other: Dict[str, List[float]] = {}

        # Initialize other keys
        first = data[0]
        for k in first.keys():
            if k not in ("step", "episode_reward", "episode_length"):
                other[k] = []

        for row in data:
            steps.append(int(row.get("step", 0)))
            rewards.append(float(row.get("episode_reward", 0.0)))
            lengths.append(float(row.get("episode_length", 0.0)))

            for k in other:
                other[k].append(float(row.get(k, 0.0)))

        return cls(steps, rewards, lengths, other)


def load_metrics(run_dir: str | Path) -> MetricsFrame:
    """Load metrics from a run directory (CSV or JSONL)."""
    run_dir = Path(run_dir)

    # Try CSV first
    csv_path = run_dir / "metrics.csv"
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return MetricsFrame.from_dict_list(data)

    # Try JSONL
    jsonl_path = run_dir / "metrics.jsonl"
    if jsonl_path.exists():
        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return MetricsFrame.from_dict_list(data)

    raise FileNotFoundError(f"No metrics found in {run_dir}")
