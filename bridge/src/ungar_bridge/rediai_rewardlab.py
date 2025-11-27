"""RediAI RewardLab Integration Bridge.

This module bridges UNGAR's decomposed rewards to RediAI's RewardLab analysis tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .training import TrainingResult
from .types import WorkflowRecorder

HAS_REDAI_REWARDLAB = True

try:
    # Adjust import paths based on actual RediAI repo
    from RediAI.rewardlab import RewardDecomposer  # noqa: F401
except ImportError:
    HAS_REDAI_REWARDLAB = False


def is_rediai_rewardlab_available() -> bool:
    """Return True if RediAI RewardLab features are available."""
    return HAS_REDAI_REWARDLAB


def build_reward_decomposition_payload(
    result: TrainingResult,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Convert training results into a RewardLab-compatible payload.

    Args:
        result: The training result object.
        experiment_id: Optional ID for grouping runs.

    Returns:
        A dictionary matching the RewardLab schema.
    """
    return {
        "experiment_id": experiment_id or "ungar_high_card_demo",
        "episodes": [
            {
                "episode_index": idx,
                "total_reward": total,
                "components": dict(components),
            }
            for idx, (total, components) in enumerate(
                zip(result.rewards, result.components), start=1
            )
        ],
    }


async def log_reward_decomposition(
    recorder: WorkflowRecorder,
    payload: dict[str, Any],
    tmp_dir: str = ".",
) -> None:
    """Log reward decomposition payload as a RediAI artifact.

    Args:
        recorder: The RediAI workflow recorder.
        payload: The decomposition data dictionary.
        tmp_dir: Directory to write the temporary file.
    """
    path = Path(tmp_dir) / "reward_decomposition.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    await recorder.record_artifact("ungar_reward_decomposition.json", str(path))
