"""Tests for RediAI RewardLab bridge."""

import json
from pathlib import Path
from typing import Any

import pytest
from ungar_bridge.rediai_rewardlab import (
    build_reward_decomposition_payload,
    log_reward_decomposition,
)
from ungar_bridge.training import TrainingResult


class FakeRecorder:
    """Mock recorder to capture artifact calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def record_artifact(self, name: str, path: str, **_: Any) -> None:
        """Store artifact call arguments."""
        self.calls.append((name, path))


def test_build_reward_decomposition_payload() -> None:
    """Test payload construction from TrainingResult."""
    # Mock result with 2 episodes
    result = TrainingResult(
        rewards=[1.0, -1.0],
        episode_lengths=[1, 1],
        config={},
        components=[
            {"win": 1.0, "base": 0.0},
            {"win": -1.0, "base": 0.0},
        ],
    )

    payload = build_reward_decomposition_payload(result, experiment_id="test_exp")

    assert payload["experiment_id"] == "test_exp"
    assert len(payload["episodes"]) == 2

    ep0 = payload["episodes"][0]
    assert ep0["episode_index"] == 1
    assert ep0["total_reward"] == 1.0
    assert ep0["components"] == {"win": 1.0, "base": 0.0}

    ep1 = payload["episodes"][1]
    assert ep1["episode_index"] == 2
    assert ep1["total_reward"] == -1.0
    assert ep1["components"] == {"win": -1.0, "base": 0.0}


@pytest.mark.asyncio
async def test_log_reward_decomposition(tmp_path: Path) -> None:
    """Test logging reward decomposition as artifact."""
    recorder = FakeRecorder()
    payload = {"experiment_id": "test", "episodes": []}

    await log_reward_decomposition(
        recorder,
        payload,
        tmp_dir=str(tmp_path),
    )

    assert len(recorder.calls) == 1
    name, path = recorder.calls[0]

    assert name == "ungar_reward_decomposition.json"
    assert str(tmp_path) in path

    # Verify content
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["experiment_id"] == "test"
