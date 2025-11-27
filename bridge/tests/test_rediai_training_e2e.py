"""End-to-end tests for RediAI training workflow with artifacts."""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import pytest
from ungar_bridge.rediai_training import train_high_card_duel_rediai


class FakeRecorder:
    """Mock recorder to capture metrics and artifacts."""

    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, dict[str, Any]]] = []
        self.artifacts: list[tuple[str, str]] = []

    async def record_metric(self, name: str, value: float, **kwargs: Any) -> None:
        """Store metric calls."""
        self.metrics.append((name, value, kwargs))

    async def record_artifact(self, name: str, path: str, **_: Any) -> None:
        """Store artifact calls."""
        self.artifacts.append((name, path))


@pytest.mark.asyncio
async def test_train_rediai_e2e_with_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Test full training loop with XAI and RewardLab logging."""

    recorder_instance = FakeRecorder()

    @asynccontextmanager
    async def mock_workflow_context() -> AsyncGenerator[FakeRecorder, None]:
        yield recorder_instance

    # Patch workflow context
    monkeypatch.setattr("ungar_bridge.rediai_training.workflow_context", mock_workflow_context)

    # Patch XAI bridge to use tmp_path
    async def mock_log_overlays(
        recorder: Any, overlays: Any, artifact_name: str, tmp_dir: str = "."
    ) -> str:
        # Just call real one but ensure tmp_dir is our test tmp_path
        from ungar_bridge.rediai_xai import log_overlays_as_artifact

        return await log_overlays_as_artifact(recorder, overlays, artifact_name, str(tmp_path))

    monkeypatch.setattr("ungar_bridge.rediai_training.log_overlays_as_artifact", mock_log_overlays)

    # Patch RewardLab bridge to use tmp_path
    async def mock_log_rewards(recorder: Any, payload: Any, tmp_dir: str = ".") -> None:
        from ungar_bridge.rediai_rewardlab import log_reward_decomposition

        await log_reward_decomposition(recorder, payload, str(tmp_path))

    monkeypatch.setattr("ungar_bridge.rediai_training.log_reward_decomposition", mock_log_rewards)

    # Run training
    result = await train_high_card_duel_rediai(
        num_episodes=5,
        seed=42,
        record_overlays=True,
    )

    # Assertions
    assert len(result.rewards) == 5

    # Check metrics
    metric_names = [m[0] for m in recorder_instance.metrics]
    assert "ungar.high_card.avg_reward" in metric_names

    # Check artifacts
    artifact_names = [a[0] for a in recorder_instance.artifacts]

    # Should have XAI overlays (since record_overlays=True)
    assert "ungar_high_card_overlays.json" in artifact_names

    # Should have RewardLab decomposition (always enabled for High Card Duel)
    assert "ungar_reward_decomposition.json" in artifact_names
