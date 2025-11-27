"""End-to-end cross-game validation test.

Ensures the pipeline handles both High Card Duel and Mini Spades seamlessly.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import pytest
from ungar_bridge.rediai_spades import train_spades_mini_rediai
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
async def test_cross_game_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Run both games training loops and verify artifacts."""

    recorder_instance = FakeRecorder()

    @asynccontextmanager
    async def mock_workflow_context() -> AsyncGenerator[FakeRecorder, None]:
        yield recorder_instance

    # Patch workflow context in both modules
    monkeypatch.setattr("ungar_bridge.rediai_training.workflow_context", mock_workflow_context)
    monkeypatch.setattr("ungar_bridge.rediai_spades.workflow_context", mock_workflow_context)

    # Patch XAI bridge to use tmp_path
    async def mock_log_overlays(
        recorder: Any, overlays: Any, artifact_name: str, tmp_dir: str = "."
    ) -> str:
        from ungar_bridge.rediai_xai import log_overlays_as_artifact

        return await log_overlays_as_artifact(recorder, overlays, artifact_name, str(tmp_path))

    monkeypatch.setattr("ungar_bridge.rediai_training.log_overlays_as_artifact", mock_log_overlays)
    monkeypatch.setattr("ungar_bridge.rediai_spades.log_overlays_as_artifact", mock_log_overlays)

    # Patch RewardLab bridge to use tmp_path
    async def mock_log_rewards(recorder: Any, payload: Any, tmp_dir: str = ".") -> None:
        from ungar_bridge.rediai_rewardlab import log_reward_decomposition

        await log_reward_decomposition(recorder, payload, str(tmp_path))

    monkeypatch.setattr("ungar_bridge.rediai_training.log_reward_decomposition", mock_log_rewards)
    monkeypatch.setattr("ungar_bridge.rediai_spades.log_reward_decomposition", mock_log_rewards)

    # 1. Run High Card Duel
    await train_high_card_duel_rediai(
        num_episodes=5,
        record_overlays=True,
    )

    # 2. Run Mini Spades
    await train_spades_mini_rediai(
        num_episodes=5,
        record_overlays=True,
    )

    # Assertions
    artifact_names = [a[0] for a in recorder_instance.artifacts]

    # Check overlays for both
    assert "ungar_high_card_overlays.json" in artifact_names
    assert "ungar_spades_overlays.json" in artifact_names

    # Check reward decomposition (both name it ungar_reward_decomposition.json currently,
    # so we should see it appear twice in the list)
    assert artifact_names.count("ungar_reward_decomposition.json") == 2

    # Check metrics
    metric_names = [m[0] for m in recorder_instance.metrics]
    assert "ungar.high_card.avg_reward" in metric_names
    assert "ungar.spades.avg_reward" in metric_names
