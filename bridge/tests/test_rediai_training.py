"""Tests for the RediAI training adapter."""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, List, Tuple

import pytest
from ungar_bridge.rediai_training import is_rediai_available, train_high_card_duel_rediai


class FakeRecorder:
    """Mock recorder to capture metrics."""
    def __init__(self) -> None:
        self.metrics: List[Tuple[str, float, dict[str, Any]]] = []

    async def record_metric(self, name: str, value: float, **kwargs: Any) -> None:
        """Store metric calls."""
        self.metrics.append((name, value, kwargs))


@pytest.mark.asyncio
async def test_train_high_card_duel_rediai_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test training wrapper with a mocked recorder."""
    
    # Capture recorder instance to inspect later
    recorder_instance = FakeRecorder()

    @asynccontextmanager
    async def mock_workflow_context() -> AsyncGenerator[FakeRecorder, None]:
        yield recorder_instance

    # Patch the workflow_context in the target module
    monkeypatch.setattr("ungar_bridge.rediai_training.workflow_context", mock_workflow_context)

    # Run training
    num_episodes = 10
    result = await train_high_card_duel_rediai(num_episodes=num_episodes, epsilon=0.0, seed=123)

    # Assertions
    assert len(result.rewards) == num_episodes
    
    # Check that metrics were recorded
    assert len(recorder_instance.metrics) >= 2
    
    metric_names = [m[0] for m in recorder_instance.metrics]
    assert "ungar.high_card.avg_reward" in metric_names
    assert "ungar.high_card.last_reward" in metric_names
    
    # Verify values exist
    for name, value, kwargs in recorder_instance.metrics:
        if name == "ungar.high_card.avg_reward":
            assert isinstance(value, float)
            assert kwargs.get("episodes") == num_episodes


def test_is_rediai_available_logic() -> None:
    """Ensure availability check runs without error."""
    # We can't easily change HAS_REDAI at runtime since it's determined at import time,
    # but we can check it returns a boolean.
    assert isinstance(is_rediai_available(), bool)

