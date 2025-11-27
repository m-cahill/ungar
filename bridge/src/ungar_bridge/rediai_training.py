"""RediAI training adapter for UNGAR games.

This module provides a wrapper around the standard training loop that hooks into
RediAI's workflow context for metric logging when available.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Protocol

from .training import TrainingResult, train_high_card_duel

HAS_REDAI = True
try:
    from RediAI.registry.recorder import workflow_context
except ImportError:
    HAS_REDAI = False

    class _DummyRecorder(Protocol):
        async def record_metric(self, name: str, value: float, **kwargs: Any) -> None:
            ...

    async def _dummy_ctx() -> _DummyRecorder:
        class _R:
            async def record_metric(self, *a: Any, **kw: Any) -> None:
                ...

        return _R()

    @asynccontextmanager
    async def workflow_context() -> Any:
        """Dummy context manager when RediAI is not available."""
        yield await _dummy_ctx()


def is_rediai_available() -> bool:
    """Return True if RediAI is installed and available."""
    return HAS_REDAI


async def train_high_card_duel_rediai(
    num_episodes: int = 1000,
    epsilon: float = 0.1,
    seed: int | None = None,
) -> TrainingResult:
    """Run High Card Duel training within a RediAI workflow context.

    If RediAI is installed, metrics (average reward, last reward) are logged
    to the workflow registry. If not, it uses a dummy recorder.

    Args:
        num_episodes: Number of episodes to train.
        epsilon: Exploration rate.
        seed: Random seed.

    Returns:
        The training result containing rewards and other stats.
    """
    async with workflow_context() as recorder:
        result = train_high_card_duel(num_episodes=num_episodes, epsilon=epsilon, seed=seed)

        if result.rewards:
            avg_reward = sum(result.rewards) / len(result.rewards)
            await recorder.record_metric(
                "ungar.high_card.avg_reward", avg_reward, episodes=num_episodes
            )
            await recorder.record_metric("ungar.high_card.last_reward", result.rewards[-1])

        return result


def require_rediai() -> None:
    """Raise RuntimeError if RediAI is not available."""
    if not is_rediai_available():
        raise RuntimeError(
            "RediAI is required for full registry integration. "
            'Install with `pip install "ungar-bridge[rediai]"` or '
            "run in local-dummy mode via train_high_card_duel()."
        )
