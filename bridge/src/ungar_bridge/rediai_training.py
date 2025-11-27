"""RediAI training adapter for UNGAR games.

This module provides a wrapper around the standard training loop that hooks into
RediAI's workflow context for metric logging when available.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from .rediai_rewardlab import (
    build_reward_decomposition_payload,
    is_rediai_rewardlab_available,
    log_reward_decomposition,
)
from .rediai_xai import is_rediai_xai_available, log_overlays_as_artifact
from .training import TrainingResult, train_high_card_duel
from .types import WorkflowRecorder

HAS_REDAI = True
try:
    from RediAI.registry.recorder import workflow_context
except ImportError:
    HAS_REDAI = False

    async def _dummy_ctx() -> WorkflowRecorder:
        class _R:
            async def record_metric(self, *a: Any, **kw: Any) -> None:
                ...

            async def record_artifact(self, *a: Any, **kw: Any) -> None:
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
    record_overlays: bool = False,
) -> TrainingResult:
    """Run High Card Duel training within a RediAI workflow context.

    If RediAI is installed, metrics (average reward, last reward) are logged
    to the workflow registry. If not, it uses a dummy recorder.

    Args:
        num_episodes: Number of episodes to train.
        epsilon: Exploration rate.
        seed: Random seed.
        record_overlays: Whether to generate and log XAI overlays.

    Returns:
        The training result containing rewards and other stats.
    """
    async with workflow_context() as recorder:
        result = train_high_card_duel(
            num_episodes=num_episodes,
            epsilon=epsilon,
            seed=seed,
            record_overlays=record_overlays,
        )

        if result.rewards:
            avg_reward = sum(result.rewards) / len(result.rewards)
            await recorder.record_metric(
                "ungar.high_card.avg_reward", avg_reward, episodes=num_episodes
            )
            await recorder.record_metric("ungar.high_card.last_reward", result.rewards[-1])

        # Log overlays if requested and present
        if record_overlays and "last_overlay" in result.config:
            # The training loop currently returns a single overlay for the last episode
            overlays = [result.config["last_overlay"]]
            await log_overlays_as_artifact(
                recorder, overlays, artifact_name="ungar_high_card_overlays.json"
            )

        # Log reward decomposition (always for High Card Duel since we added it)
        if result.components:
            payload = build_reward_decomposition_payload(result)
            await log_reward_decomposition(recorder, payload)

        return result


def require_rediai() -> None:
    """Raise RuntimeError if RediAI is not available."""
    if not is_rediai_available():
        raise RuntimeError(
            "RediAI is required for full registry integration. "
            'Install with `pip install "ungar-bridge[rediai]"` or '
            "run in local-dummy mode via train_high_card_duel()."
        )


def require_full_rediai_stack() -> None:
    """Ensure all RediAI components (Core, XAI, RewardLab) are available.

    This helper is for production deployments that expect full functionality.
    """
    if not (
        is_rediai_available() and is_rediai_xai_available() and is_rediai_rewardlab_available()
    ):
        raise RuntimeError(
            "Full RediAI stack not available. Install rediai and ensure XAI/RewardLab "
            "are enabled, or run plain train_high_card_duel() without RediAI integration."
        )
