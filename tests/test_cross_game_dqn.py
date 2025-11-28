"""End-to-end tests for cross-game DQN training."""

import pytest
from ungar.training.config import DQNConfig
from ungar.training.train_dqn import train_dqn


@pytest.mark.parametrize("game_name", ["high_card_duel", "spades_mini", "gin_rummy"])
def test_train_dqn_smoke(game_name: str) -> None:
    """Smoke test: train DQN for a few episodes on each game."""
    config = DQNConfig(total_episodes=5, batch_size=16, replay_capacity=100)

    result = train_dqn(
        game_name=game_name,
        config=config,
        seed=42,
    )

    assert len(result.rewards) == 5
    assert "avg_reward" in result.metrics
    assert isinstance(result.metrics["avg_reward"], float)
