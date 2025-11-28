"""End-to-end tests for cross-game PPO training."""

import pytest
from ungar.training.config import PPOConfig
from ungar.training.train_ppo import train_ppo


@pytest.mark.parametrize("game_name", ["high_card_duel", "spades_mini", "gin_rummy"])
def test_train_ppo_smoke(game_name: str) -> None:
    """Smoke test: train PPO for a few episodes on each game."""
    config = PPOConfig(
        total_episodes=5,
        batch_size=16,
        minibatch_size=8,
        update_epochs=2,
        max_steps_per_episode=100,
        seed=42,
    )

    result = train_ppo(game_name=game_name, config=config, seed=42)

    assert len(result.rewards) == 5
    assert "avg_reward" in result.metrics
    assert isinstance(result.metrics["avg_reward"], float)
