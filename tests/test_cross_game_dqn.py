"""End-to-end tests for cross-game DQN training."""

import pytest
from ungar.training.train_dqn import train_dqn


@pytest.mark.parametrize("game_name", ["high_card_duel", "spades_mini", "gin_rummy"])
def test_train_dqn_smoke(game_name: str) -> None:
    """Smoke test: train DQN for a few episodes on each game."""
    result = train_dqn(
        game_name=game_name,
        episodes=5,
        buffer_size=100,  # Small buffer for speed
        lr=0.001,
    )

    assert len(result.rewards) == 5
    assert "avg_reward" in result.metrics
    # We don't assert performance, just that it runs without crashing

    # Check that it actually ran (non-zero steps implied if rewards exist)
    # High Card Duel reward is non-deterministic but -1, 0, or 1.
    assert isinstance(result.metrics["avg_reward"], float)
