"""Tests for the UNGAR bridge package."""

import numpy as np
import pytest
from ungar.game import GameEnv, Move
from ungar.games.high_card_duel import make_high_card_duel_spec
from ungar_bridge.noop_adapter import NoOpAdapter
from ungar_bridge.rl_adapter import UngarGymEnv


def test_noop_adapter() -> None:
    """Test the NoOpAdapter passes things through."""
    adapter = NoOpAdapter()
    adapter.initialize({})

    # Test state pass-through
    assert adapter.state_to_external("foo") == "foo"

    # Test move pass-through
    move = Move(id=0, name="test")
    assert adapter.external_to_move(move) == move

    with pytest.raises(TypeError):
        adapter.external_to_move("not-a-move")


def test_rl_gym_env_high_card_duel() -> None:
    """Test the Gym-like wrapper with HighCardDuel."""
    spec = make_high_card_duel_spec()
    game_env = GameEnv(spec)
    gym_env = UngarGymEnv(game_env, player_id=0)

    # Reset
    obs, info = gym_env.reset(seed=42)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (4, 14, 3)  # HighCardDuel tensor shape
    assert "legal_moves" in info

    # Step 1 (P0 reveal)
    move = info["legal_moves"][0]
    obs, reward, done, truncated, info = gym_env.step(move)

    # In HighCardDuel, P0 moves, then it's P1's turn.
    # But our current Gym wrapper just steps the underlying env.
    # P0 receives 0 reward until game ends.
    assert not done
    assert reward == 0.0

    # Step 2 (P1 reveal - simulating opponent)
    # Since we are sharing the env instance, we can just step again.
    # In a real Gym setup, we'd need logic to handle opponent turns.
    move = info["legal_moves"][0]
    obs, reward, done, truncated, info = gym_env.step(move)

    assert done
    assert reward != 0.0  # Win or lose
    assert info["legal_moves"] == []
