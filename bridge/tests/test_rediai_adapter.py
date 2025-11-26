import numpy as np
import pytest
from ungar.game import GameEnv
from ungar.games.high_card_duel import make_high_card_duel_spec
from ungar_bridge.rediai_adapter import HAS_REDAI, RediAIUngarAdapter, make_rediai_env


def test_adapter_initialization():
    """Test that we can initialize the adapter with a UNGAR env."""
    spec = make_high_card_duel_spec()
    env = GameEnv(spec)
    adapter = RediAIUngarAdapter(env)
    assert adapter.env is env


@pytest.mark.asyncio
async def test_adapter_api_compliance():
    """Test the adapter implementation of the EnvAdapter protocol."""
    spec = make_high_card_duel_spec()
    env = GameEnv(spec)
    adapter = RediAIUngarAdapter(env)

    # 1. Reset
    obs = await adapter.reset()
    assert obs is not None
    # Check shape: 4 x 14 x n (HighCardDuel has 3 planes)
    assert obs.shape == (4, 14, 3)
    assert obs.dtype == np.float32

    # 2. Legal moves
    moves = adapter.legal_moves()
    assert len(moves) > 0

    # 3. Step
    # HighCardDuel requires 2 moves to finish (one per player)
    move = moves[0]
    next_obs, reward, done, info = await adapter.step(move)

    # Check return types for first step
    assert isinstance(next_obs, np.ndarray)
    assert next_obs.shape == (4, 14, 3)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert done is False  # First move, game not over

    # Step again to finish
    moves = adapter.legal_moves()
    assert len(moves) > 0
    move = moves[0]
    next_obs, reward, done, info = await adapter.step(move)
    assert done is True


def test_make_rediai_env_guard():
    """Test that factory function checks for RediAI availability."""
    if not HAS_REDAI:
        with pytest.raises(RuntimeError, match="RediAI is not installed"):
            make_rediai_env("high_card_duel")
    else:
        # If RediAI happens to be installed (e.g. user env), it should work
        adapter = make_rediai_env("high_card_duel")
        assert isinstance(adapter, RediAIUngarAdapter)


def test_make_rediai_env_unknown_game():
    """Test error handling for unknown games."""
    if HAS_REDAI:
        with pytest.raises(ValueError, match="Unknown game"):
            make_rediai_env("invalid_game_name")
    # If not HAS_REDAI, the runtime error takes precedence, which is fine.
