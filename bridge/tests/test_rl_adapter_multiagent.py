import numpy as np
import pytest
from ungar.game import GameEnv
from ungar.games.high_card_duel import make_high_card_duel_spec
from ungar_bridge.rl_adapter import UngarGymEnv


@pytest.fixture
def duel_env() -> UngarGymEnv:
    """Fixture for High Card Duel wrapped in RL adapter."""
    spec = make_high_card_duel_spec()
    game_env = GameEnv(spec)
    return UngarGymEnv(game_env)


def test_initial_state(duel_env: UngarGymEnv) -> None:
    """Test initial state properties after reset."""
    obs, info = duel_env.reset(seed=42)

    # Check dimensions
    assert duel_env.num_players == 2
    assert duel_env.current_player == 0  # P0 always starts

    # Obs should be numpy array (4x14xn)
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 3
    assert obs.shape[:2] == (4, 14)

    # Legal actions should be [0] (reveal)
    actions = duel_env.legal_actions()
    assert actions == [0]


def test_step_progression(duel_env: UngarGymEnv) -> None:
    """Test stepping through a full episode."""
    duel_env.reset(seed=42)

    # ----------------
    # Step 1: Player 0 moves
    # ----------------
    assert duel_env.current_player == 0
    actions = duel_env.legal_actions()
    obs, reward, terminated, truncated, info = duel_env.step(actions[0])

    # High Card Duel: P0 reveals -> P1's turn
    assert terminated is False
    assert truncated is False
    assert reward == 0.0  # No reward until end
    assert duel_env.current_player == 1

    # ----------------
    # Step 2: Player 1 moves
    # ----------------
    actions = duel_env.legal_actions()
    obs, reward, terminated, truncated, info = duel_env.step(actions[0])

    # Game should end
    assert terminated is True
    assert truncated is False
    assert isinstance(reward, float)  # Should be 1.0, -1.0, or 0.0

    # Verify we can't step again
    with pytest.raises(RuntimeError, match="Cannot step a terminated environment"):
        duel_env.step(0)


def test_reset_clears_state(duel_env: UngarGymEnv) -> None:
    """Ensure reset allows replaying."""
    duel_env.reset()
    duel_env.step(0)
    duel_env.step(0)
    assert duel_env.legal_actions() == []

    # Reset again
    duel_env.reset()
    assert duel_env.current_player == 0
    assert len(duel_env.legal_actions()) > 0


def test_invalid_action(duel_env: UngarGymEnv) -> None:
    """Test behavior with illegal actions."""
    duel_env.reset()
    with pytest.raises(ValueError, match="Illegal action"):
        duel_env.step(999)  # 999 is not a valid move ID
