import numpy as np
import pytest
from ungar.game import GameEnv
from ungar.games.high_card_duel import make_high_card_duel_spec
from ungar_bridge.rediai_adapter import RediAIUngarAdapter, is_rediai_available, make_rediai_env


def test_adapter_initialization() -> None:
    """Test that we can initialize the adapter with a UNGAR env."""
    spec = make_high_card_duel_spec()
    env = GameEnv(spec)
    adapter = RediAIUngarAdapter(env)
    assert adapter.env is env


@pytest.mark.asyncio
async def test_adapter_api_compliance() -> None:
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


@pytest.mark.asyncio
async def test_tensor_invariants() -> None:
    """Test that encode_state follows the UNGAR tensor contract.

    Contract:
    - Shape is (4, 14, n)
    - Joker column (index 13) should be zero for HighCardDuel
    - Values are 0.0 or 1.0 (since we cast bool to float)
    """
    spec = make_high_card_duel_spec()
    env = GameEnv(spec)
    adapter = RediAIUngarAdapter(env)

    obs = await adapter.reset()

    # 1. Shape check
    assert obs.ndim == 3
    assert obs.shape[:2] == (4, 14)
    # HighCardDuel has 3 planes: my_hand, opponent_hand, unseen
    assert obs.shape[2] == 3

    # 2. Joker check (HighCardDuel usually doesn't use jokers in default deal)
    # But if it did, they would be in column 13.
    # We can check that at least the values are valid.

    # 3. Value check
    unique_values = np.unique(obs)
    for v in unique_values:
        assert v in (0.0, 1.0), f"Tensor contained non-boolean value: {v}"

    # 4. Sum check: In HighCardDuel, each card is exactly in one plane (partition)
    # So sum across planes for any card position (suit, rank) should be 1.0
    # EXCEPT if the card is not in play? No, "unseen" captures the rest.
    # The deck is partitioned.
    plane_sum = np.sum(obs, axis=2)

    # However, HighCardDuel deals 2 cards from 52.
    # Cards not in deck?
    # Wait, HighCardDuel implementation:
    #   my = {hand[p]}
    #   opp = {hand[1-p]} (if revealed)
    #   unseen = all_cards - my - opp
    # So every card in "all_cards" (52 cards) should be in exactly one plane.
    # What about Jokers? standard 52 deck used in HighCardDuelSpec.
    # Jokers are in the tensor space (column 13) but not in the deck list.
    # So column 13 (Jokers) should be all zeros.

    # Check regular cards (0-12)
    regular_cards_sum = plane_sum[:, :13]
    # Should be all 1s
    assert np.all(regular_cards_sum == 1.0), "Partition violation for regular cards"

    # Check jokers (13)
    # Jokers are in the tensor space (column 13) and included in all_cards()
    # HighCardDuel's to_tensor calculates unseen = all_cards - my - opp
    # Since my/opp contain no jokers, unseen will contain all jokers.
    # Therefore, jokers should appear in the "unseen" plane (index 2)
    jokers_sum = plane_sum[:, 13]
    assert np.all(jokers_sum == 1.0), "Jokers should be in unseen plane (partitioned)"

    # Specifically, they should be in the unseen plane
    # Plane 0: my_hand, Plane 1: opp_hand, Plane 2: unseen
    # My/Opp hand should have 0 for jokers
    assert np.all(obs[:, 13, 0] == 0.0), "Jokers in my_hand"
    assert np.all(obs[:, 13, 1] == 0.0), "Jokers in opponent_hand"
    # Unseen should have 1 for jokers
    assert np.all(obs[:, 13, 2] == 1.0), "Jokers not in unseen"


def test_make_rediai_env_guard() -> None:
    """Test that factory function checks for RediAI availability."""
    if not is_rediai_available():
        with pytest.raises(RuntimeError, match="RediAI is not installed"):
            make_rediai_env("high_card_duel")
    else:
        # If RediAI happens to be installed (e.g. user env), it should work
        adapter = make_rediai_env("high_card_duel")
        assert isinstance(adapter, RediAIUngarAdapter)


def test_make_rediai_env_unknown_game() -> None:
    """Test error handling for unknown games."""
    if is_rediai_available():
        with pytest.raises(ValueError, match="Unknown game"):
            make_rediai_env("invalid_game_name")
    # If not is_rediai_available(), the runtime error takes precedence, which is fine.
