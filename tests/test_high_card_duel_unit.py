"""Unit tests for High Card Duel logic."""

import pytest
from ungar.cards import Card
from ungar.enums import Rank, Suit
from ungar.game import GameEnv, IllegalMoveError, Move
from ungar.games.high_card_duel import (
    HighCardDuelState,
    high_card_value,
    make_high_card_duel_spec,
)


@pytest.mark.smoke
def test_high_card_value() -> None:
    """Test card ranking logic."""
    c2 = Card(Suit.SPADES, Rank.TWO)
    c3 = Card(Suit.SPADES, Rank.THREE)
    ace = Card(Suit.SPADES, Rank.ACE)

    assert high_card_value(c2) < high_card_value(c3)
    assert high_card_value(c3) < high_card_value(ace)
    # Jokers exist in all suits in 4x14, e.g. Spades Joker
    joker = Card(Suit.SPADES, Rank.JOKER)
    assert high_card_value(ace) < high_card_value(joker)


@pytest.mark.smoke
def test_game_flow() -> None:
    """Test a full game flow with fixed hands."""
    # Mock state with known hands: P0 wins
    hands = (Card(Suit.SPADES, Rank.ACE), Card(Suit.SPADES, Rank.TWO))
    state = HighCardDuelState(
        hands=hands,
        revealed=(False, False),
        _current_player=0,
    )

    # Check initial
    assert state.current_player() == 0
    assert not state.is_terminal()
    assert state.legal_moves() == (Move(id=0, name="reveal"),)

    # P0 reveals
    state = state.apply_move(Move(id=0, name="reveal"))
    assert state.current_player() == 1
    assert not state.is_terminal()
    assert state.revealed == (True, False)

    # P1 reveals
    state = state.apply_move(Move(id=0, name="reveal"))
    assert state.is_terminal()
    assert state.revealed == (True, True)

    # Rewards
    rewards = state.returns()
    assert rewards == (1.0, -1.0)


def test_tie() -> None:
    """Test tie scenario (same rank, different suits not possible in 1 deck without reuse,
    but strictly speaking logic should handle equality)."""
    # Force equality by reusing card object or same rank different suit
    c1 = Card(Suit.SPADES, Rank.ACE)
    c2 = Card(Suit.HEARTS, Rank.ACE)
    state = HighCardDuelState(
        hands=(c1, c2),
        revealed=(True, True),
        _current_player=-1,
    )
    assert state.returns() == (0.0, 0.0)


def test_spec_determinism() -> None:
    """Test that seeding works."""
    spec = make_high_card_duel_spec()
    state1 = spec.initial_state(seed=42)
    state2 = spec.initial_state(seed=42)
    state3 = spec.initial_state(seed=999)

    # Test equality
    assert isinstance(state1, HighCardDuelState)
    assert isinstance(state2, HighCardDuelState)
    assert isinstance(state3, HighCardDuelState)

    # Check attributes exist and match
    assert state1.hands == state2.hands
    assert state1.hands != state3.hands  # High probability


def test_illegal_moves() -> None:
    """Test error handling."""
    spec = make_high_card_duel_spec()
    env = GameEnv(spec)
    env.reset()

    # Wrong move name/id
    with pytest.raises(IllegalMoveError):
        env.step(Move(id=1, name="fold"))  # id 1 is irrelevant, name is key, or logic

    # Play after terminal
    env.step(Move(id=0, name="reveal"))
    env.step(Move(id=0, name="reveal"))
    # Now terminal
    with pytest.raises(IllegalMoveError):
        env.step(Move(id=0, name="reveal"))
