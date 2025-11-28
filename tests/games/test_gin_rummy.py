"""Tests for Gin Rummy logic."""

from ungar.cards import Card
from ungar.enums import Rank, Suit
from ungar.game import Move
from ungar.games.gin_rummy import (
    GinRummyState,
    calculate_deadwood,
    get_card_value,
    make_gin_rummy_spec,
)


def test_card_values() -> None:
    """Test point values."""
    assert get_card_value(Card(Suit.SPADES, Rank.ACE)) == 1
    assert get_card_value(Card(Suit.HEARTS, Rank.FIVE)) == 5
    assert get_card_value(Card(Suit.CLUBS, Rank.TEN)) == 10
    assert get_card_value(Card(Suit.DIAMONDS, Rank.KING)) == 10


def test_deadwood_calculation_naive() -> None:
    """Test naive deadwood sum (no melds)."""
    hand = [Card(Suit.SPADES, Rank.ACE), Card(Suit.SPADES, Rank.KING)]
    assert calculate_deadwood(hand) == 11
    assert calculate_deadwood([]) == 0


def test_initial_deal() -> None:
    """Test dealing logic."""
    spec = make_gin_rummy_spec()
    state = spec.initial_state(seed=42)
    assert isinstance(state, GinRummyState)

    # 10 cards per player
    assert len(state.hands[0]) == 10
    assert len(state.hands[1]) == 10

    # One discard
    assert len(state.discard_pile) == 1

    # Remainder in stock (52 - 20 - 1 = 31)
    assert len(state.stock_pile) == 31

    assert state.phase == "DRAW"
    assert state.player_turn == 0


def test_draw_stock() -> None:
    """Test drawing from stock."""
    spec = make_gin_rummy_spec()
    state = spec.initial_state(seed=42)

    # Draw from stock (ID 0)
    move = Move(id=0, name="DRAW_STOCK")
    next_state = state.apply_move(move)

    assert isinstance(next_state, GinRummyState)
    assert len(next_state.hands[0]) == 11
    assert len(next_state.stock_pile) == 30
    assert next_state.phase == "DISCARD"
    assert next_state.player_turn == 0  # Still my turn to discard


def test_draw_discard() -> None:
    """Test drawing from discard."""
    spec = make_gin_rummy_spec()
    state = spec.initial_state(seed=42)

    # Draw from discard (ID 1)
    move = Move(id=1, name="DRAW_DISCARD")

    # Should get the top discard
    assert isinstance(state, GinRummyState)
    top_discard = state.discard_pile[-1]

    next_state = state.apply_move(move)
    assert isinstance(next_state, GinRummyState)

    assert len(next_state.hands[0]) == 11
    assert next_state.hands[0][-1] == top_discard
    assert len(next_state.discard_pile) == 0
    assert next_state.phase == "DISCARD"


def test_discard_turn_cycle() -> None:
    """Test discard passes turn."""
    spec = make_gin_rummy_spec()
    state = spec.initial_state(seed=42)

    # Draw first
    state = state.apply_move(Move(id=0, name="DRAW_STOCK"))
    assert isinstance(state, GinRummyState)

    # Discard first card in hand
    card_to_discard = state.hands[0][0]
    move_id = 100 + card_to_discard.to_index()
    state = state.apply_move(Move(id=move_id, name="DISCARD"))

    assert isinstance(state, GinRummyState)
    assert state.phase == "DRAW"
    assert state.player_turn == 1
    assert state.discard_pile[-1] == card_to_discard
    assert len(state.hands[0]) == 10


def test_knock_and_scoring() -> None:
    """Test knocking."""
    spec = make_gin_rummy_spec()
    state = spec.initial_state(seed=42)
    assert isinstance(state, GinRummyState)

    # Force player 0 hand to be very low value (all Aces/Twos) for testing
    low_cards = [
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.DIAMONDS, Rank.ACE),
        Card(Suit.CLUBS, Rank.ACE),
        Card(Suit.SPADES, Rank.TWO),
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.DIAMONDS, Rank.TWO),
        Card(Suit.CLUBS, Rank.TWO),
        Card(Suit.SPADES, Rank.THREE),
        Card(Suit.HEARTS, Rank.THREE),
    ]
    # Stock with a King to draw
    stock = (Card(Suit.SPADES, Rank.KING),)

    state = GinRummyState(
        hands=(tuple(low_cards), tuple(low_cards)),  # Both have low cards
        stock_pile=stock,
        discard_pile=(),
        player_turn=0,
        phase="DRAW",
    )

    # Draw
    state = state.apply_move(Move(id=0, name="DRAW_STOCK"))
    assert isinstance(state, GinRummyState)

    # Discard the King (high value) and Knock (ID 200+)
    # King index is last drawn
    king = state.hands[0][-1]
    knock_move = Move(id=200 + king.to_index(), name="KNOCK")

    final_state = state.apply_move(knock_move)
    assert isinstance(final_state, GinRummyState)

    assert final_state.is_terminal()
    assert final_state.knocker == 0
    assert not final_state.gin

    # Check returns
    # P0 deadwood: ~20
    # P1 deadwood: ~20
    # Should be undercut or draw-ish
    rewards = final_state.returns()
    assert rewards[0] + rewards[1] == 0.0
