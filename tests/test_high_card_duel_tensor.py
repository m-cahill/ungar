"""Tests for High Card Duel tensor representation."""

from ungar.cards import Card
from ungar.enums import Rank, Suit
from ungar.games.high_card_duel import HighCardDuelState


def test_tensor_invariants() -> None:
    """Test tensor partition invariants for HighCardDuel."""
    # Setup state: P0 has Ace, P1 has King.
    # P0 revealed, P1 not.
    p0_card = Card(Suit.SPADES, Rank.ACE)
    p1_card = Card(Suit.SPADES, Rank.KING)

    state = HighCardDuelState(
        hands=(p0_card, p1_card),
        revealed=(True, False),
        _current_player=1,
    )

    # 1. P0's perspective
    # P0 knows own card (Ace). P1 is not revealed -> unkown.
    # So Ace in my_hand. King in unseen.
    t0 = state.to_tensor(player=0)
    t0.validate_partition(["my_hand", "opponent_hand", "unseen"])

    # Check Ace is in my_hand
    assert p0_card in t0.cards_in_plane("my_hand")
    # Check King is in unseen
    assert p1_card in t0.cards_in_plane("unseen")
    # Check opponent_hand is empty
    assert not t0.cards_in_plane("opponent_hand")

    # 2. P1's perspective
    # P1 knows own card (King). P0 IS revealed -> known.
    # Ace in opponent_hand. King in my_hand.
    t1 = state.to_tensor(player=1)
    t1.validate_partition(["my_hand", "opponent_hand", "unseen"])

    assert p1_card in t1.cards_in_plane("my_hand")
    assert p0_card in t1.cards_in_plane("opponent_hand")
    # unseen should contain the rest of the deck
    assert len(t1.cards_in_plane("unseen")) == 54
    # 52 cards + 4 Jokers = 56 total slots (per tensor definition)
    # Actually tensor uses 4x14 = 56.
    # So unseen should have 54 cards.
    assert len(t1.cards_in_plane("unseen")) == 54


def test_tensor_fully_revealed() -> None:
    """Test tensor when everything is public."""
    p0_card = Card(Suit.SPADES, Rank.ACE)
    p1_card = Card(Suit.SPADES, Rank.KING)
    state = HighCardDuelState(
        hands=(p0_card, p1_card),
        revealed=(True, True),
        _current_player=-1,
    )

    t0 = state.to_tensor(player=0)
    t0.validate_partition(["my_hand", "opponent_hand", "unseen"])

    assert p0_card in t0.cards_in_plane("my_hand")
    assert p1_card in t0.cards_in_plane("opponent_hand")
    assert len(t0.cards_in_plane("unseen")) == 54
