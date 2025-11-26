from ungar.cards import Card, all_cards
from ungar.enums import CARD_COUNT, RANK_COUNT, SUIT_COUNT


def test_counts() -> None:
    assert SUIT_COUNT == 4
    assert RANK_COUNT == 14
    assert CARD_COUNT == 56


def test_all_cards() -> None:
    deck = all_cards()
    assert len(deck) == 56
    assert len(set(deck)) == 56  # All unique


def test_index_roundtrip_exhaustive() -> None:
    """Verify to_index/from_index consistency for all 56 cards."""
    for i in range(56):
        card = Card.from_index(i)
        assert card.to_index() == i


def test_from_index_bounds() -> None:
    """Verify out-of-bounds indices raise ValueError."""
    import pytest

    with pytest.raises(ValueError, match="Card index must be in"):
        Card.from_index(-1)

    with pytest.raises(ValueError, match="Card index must be in"):
        Card.from_index(56)
