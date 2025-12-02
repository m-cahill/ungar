import pytest
from ungar.cards import Card, all_cards
from ungar.enums import Rank, Suit
from ungar.tensor import CardTensor


@pytest.mark.smoke
def test_validate_exclusive_planes_ok() -> None:
    p1_hand = {Card(Suit.SPADES, Rank.ACE)}
    p2_hand = {Card(Suit.HEARTS, Rank.ACE)}

    tensor = CardTensor.from_plane_card_map({"p1": p1_hand, "p2": p2_hand})
    tensor.validate_exclusive_planes(["p1", "p2"])


def test_validate_exclusive_planes_overlap() -> None:
    card = Card(Suit.SPADES, Rank.ACE)
    # Same card in both planes
    tensor = CardTensor.from_plane_card_map({"p1": {card}, "p2": {card}})

    with pytest.raises(ValueError, match="more than one"):
        tensor.validate_exclusive_planes(["p1", "p2"])


def test_validate_partition_ok() -> None:
    # Simple partition: reds vs blacks
    reds = {c for c in all_cards() if c.suit in (Suit.HEARTS, Suit.DIAMONDS)}
    blacks = {c for c in all_cards() if c.suit in (Suit.SPADES, Suit.CLUBS)}

    tensor = CardTensor.from_plane_card_map({"red": reds, "black": blacks})
    tensor.validate_partition(["red", "black"])


def test_validate_partition_missing_card() -> None:
    # Missing Spades Ace
    all_minus_one = set(all_cards()) - {Card(Suit.SPADES, Rank.ACE)}
    tensor = CardTensor.from_plane_card_map({"all": all_minus_one})

    with pytest.raises(ValueError, match="not present in any plane"):
        tensor.validate_partition(["all"])


def test_validate_partition_overlap() -> None:
    # Complete but overlapping
    all_c = set(all_cards())
    # Put Ace of Spades in both
    tensor = CardTensor.from_plane_card_map(
        {"all": all_c, "extra": {Card(Suit.SPADES, Rank.ACE)}}
    )

    with pytest.raises(ValueError, match="appears in multiple planes"):
        tensor.validate_partition(["all", "extra"])


def test_validation_unknown_plane() -> None:
    tensor = CardTensor.empty(["a"])
    with pytest.raises(KeyError, match="Unknown plane"):
        tensor.validate_exclusive_planes(["b"])
