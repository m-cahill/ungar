import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from ungar.cards import Card, all_cards
from ungar.enums import Rank, Suit
from ungar.tensor import CardTensor


def test_roundtrip_two_plane_layout() -> None:
    my_hand = {
        Card(Suit.SPADES, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
    }
    discard = set(all_cards()) - my_hand

    tensor = CardTensor.from_plane_card_map({"my_hand": my_hand, "discard": discard})

    assert set(tensor.cards_in_plane("my_hand")) == my_hand
    assert set(tensor.cards_in_plane("discard")) == discard


def test_plane_access() -> None:
    tensor = CardTensor.empty(["p1"])
    assert tensor.plane("p1").shape == (4, 14)

    with pytest.raises(KeyError):
        tensor.plane("missing")


def test_flat_plane() -> None:
    card = Card(Suit.SPADES, Rank.ACE)  # Index 0
    tensor = CardTensor.from_plane_card_map({"one": [card]})

    flat = tensor.flat_plane("one")
    assert flat.shape == (56,)
    assert flat[0] is np.True_
    assert flat[1] is np.False_

    # Roundtrip flat
    restored = CardTensor.from_flat_plane("one", flat)
    assert set(restored.cards_in_plane("one")) == {card}


def test_from_flat_plane_invalid_size() -> None:
    flat = np.zeros(55, dtype=bool)
    with pytest.raises(ValueError, match="must have size 56"):
        CardTensor.from_flat_plane("x", flat)


@given(st.lists(st.integers(min_value=0, max_value=55), unique=True))
def test_property_roundtrip_random_cards(indices: list[int]) -> None:
    cards = {Card.from_index(i) for i in indices}
    tensor = CardTensor.from_plane_card_map({"test_plane": cards})
    restored = set(tensor.cards_in_plane("test_plane"))
    assert restored == cards
