from hypothesis import given
from hypothesis import strategies as st

from ungar.cards import Card


@given(st.integers(min_value=0, max_value=55))
def test_index_bijection(index: int) -> None:
    """Property: to_index(from_index(i)) == i for all valid indices."""
    assert Card.from_index(index).to_index() == index


@given(st.integers(min_value=0, max_value=55))
def test_from_index_invariants(index: int) -> None:
    """Property: Card created from index has valid suit/rank."""
    card = Card.from_index(index)
    assert card.suit in list(card.suit.__class__)
    assert card.rank in list(card.rank.__class__)
