from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .enums import CARD_COUNT, RANK_COUNT, Rank, Suit


@dataclass(frozen=True, slots=True)
class Card:
    suit: Suit
    rank: Rank

    CARD_COUNT: ClassVar[int] = CARD_COUNT

    def to_index(self) -> int:
        """Return canonical 0-based index in [0, 55] for this card."""
        suit_index = list(Suit).index(self.suit)
        rank_index = list(Rank).index(self.rank)
        return suit_index * len(Rank) + rank_index

    @staticmethod
    def from_index(index: int) -> "Card":
        """Create a Card from canonical 0-based index in [0, 55]."""
        if not 0 <= index < Card.CARD_COUNT:
            msg = f"Card index must be in [0, {Card.CARD_COUNT - 1}], got {index}"
            raise ValueError(msg)
        suit_index, rank_index = divmod(index, len(Rank))
        suit = list(Suit)[suit_index]
        rank = list(Rank)[rank_index]
        return Card(suit=suit, rank=rank)


def all_cards() -> tuple[Card, ...]:
    """Return all 56 cards (including Jokers) in canonical index order."""
    return tuple(Card.from_index(i) for i in range(CARD_COUNT))
