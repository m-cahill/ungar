"""Enumerations for card suits and ranks."""

from __future__ import annotations

from enum import Enum, auto

SUIT_COUNT = 4
RANK_COUNT = 14
CARD_COUNT = 56  # 4 suits * 14 ranks


class Suit(Enum):
    """Standard playing card suits."""

    SPADES = auto()
    HEARTS = auto()
    DIAMONDS = auto()
    CLUBS = auto()

    def __str__(self) -> str:
        """Return the name of the suit."""
        return self.name


class Rank(Enum):
    """Standard playing card ranks plus Joker."""

    ACE = auto()
    TWO = auto()
    THREE = auto()
    FOUR = auto()
    FIVE = auto()
    SIX = auto()
    SEVEN = auto()
    EIGHT = auto()
    NINE = auto()
    TEN = auto()
    JACK = auto()
    QUEEN = auto()
    KING = auto()
    JOKER = auto()  # Column 13

    def __str__(self) -> str:
        """Return the name of the rank."""
        return self.name
