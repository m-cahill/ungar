"""High Card Duel game implementation."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence, Tuple

from ungar.cards import Card, all_cards
from ungar.enums import Rank
from ungar.game import GameSpec, GameState, IllegalMoveError, Move
from ungar.tensor import CardTensor

HIGH_CARD_DUEL_NAME = "high_card_duel"
NUM_PLAYERS = 2


def high_card_value(card: Card) -> int:
    """Return a numeric value for ranking cards in HighCardDuel.

    Joker is treated as highest rank. Ace is second highest.
    """
    if card.rank is Rank.JOKER:
        return 100
    if card.rank is Rank.ACE:
        return 99
    return card.rank.value


@dataclass(frozen=True, slots=True)
class HighCardDuelState(GameState):
    """State for High Card Duel.

    Each player has 1 card.
    Phase 1: Player 0 reveals.
    Phase 2: Player 1 reveals.
    End: Compare cards.
    """

    hands: Tuple[Card, Card]
    revealed: Tuple[bool, bool]
    _current_player: int  # 0 or 1, -1 when terminal

    def current_player(self) -> int:
        return self._current_player

    def legal_moves(self) -> Sequence[Move]:
        if self.is_terminal():
            return ()
        return (Move(id=0, name="reveal"),)

    def is_terminal(self) -> bool:
        return all(self.revealed)

    def apply_move(self, move: Move) -> "HighCardDuelState":
        if self.is_terminal():
            raise IllegalMoveError("Game is already terminal.")
        legal = self.legal_moves()
        if move not in legal:
            raise IllegalMoveError(f"Illegal move {move} in state {self!r}")

        # Reveal for current player
        revealed = list(self.revealed)
        revealed[self._current_player] = True
        # Next player or terminal
        next_player = 1 - self._current_player
        if all(revealed):
            next_player = -1

        return HighCardDuelState(
            hands=self.hands,
            revealed=tuple(revealed),  # type: ignore[arg-type]
            _current_player=next_player,
        )

    def returns(self) -> Tuple[float, float]:
        if not self.is_terminal():
            raise RuntimeError("Returns only defined for terminal states.")

        v0 = high_card_value(self.hands[0])
        v1 = high_card_value(self.hands[1])
        if v0 > v1:
            return (1.0, -1.0)
        if v1 > v0:
            return (-1.0, 1.0)
        return (0.0, 0.0)

    def to_tensor(self, player: int) -> CardTensor:
        """Return tensor view for player.

        Planes:
        - my_hand
        - opponent_hand (only if revealed)
        - unseen (everything else)
        """
        all_deck = set(all_cards())
        my = {self.hands[player]}
        opp = {self.hands[1 - player]} if self.revealed[1 - player] else set()
        unseen = all_deck - my - opp

        return CardTensor.from_plane_card_map(
            {
                "my_hand": my,
                "opponent_hand": opp,
                "unseen": unseen,
            }
        )


@dataclass(frozen=True, slots=True)
class HighCardDuelSpec(GameSpec):
    """Specification for High Card Duel."""

    def initial_state(self, seed: int | None = None) -> GameState:
        rng = Random(seed)  # nosec B311: pseudo-random is sufficient for game simulation
        # Standard 52 cards only (exclude Jokers)
        deck = [c for c in all_cards() if c.rank is not Rank.JOKER]
        rng.shuffle(deck)
        hands = (deck[0], deck[1])
        return HighCardDuelState(
            hands=hands,
            revealed=(False, False),
            _current_player=0,
        )


def make_high_card_duel_spec() -> HighCardDuelSpec:
    """Create a spec for High Card Duel."""
    return HighCardDuelSpec(name=HIGH_CARD_DUEL_NAME, num_players=NUM_PLAYERS)
