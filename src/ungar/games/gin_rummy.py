"""Gin Rummy implementation.

A simplified but structurally complete Gin Rummy environment for UNGAR.
Rules MVP:
- 2 players
- 10 cards dealt
- Draw from Stock or Discard
- Discard 1 card
- Knock if deadwood <= 10
- Gin if deadwood == 0
- Simple scoring
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ungar.cards import Card
from ungar.game import GameSpec, GameState, IllegalMoveError, Move, PlayerId, Reward
from ungar.tensor import CardTensor


class GinMoveType(enum.Enum):
    DRAW_STOCK = 0
    DRAW_DISCARD = 1
    DISCARD = 2
    KNOCK = 3  # Combined with discard usually, but let's make it explicit or part of discard


@dataclass(frozen=True, slots=True)
class GinMove(Move):
    """Extended Move with type info for Gin Rummy."""

    move_type: GinMoveType
    card_index: int | None = None  # relevant for DISCARD


def get_card_value(card: Card) -> int:
    """Return Gin Rummy point value of a card."""
    # Ace = 1, 2-9 = face value, 10/J/Q/K = 10
    # Rank enum is 1-based (ACE=1 ... KING=13)
    rank_val = card.rank.value
    if rank_val >= 10:
        return 10
    return rank_val


def calculate_deadwood(hand: Sequence[Card]) -> int:
    """Calculate deadwood points for a hand.

    This is a simplified greedy meld detector for the MVP.
    Optimal meld detection is NP-hard (set cover), but for 10 cards it's tractable.
    """
    if not hand:
        return 0

    # Naive implementation: assume no melds for MVP baseline to get structure working.
    # TODO: Implement real meld detection (sets and runs).
    # For now, everything is deadwood.
    return sum(get_card_value(c) for c in hand)


@dataclass(frozen=True)
class GinRummyState(GameState):
    """State for Gin Rummy."""

    hands: Tuple[Tuple[Card, ...], ...]
    stock_pile: Tuple[Card, ...]
    discard_pile: Tuple[Card, ...]
    player_turn: PlayerId
    phase: str  # "DRAW" or "DISCARD"

    # Game end state
    knocker: PlayerId | None = None
    gin: bool = False

    # Constants
    NUM_PLAYERS = 2
    CARDS_PER_HAND = 10
    MAX_DEADWOOD_KNOCK = 10

    def current_player(self) -> PlayerId:
        return self.player_turn

    def legal_moves(self) -> Sequence[Move]:
        if self.is_terminal():
            return []

        moves: List[Move] = []

        if self.phase == "DRAW":
            # Can draw from stock if not empty
            if self.stock_pile:
                moves.append(Move(id=0, name="DRAW_STOCK"))

            # Can draw from discard if not empty
            if self.discard_pile:
                moves.append(Move(id=1, name="DRAW_DISCARD"))

        elif self.phase == "DISCARD":
            hand = self.hands[self.player_turn]

            # Can discard any card in hand
            # Move ID structure: 100 + card_index to separate from draw actions
            for card in hand:
                moves.append(Move(id=100 + card.to_index(), name=f"DISCARD_{card}"))

            # Can knock if deadwood condition met (after discard)
            # This logic is complex because knocking happens *with* a discard.
            # For MVP, let's treat "DISCARD_X" as "Discard X and pass turn".
            # We might need a separate "KNOCK_X" move or just auto-knock if optimal?
            # Let's add explicit KNOCK moves: 200 + card_index

            # Check if we can knock with each card
            for card in hand:
                # Simulate hand after discard
                temp_hand = list(hand)
                temp_hand.remove(card)
                dw = calculate_deadwood(temp_hand)
                if dw <= self.MAX_DEADWOOD_KNOCK:
                    moves.append(Move(id=200 + card.to_index(), name=f"KNOCK_{card}"))

        return moves

    def is_terminal(self) -> bool:
        return self.knocker is not None or (
            not self.stock_pile and self.phase == "DRAW" and not self.discard_pile
        )  # simple stock empty rule

    def apply_move(self, move: Move) -> GameState:
        # Basic validation
        # legal = self.legal_moves() ... (omitted for speed, relying on agent)

        if self.phase == "DRAW":
            if move.id == 0:  # STOCK
                if not self.stock_pile:
                    # Stock empty - in real rules usually draw/reshuffle or end.
                    # Here let's end game or raise error if legal_moves check failed.
                    raise IllegalMoveError("Cannot draw from empty stock")

                card = self.stock_pile[-1]
                new_stock = self.stock_pile[:-1]
                new_hand = self.hands[self.player_turn] + (card,)

                return GinRummyState(
                    hands=self._update_hand(self.player_turn, new_hand),
                    stock_pile=new_stock,
                    discard_pile=self.discard_pile,
                    player_turn=self.player_turn,
                    phase="DISCARD",
                )

            elif move.id == 1:  # DISCARD
                if not self.discard_pile:
                    raise IllegalMoveError("Cannot draw from empty discard")

                card = self.discard_pile[-1]
                new_discard = self.discard_pile[:-1]
                new_hand = self.hands[self.player_turn] + (card,)

                return GinRummyState(
                    hands=self._update_hand(self.player_turn, new_hand),
                    stock_pile=self.stock_pile,
                    discard_pile=new_discard,
                    player_turn=self.player_turn,
                    phase="DISCARD",
                )

        elif self.phase == "DISCARD":
            is_knock = move.id >= 200
            card_idx = (move.id - 200) if is_knock else (move.id - 100)
            card = Card.from_index(card_idx)

            current_hand = list(self.hands[self.player_turn])
            if card not in current_hand:
                raise IllegalMoveError(f"Card {card} not in hand")

            current_hand.remove(card)
            new_hand = tuple(current_hand)
            new_discard = self.discard_pile + (card,)

            if is_knock:
                # Terminal state
                return GinRummyState(
                    hands=self._update_hand(self.player_turn, new_hand),
                    stock_pile=self.stock_pile,
                    discard_pile=new_discard,
                    player_turn=self.player_turn,
                    phase="END",
                    knocker=self.player_turn,
                    gin=(calculate_deadwood(new_hand) == 0),
                )
            else:
                # Pass turn
                return GinRummyState(
                    hands=self._update_hand(self.player_turn, new_hand),
                    stock_pile=self.stock_pile,
                    discard_pile=new_discard,
                    player_turn=(self.player_turn + 1) % 2,
                    phase="DRAW",
                )

        raise IllegalMoveError(f"Unknown move {move.id}")

    def _update_hand(
        self, player: PlayerId, new_hand: Tuple[Card, ...]
    ) -> Tuple[Tuple[Card, ...], ...]:
        h = list(self.hands)
        h[player] = new_hand
        return tuple(h)

    def returns(self) -> Tuple[Reward, ...]:
        if not self.is_terminal():
            return (0.0, 0.0)

        if self.knocker is None:
            return (0.0, 0.0)  # Draw

        # Scoring
        knocker_hand = self.hands[self.knocker]
        opponent = (self.knocker + 1) % 2
        opponent_hand = self.hands[opponent]

        knocker_dw = calculate_deadwood(knocker_hand)
        opponent_dw = calculate_deadwood(opponent_hand)

        # Simple scoring
        # If Gin: 25 + (opp_dw)
        # If Knock: (opp_dw - knocker_dw)
        # Undercut: (knocker_dw - opp_dw) + 25 to opponent

        score = 0.0
        winner = self.knocker

        if self.gin:
            score = 25.0 + opponent_dw
        elif knocker_dw < opponent_dw:
            score = float(opponent_dw - knocker_dw)
        else:
            # Undercut
            score = 25.0 + (knocker_dw - opponent_dw)
            winner = opponent

        rewards = [0.0, 0.0]
        rewards[winner] = score
        rewards[(winner + 1) % 2] = -score
        return tuple(rewards)

    def to_tensor(self, player: PlayerId) -> CardTensor:
        """Observation tensor.

        Planes:
        0: my_hand
        1: in_discard (visible)
        2: top_discard (1-hot)
        3: legal_draw (if phase=DRAW)
        4: legal_discard (if phase=DISCARD)
        """
        planes = {
            "my_hand": self.hands[player],
            "in_discard": self.discard_pile,
            "top_discard": [self.discard_pile[-1]] if self.discard_pile else [],
        }

        # Helper for legal moves mask
        legal = self.legal_moves() if self.current_player() == player else []
        legal_ids = {m.id for m in legal}

        # Visualize draw options (global flags spread across cards?)
        # Or specifically highlight top discard if drawable
        drawable_discard = []
        if 1 in legal_ids and self.discard_pile:  # ID 1 is DRAW_DISCARD
            drawable_discard = [self.discard_pile[-1]]
        planes["legal_draw_discard"] = drawable_discard

        # Visualize discardable/knockable cards
        # Map IDs 100+ and 200+ back to cards
        discardable = []
        knockable = []
        for m in legal:
            if 100 <= m.id < 200:
                discardable.append(Card.from_index(m.id - 100))
            elif m.id >= 200:
                knockable.append(Card.from_index(m.id - 200))

        planes["legal_discard"] = discardable
        planes["legal_knock"] = knockable

        return CardTensor.from_plane_card_map(planes)


@dataclass(frozen=True, slots=True)
class GinRummySpec(GameSpec):
    name: str = "gin_rummy"
    num_players: int = 2

    def initial_state(self, seed: int | None = None) -> GameState:
        import random

        from ungar.enums import RANK_COUNT, SUIT_COUNT

        rng = random.Random(seed)

        # Standard 52 card deck
        standard_deck_size = SUIT_COUNT * (RANK_COUNT - 1)
        deck = [Card.from_index(i) for i in range(standard_deck_size)]
        rng.shuffle(deck)

        hands: List[List[Card]] = [[], []]
        for _ in range(10):
            for p in range(2):
                hands[p].append(deck.pop())

        # One card to discard pile to start
        discard = [deck.pop()]

        return GinRummyState(
            hands=tuple(tuple(h) for h in hands),
            stock_pile=tuple(deck),
            discard_pile=tuple(discard),
            player_turn=0,
            phase="DRAW",
        )


def make_gin_rummy_spec() -> GameSpec:
    return GinRummySpec()
