"""Mini Spades (2-player trick-taking game).

A simplified version of Spades for testing UNGAR's generality.
Rules:
- 2 players
- 5 cards per player
- Spades is trump
- Highest card of led suit wins unless trumped
- 5 tricks total
- Rewards: +1 per trick won, +5 for game winner
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

from ungar.cards import Card
from ungar.enums import Suit
from ungar.game import GameSpec, GameState, IllegalMoveError, Move, PlayerId, Reward
from ungar.tensor import CardTensor


@dataclass(frozen=True, slots=True)
class TrickState:
    """State of the current trick."""

    # List of (player_id, card) in order played
    moves: Tuple[Tuple[PlayerId, Card], ...] = field(default_factory=tuple)
    led_suit: Suit | None = None

    def winner(self, trump_suit: Suit = Suit.SPADES) -> PlayerId | None:
        """Return the winner of the trick, or None if empty."""
        if not self.moves:
            return None

        winning_player, winning_card = self.moves[0]

        for player, card in self.moves[1:]:
            # If current winner isn't trump but this card is, this card wins
            if winning_card.suit != trump_suit and card.suit == trump_suit:
                winning_player, winning_card = player, card
            # If suits match, higher rank wins (using index comparison)
            elif card.suit == winning_card.suit:
                if card.rank.value > winning_card.rank.value:  # Enum value order
                    # Wait, Rank enum might not be ordered correctly by value by default if auto()
                    # Let's check Rank definition. ACE is typically low or high.
                    # In UNGAR Rank enum: ACE, TWO, ... KING.
                    # Standard Spades: Ace is high.
                    # UNGAR Rank order: ACE(1), TWO(2)... KING(13).
                    # We need to clarify Ace handling. For "Mini Spades", let's assume Ace is low (1)
                    # to keep it simple and consistent with enum order, OR map ranks.
                    # Let's stick to enum index for simplicity: higher index = stronger.
                    # ACE=1 (lowest), KING=13 (highest).
                    winning_player, winning_card = player, card

        return winning_player


@dataclass(frozen=True)
class SpadesMiniState(GameState):
    """State for Mini Spades."""

    hands: Tuple[Tuple[Card, ...], ...]
    tricks_won: Tuple[int, ...]
    current_trick: TrickState
    history: Tuple[Card, ...]  # All cards played so far
    player_turn: PlayerId
    num_tricks_played: int = 0

    # Game constants
    NUM_PLAYERS = 2
    TRUMP_SUIT = Suit.SPADES
    CARDS_PER_HAND = 5

    def current_player(self) -> PlayerId:
        """Return the ID of the player whose turn it is."""
        return self.player_turn

    def legal_moves(self) -> Sequence[Move]:
        """Return legal moves (cards) for the current player."""
        hand = self.hands[self.player_turn]

        # If leading (trick empty), any card is legal
        if not self.current_trick.led_suit:
            return [Move(id=c.to_index(), name=str(c)) for c in hand]

        # If following, must follow suit if possible
        led_suit = self.current_trick.led_suit
        same_suit_cards = [c for c in hand if c.suit == led_suit]

        if same_suit_cards:
            return [Move(id=c.to_index(), name=str(c)) for c in same_suit_cards]

        # If cannot follow suit, any card is legal
        return [Move(id=c.to_index(), name=str(c)) for c in hand]

    def is_terminal(self) -> bool:
        """Game ends after 5 tricks."""
        return self.num_tricks_played >= self.CARDS_PER_HAND

    def apply_move(self, move: Move) -> GameState:
        """Apply a move."""
        # Validate move
        legal = self.legal_moves()
        if move not in legal:
            raise IllegalMoveError(f"Move {move} is not legal. Legal: {legal}")

        # Find card object
        card_idx = move.id
        card = Card.from_index(card_idx)

        # Update hands (remove card)
        new_hands_list = list(self.hands)
        current_hand = list(new_hands_list[self.player_turn])
        # We need to remove by value, Card is frozen/eq-able
        current_hand.remove(card)
        new_hands_list[self.player_turn] = tuple(current_hand)

        # Update trick
        new_moves = self.current_trick.moves + ((self.player_turn, card),)
        led_suit = self.current_trick.led_suit
        if led_suit is None:
            led_suit = card.suit

        new_trick = TrickState(moves=new_moves, led_suit=led_suit)
        new_history = self.history + (card,)

        # Check if trick is complete
        if len(new_trick.moves) == self.NUM_PLAYERS:
            # Trick complete
            winner = new_trick.winner(self.TRUMP_SUIT)
            assert winner is not None

            new_tricks_won_list = list(self.tricks_won)
            new_tricks_won_list[winner] += 1

            return SpadesMiniState(
                hands=tuple(new_hands_list),
                tricks_won=tuple(new_tricks_won_list),
                current_trick=TrickState(),  # Reset trick
                history=new_history,
                player_turn=winner,  # Winner leads next
                num_tricks_played=self.num_tricks_played + 1,
            )
        else:
            # Trick continues
            next_player = (self.player_turn + 1) % self.NUM_PLAYERS
            return SpadesMiniState(
                hands=tuple(new_hands_list),
                tricks_won=self.tricks_won,
                current_trick=new_trick,
                history=new_history,
                player_turn=next_player,
                num_tricks_played=self.num_tricks_played,
            )

    def returns(self) -> Tuple[Reward, ...]:
        """Final rewards."""
        if not self.is_terminal():
            return (0.0,) * self.NUM_PLAYERS

        # +1 per trick won
        rewards = list(float(x) for x in self.tricks_won)

        # +5 for game winner (most tricks)
        max_tricks = max(self.tricks_won)
        # Handle ties? Split pot or no bonus?
        # Simple: if you have max tricks, you get bonus. (Both get if tie)
        for i, tricks in enumerate(self.tricks_won):
            if tricks == max_tricks:
                rewards[i] += 5.0

        return tuple(rewards)

    def to_tensor(self, player: PlayerId) -> CardTensor:
        """Observation tensor.

        Planes:
        0: my_hand
        1: legal_moves
        2: current_trick (cards played in current trick)
        3: history (all played cards)
        """
        planes = {
            "my_hand": self.hands[player],
            "legal_moves": [
                Card.from_index(m.id)
                for m in self.legal_moves()
                if self.current_player() == player
            ],
            "current_trick": [card for _, card in self.current_trick.moves],
            "history": self.history,
        }

        # If it's not my turn, legal_moves is empty visually
        if self.current_player() != player:
            planes["legal_moves"] = []

        return CardTensor.from_plane_card_map(planes)


@dataclass(frozen=True, slots=True)
class SpadesMiniSpec(GameSpec):
    """Spec for Mini Spades."""

    name: str = "spades_mini"
    num_players: int = 2

    def initial_state(self, seed: int | None = None) -> GameState:
        """Deal cards and start game."""
        import random

        from ungar.enums import RANK_COUNT, SUIT_COUNT

        rng = random.Random(seed)

        # Create deck (standard 52, ignore jokers for now to keep simple)
        # UNGAR supports jokers but standard Spades uses 52 cards
        standard_deck_size = SUIT_COUNT * (RANK_COUNT - 1)  # Exclude joker column
        deck = list(range(standard_deck_size))
        rng.shuffle(deck)

        hands: list[list[Card]] = [[], []]
        for i in range(5):
            for p in range(2):
                card_idx = deck.pop()
                hands[p].append(Card.from_index(card_idx))

        # Sort hands for niceness (optional, but good for humans/debugging)
        for h in hands:
            h.sort(key=lambda c: c.to_index())

        return SpadesMiniState(
            hands=tuple(tuple(h) for h in hands),
            tricks_won=(0, 0),
            current_trick=TrickState(),
            history=(),
            player_turn=0,  # Player 0 leads
        )


def make_spades_mini_spec() -> GameSpec:
    return SpadesMiniSpec()
