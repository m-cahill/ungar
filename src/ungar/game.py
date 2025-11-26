"""Core game interfaces and protocols for UNGAR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, Tuple

from .tensor import CardTensor

PlayerId = int
Reward = float
MoveId = int


@dataclass(frozen=True, slots=True)
class Move:
    """A game move in UNGAR's core model.

    Attributes:
        id: Stable integer identifier (0..N-1 per game).
        name: Human-readable name, e.g. "reveal".
    """

    id: MoveId
    name: str


class IllegalMoveError(Exception):
    """Raised when an illegal move is applied to a GameState."""


class GameState(Protocol):
    """Generic interface for turn-based, finite card games."""

    def current_player(self) -> PlayerId:
        """Return the ID of the player whose turn it is (0..N-1)."""

    def legal_moves(self) -> Sequence[Move]:
        """Return all legal moves for the current player."""

    def is_terminal(self) -> bool:
        """Return True if the game is over."""

    def apply_move(self, move: Move) -> "GameState":
        """Return the next state after applying the given move.

        Raises:
            IllegalMoveError: If the move is not legal in this state.
        """

    def returns(self) -> Tuple[Reward, ...]:
        """Return final rewards per player.

        Precondition:
            is_terminal() is True.
        """

    def to_tensor(self, player: PlayerId) -> CardTensor:
        """Return the 4×14×n observation tensor for `player`."""


@dataclass(frozen=True, slots=True)
class GameSpec:
    """Static description of a game."""

    name: str
    num_players: int

    def initial_state(self, seed: int | None = None) -> GameState:
        """Return a fresh initial state, optionally seeded."""
        raise NotImplementedError


@dataclass
class GameEnv:
    """Simple environment wrapper around a GameSpec."""

    spec: GameSpec
    state: GameState | None = None

    def reset(self, seed: int | None = None) -> GameState:
        """Reset environment to initial state."""
        self.state = self.spec.initial_state(seed=seed)
        return self.state

    def step(
        self, move: Move
    ) -> tuple[GameState, Tuple[Reward, ...], bool, Mapping[str, Any]]:
        """Apply move to current state."""
        if self.state is None:
            msg = "Environment must be reset() before step()."
            raise RuntimeError(msg)
        next_state = self.state.apply_move(move)
        self.state = next_state
        done = next_state.is_terminal()
        rewards = next_state.returns() if done else tuple()
        info: Mapping[str, Any] = {}
        return next_state, rewards, done, info

