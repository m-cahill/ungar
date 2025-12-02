"""Adapter for High Card Duel."""

from typing import List, Protocol

from ungar.game import GameEnv, Move
from ungar.games.high_card_duel import make_high_card_duel_spec


class GameAdapter(Protocol):
    """Protocol for game-specific logic needed by RL agents."""

    @property
    def action_space_size(self) -> int:
        ...

    @property
    def tensor_shape(self) -> int:
        ...

    def create_env(self) -> GameEnv:
        ...

    def moves_to_indices(self, moves: List[Move]) -> List[int]:
        ...

    def index_to_move(self, idx: int, legal_moves: List[Move]) -> Move:
        ...


class HighCardAdapter:
    """Adapter for High Card Duel."""

    @property
    def action_space_size(self) -> int:
        return 1  # Only "reveal" action is 0

    @property
    def tensor_shape(self) -> int:
        # 4 suits * 14 ranks * 3 planes = 168
        return 168

    def create_env(self) -> GameEnv:
        return GameEnv(make_high_card_duel_spec())

    def moves_to_indices(self, moves: List[Move]) -> List[int]:
        return [m.id for m in moves]

    def index_to_move(self, idx: int, legal_moves: List[Move]) -> Move:
        for m in legal_moves:
            if m.id == idx:
                return m
        raise ValueError(f"Action {idx} not in legal moves")
