"""Adapter for Mini Spades."""

from typing import List

from ungar.enums import CARD_COUNT
from ungar.game import GameEnv, Move
from ungar.games.spades_mini import make_spades_mini_spec


class SpadesAdapter:
    """Adapter for Mini Spades."""

    @property
    def action_space_size(self) -> int:
        # Cards are indexed 0..55. Move.id corresponds to card index.
        return CARD_COUNT

    @property
    def tensor_shape(self) -> int:
        # 4 suits * 14 ranks * 4 planes = 224
        return 224

    def create_env(self) -> GameEnv:
        return GameEnv(make_spades_mini_spec())

    def moves_to_indices(self, moves: List[Move]) -> List[int]:
        return [m.id for m in moves]

    def index_to_move(self, idx: int, legal_moves: List[Move]) -> Move:
        for m in legal_moves:
            if m.id == idx:
                return m
        raise ValueError(f"Action {idx} not in legal moves")
