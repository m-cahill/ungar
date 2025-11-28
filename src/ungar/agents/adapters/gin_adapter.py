"""Adapter for Gin Rummy."""

from typing import List

from ungar.game import GameEnv, Move
from ungar.games.gin_rummy import make_gin_rummy_spec


class GinAdapter:
    """Adapter for Gin Rummy."""

    @property
    def action_space_size(self) -> int:
        # IDs:
        # 0: Draw Stock
        # 1: Draw Discard
        # 100-155: Discard card X
        # 200-255: Knock card X
        # Max ID is around 255. Let's reserve 300 to be safe/simple.
        return 300

    @property
    def tensor_shape(self) -> int:
        # 4 suits * 14 ranks * 6 planes = 336
        return 336

    def create_env(self) -> GameEnv:
        return GameEnv(make_gin_rummy_spec())

    def moves_to_indices(self, moves: List[Move]) -> List[int]:
        return [m.id for m in moves]

    def index_to_move(self, idx: int, legal_moves: List[Move]) -> Move:
        for m in legal_moves:
            if m.id == idx:
                return m
        raise ValueError(f"Action {idx} not in legal moves")
