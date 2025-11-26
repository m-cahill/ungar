"""Simulation runner for playing games."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import List, Tuple

from .game import GameEnv, GameSpec, Move


@dataclass
class Episode:
    """A recorded sequence of a game."""

    states: List[object]
    moves: List[Move]
    rewards: Tuple[float, ...]


def play_random_episode(spec: GameSpec, seed: int | None = None) -> Episode:
    """Play a full episode using random valid moves."""
    rng = Random(seed)
    env = GameEnv(spec=spec)
    state = env.reset(seed=seed)

    states: List[object] = [state]
    moves: List[Move] = []

    # Just in case of infinite loops in broken games, we might want a limit, 
    # but for M02 the loop condition is strict.
    while not state.is_terminal():
        legal = state.legal_moves()
        if not legal:
            raise RuntimeError("Non-terminal state with no legal moves.")
        move = rng.choice(legal)
        state, rewards, done, info = env.step(move)
        states.append(state)
        moves.append(move)
        if done:
            return Episode(states=states, moves=moves, rewards=rewards)

    # If we somehow started in terminal (should not happen usually but valid)
    return Episode(states=states, moves=moves, rewards=state.returns())

