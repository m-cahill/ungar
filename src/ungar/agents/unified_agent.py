"""Unified Agent Interface.

Defines the protocol for all agents in UNGAR (random, RL, etc.)
and provides a common transition structure for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np


@dataclass(frozen=True)
class Transition:
    """A single experience tuple for RL training."""

    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    legal_moves_next: List[int]


class UnifiedAgent(Protocol):
    """Protocol for any agent that can act in an UNGAR environment."""

    def act(self, obs: np.ndarray, legal_moves: List[int]) -> int:
        """Select an action given observation and legal moves.

        Args:
            obs: The current observation tensor (flattened or structured).
            legal_moves: List of legal action indices.

        Returns:
            Selected action index.
        """
        ...

    def train_step(self, transition: Transition) -> None:
        """Perform a single training step (e.g. push to buffer, update weights).

        Args:
            transition: Experience tuple.
        """
        ...

    def save(self, path: str) -> None:
        """Save agent state to disk."""
        ...

    def load(self, path: str) -> None:
        """Load agent state from disk."""
        ...


class RandomUnifiedAgent:
    """Baseline agent that selects actions uniformly at random."""

    def __init__(self, seed: int | None = None) -> None:
        import random

        self.rng = random.Random(seed)

    def act(self, obs: np.ndarray, legal_moves: List[int]) -> int:
        if not legal_moves:
            raise ValueError("No legal moves provided to agent.")
        return self.rng.choice(legal_moves)

    def train_step(self, transition: Transition) -> None:
        # Random agent does not learn
        pass

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass
