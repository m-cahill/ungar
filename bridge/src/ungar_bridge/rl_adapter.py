"""RL-style adapter for UNGAR games (Gym-like interface)."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from ungar.game import GameEnv

from .adapter_base import UngarRLAdapter


class UngarGymEnv(UngarRLAdapter):
    """A Gym-like wrapper around UNGAR GameEnv supporting multi-agent play."""

    def __init__(self, game_env: GameEnv) -> None:
        """Initialize the RL environment wrapper.

        Args:
            game_env: The underlying UNGAR GameEnv to wrap.
        """
        self.game_env = game_env
        self._current_player = 0
        self._num_players = game_env.spec.num_players

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Random seed for the new game state.
            options: Additional options (unused for now).

        Returns:
            A tuple of (observation, info).
        """
        state = self.game_env.reset(seed=seed)
        self._current_player = state.current_player()
        
        # Get observation for the current player
        tensor = state.to_tensor(self._current_player)
        
        return tensor.data, {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute a step using the given action index.

        Args:
            action: The index of the move to apply from the legal moves list.

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        state = self.game_env.state
        if state is None:
            raise RuntimeError("Environment must be reset before calling step.")

        if state.is_terminal():
            # If called on terminal state, follow Gym convention (or raise)
            # M07 plan says: Raise or handle. Raising is safer for logic bugs.
            raise RuntimeError("Cannot step a terminated environment.")

        legal_moves = state.legal_moves()
        
        # Convert index to actual Move object
        # Note: This relies on legal_actions() returning indices 0..len(legal)-1
        # or using the Move.id directly?
        # The protocol says step(action: int). 
        # A common RL pattern is Discrete(N) where action is an index into the full action space.
        # However, for variable legal moves, we usually pass the index into the legal moves list
        # OR the Move.id if the action space is fixed max size.
        # For High Card Duel, Move.id is stable (0=reveal).
        
        # Let's support action as an index into the *current* legal moves for simplicity 
        # in this masked-action setting, or map it if we had a full action space.
        # Given "legal_actions() -> list[int]", let's assume these are indices into a fixed action space
        # OR Move IDs.
        # HighCardDuel only has one move type "reveal" (id=0).
        
        # If the input `action` corresponds to a Move.id:
        target_move = None
        for move in legal_moves:
            if move.id == action:
                target_move = move
                break
        
        if target_move is None:
             raise ValueError(f"Illegal action {action} for current player {self._current_player}")

        # Apply move
        next_state, rewards, done, info = self.game_env.step(target_move)
        
        # Reward for the player who acted
        step_reward = 0.0
        if done:
            # If game ended, we get rewards.
            # rewards tuple matches player indices.
            step_reward = rewards[self._current_player]
        
        # Update current player
        # If done, current_player might be meaningless (-1), but usually last player stays?
        # UNGAR HighCardDuelState returns -1 when terminal.
        # We should stick to the player who just acted or 0? 
        # For RL loop (obs, reward...), we need the obs for the *next* agent to act.
        # If done, obs usually is final state for someone.
        
        if done:
            # Game over. No next player.
            # We can return the state for the player who just acted, or default P0.
            # High Card Duel terminal state has -1.
            self._current_player = 0  # Default to P0 for terminal obs
        else:
            self._current_player = next_state.current_player()

        # Get observation for the *next* player (or P0 if done)
        tensor = next_state.to_tensor(self._current_player)
        
        return (
            tensor.data,
            step_reward,
            done,
            False,  # truncated
            dict(info)
        )

    def legal_actions(self) -> List[int]:
        """Return a list of legal action IDs for the current player."""
        if self.game_env.state is None or self.game_env.state.is_terminal():
            return []
        return [m.id for m in self.game_env.state.legal_moves()]

    @property
    def current_player(self) -> int:
        """Return the ID of the current player."""
        return self._current_player

    @property
    def num_players(self) -> int:
        """Return the total number of players."""
        return self._num_players
