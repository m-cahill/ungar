"""RL-style adapter for UNGAR games (Gym-like interface)."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from ungar.game import GameEnv, Move

from .adapter_base import BridgeAdapter


class EnvLike(Protocol):
    """Protocol for Gym-like environments."""

    def reset(self, seed: int | None = None) -> Any:
        ...

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        ...


class RLAdapter(BridgeAdapter):
    """Adapts a UNGAR GameEnv to an RL-friendly interface."""

    def __init__(self, env: GameEnv) -> None:
        self.env = env

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with optional config."""
        pass

    def state_to_external(self, state: Any) -> dict[str, np.ndarray]:
        """Convert game state to observation dict (tensor per player)."""
        # For HighCardDuel, we might just return the current player's view
        # But a general RL adapter usually serves one agent perspective or returns a dict of all.
        # Here, we return a dict mapping player_id -> tensor
        # Note: This assumes 'state' is a GameState which has to_tensor
        # But 'state' passed here might be whatever UNGAR returns.
        # Ideally, this adapter wraps the whole interaction loop.
        raise NotImplementedError("Direct state conversion not used in Gym loop.")

    def external_to_move(self, external_action: Any) -> Move:
        """Convert external action (int/index) to Move."""
        # Assuming external_action is an index into legal_moves or a MoveId
        # For simplicity in this MVP, assume it's a Move object or we recreate it.
        if isinstance(external_action, Move):
            return external_action
        # If int, we need context to know which move it maps to.
        # This suggests the adapter needs to be stateful or the Env wrapper handles it.
        raise NotImplementedError("Stateless action conversion requires context.")


class UngarGymEnv:
    """A Gym-like wrapper around UNGAR GameEnv."""

    def __init__(self, game_env: GameEnv, player_id: int = 0) -> None:
        self.game_env = game_env
        self.player_id = player_id
        self._last_obs: dict[str, Any] = {}

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        state = self.game_env.reset(seed=seed)
        # Fast-forward to player's turn or end if necessary?
        # For HighCardDuel, P0 acts first.
        tensor = state.to_tensor(self.player_id)
        return tensor.data, {"legal_moves": state.legal_moves()}

    def step(self, action: Move) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Apply move
        # Note: UNGAR games are multi-agent. A simple Gym env usually controls one agent.
        # If it's P0's turn, apply action. If it's P1's turn, we need an opponent policy.
        # For M04 MVP, we'll just expose the raw step and assume external loop handles turns.
        state, rewards, done, info = self.game_env.step(action)

        tensor = state.to_tensor(self.player_id)
        reward = rewards[self.player_id] if done else 0.0
        truncated = False

        return (
            tensor.data,
            reward,
            done,
            truncated,
            {"legal_moves": state.legal_moves() if not done else []},
        )
