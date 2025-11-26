"""RediAI adapter for UNGAR games.

This module provides the bridge between UNGAR's GameEnv and RediAI's EnvAdapter.
It is designed to be safe to import even if RediAI is not installed.
"""

from __future__ import annotations

from typing import Any, List, Protocol, Tuple

import numpy as np
from ungar.game import GameEnv, Move

# --- RediAI Availability Check & Protocol Stub ---

HAS_REDAI = True

try:
    from RediAI.env import EnvAdapter  # type: ignore[import-untyped]
except ImportError:
    HAS_REDAI = False

    class EnvAdapter(Protocol):  # type: ignore[no-redef]
        """Minimal protocol for RediAI EnvAdapter to allow type checking without RediAI."""

        async def reset(self) -> Any:
            """Reset the environment."""

        async def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
            """Execute a step in the environment."""

        def legal_moves(self) -> List[Any]:
            """Return legal moves."""

        def encode_state(self) -> Any:
            """Return state encoding."""


# --- Adapter Implementation ---


class RediAIUngarAdapter(EnvAdapter):
    """Wraps a UNGAR GameEnv to conform to RediAI's EnvAdapter interface."""

    def __init__(self, env: GameEnv):
        """Initialize the adapter.

        Args:
            env: The UNGAR environment instance to wrap.
        """
        self.env = env
        self._last_obs: Any = None

    async def reset(self) -> Any:
        """Reset the environment and return the initial observation."""
        self.env.reset()
        # RediAI expects state tensor + metadata.
        # For now, we'll return the tensor for the current player.
        return self.encode_state()

    async def step(self, action: Move) -> Tuple[Any, float, bool, dict]:
        """Apply an action to the environment.

        Args:
            action: The Move object to apply.

        Returns:
            A tuple of (observation, reward, done, info).
        """
        _, rewards, done, info = self.env.step(action)
        
        # RediAI typically expects a single scalar reward for the current agent.
        # UNGAR returns a tuple of rewards for all players.
        # We need to decide whose reward to return.
        # For a simple adapter, we assume we are controlling the current player *before* the move?
        # Or typically RediAI agents operate in a specific seat.
        # Given M05 is "minimal", we will return the reward for player 0 or the acting player.
        # However, step() returns rewards AFTER the move.
        # In a 2-player zero-sum game, if player 0 moved, they might get a reward now.
        
        # Let's sum the rewards for now or pick the first one as a simplification for M05,
        # acknowledging this might need refinement for multi-agent.
        # HighCardDuel returns (reward_p0, reward_p1).
        
        # We will assume single agent training perspective for now (Player 0).
        current_reward = rewards[0] if rewards else 0.0

        obs = self.encode_state()
        return obs, current_reward, done, dict(info)

    def legal_moves(self) -> List[Move]:
        """Return the list of legal moves for the current state."""
        if self.env.state is None:
            return []
        return list(self.env.state.legal_moves())

    def encode_state(self) -> Any:
        """Encode the current state into a tensor.

        Returns:
            The 4x14xn tensor for the current player.
        """
        if self.env.state is None:
            # Should not happen if reset() called
            # Return empty or raise? Raise is safer.
            raise RuntimeError("Environment state is None. Call reset() first.")
        
        # We default to observing from Player 0's perspective for this simple adapter
        # or the current player.
        # If it's a turn-based game, usually we want the observation for the player whose turn it is.
        player_id = self.env.state.current_player()
        if player_id == -1:
            # Game is over. Default to Player 0 for the terminal observation.
            player_id = 0
            
        tensor = self.env.state.to_tensor(player_id)
        
        # Convert UNGAR's bool tensor to float for ML frameworks if needed,
        # but RediAI contract says "returns 4x14xn tensor".
        # We will return the numpy array.
        return tensor.data.astype(np.float32)


def make_rediai_env(game_name: str) -> EnvAdapter:
    """Create a RediAI-compatible environment adapter for the named game.

    Args:
        game_name: Name of the game to create (e.g. "high_card_duel").

    Returns:
        An instance of RediAIUngarAdapter.

    Raises:
        RuntimeError: If RediAI is not installed.
        ValueError: If the game name is unknown.
    """
    if not HAS_REDAI:
        raise RuntimeError(
            "RediAI is not installed. Install with `pip install ungar-bridge[rediai]`."
        )

    from ungar.games.high_card_duel import make_high_card_duel_spec
    
    # Simple registry dispatch
    if game_name == "high_card_duel":
        spec = make_high_card_duel_spec()
        env = GameEnv(spec)
        return RediAIUngarAdapter(env)
    
    raise ValueError(f"Unknown game: {game_name}")

