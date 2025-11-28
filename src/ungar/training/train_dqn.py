"""DQN Training Runner.

Unified training loop for any supported game using DQNLiteAgent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from ungar.agents.adapters.gin_adapter import GinAdapter
from ungar.agents.adapters.high_card_adapter import HighCardAdapter
from ungar.agents.adapters.spades_adapter import SpadesAdapter
from ungar.agents.dqn_lite import DQNLiteAgent
from ungar.agents.unified_agent import Transition


@dataclass
class TrainingResult:
    rewards: List[float]
    metrics: Dict[str, float]


def get_adapter(game_name: str) -> HighCardAdapter | SpadesAdapter | GinAdapter:
    if game_name == "high_card_duel":
        return HighCardAdapter()
    elif game_name == "spades_mini":
        return SpadesAdapter()
    elif game_name == "gin_rummy":
        return GinAdapter()
    else:
        raise ValueError(f"Unknown game: {game_name}")


def train_dqn(
    game_name: str,
    episodes: int = 200,
    seed: int | None = None,
    lr: float = 0.0005,
    buffer_size: int = 5000,
) -> TrainingResult:
    """Train a DQN agent on the specified game."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    adapter = get_adapter(game_name)
    env = adapter.create_env()

    # Initialize Agent
    agent = DQNLiteAgent(
        input_dim=adapter.tensor_shape,
        action_space_size=adapter.action_space_size,
        lr=lr,
        buffer_size=buffer_size,
        seed=seed,
    )

    rewards_history = []

    for i in range(episodes):
        ep_seed = seed + i if seed is not None else None
        state = env.reset(seed=ep_seed)

        # We only control Player 0 for simplicity in this loop
        # Opponent plays randomly or we self-play?
        # For M12 MVP, let's assume we control current_player, but share the buffer.
        # This effectively trains a self-play agent.

        episode_reward = 0.0
        steps = 0
        done = False

        while not done and steps < 1000:  # Safety break
            current_player = state.current_player()
            obs_tensor = state.to_tensor(current_player)
            # Flatten the entire (4, 14, n) tensor
            obs_flat = obs_tensor.data.flatten().astype(np.float32)

            legal_moves = state.legal_moves()
            if not legal_moves:
                break

            legal_indices = adapter.moves_to_indices(list(legal_moves))

            action_idx = agent.act(obs_flat, legal_indices)
            move = adapter.index_to_move(action_idx, list(legal_moves))

            next_state, rewards, done, _ = env.step(move)

            # Compute reward for THIS step (if any) or final
            # In turn-based zero-sum, immediate reward is usually 0 until end.
            # But we need to store transition for the player who acted.
            # The next state is for the NEXT player.
            # Standard RL needs (s, a, r, s').
            # If s' is opponent's turn, it's tricky.
            # For MVP DQN-Lite test:
            # We store (s, a, r, s') where s' is the state presented to the SAME player next time?
            # Or just next state raw.
            # Let's store raw next state for now.

            # Get reward for player who acted
            step_reward = 0.0
            if done:
                step_reward = rewards[current_player]
                episode_reward = step_reward  # Track final outcome for P0 usually

            # Next observation (from perspective of next player, which is standard self-play)
            next_player = next_state.current_player() if not done else 0
            # If done, next_state might be terminal, to_tensor might fail or return empty?
            # UNGAR states usually support to_tensor even if terminal.

            next_obs_flat = np.zeros_like(obs_flat)
            legal_next_indices = []

            if not done:
                next_tensor = next_state.to_tensor(next_player)
                next_obs_flat = next_tensor.data.flatten().astype(np.float32)
                legal_next = next_state.legal_moves()
                legal_next_indices = adapter.moves_to_indices(list(legal_next))

            transition = Transition(
                obs=obs_flat,
                action=action_idx,
                reward=step_reward,
                next_obs=next_obs_flat,
                done=done,
                legal_moves_next=legal_next_indices,
            )

            agent.train_step(transition)

            state = next_state
            steps += 1

        rewards_history.append(episode_reward)

    return TrainingResult(
        rewards=rewards_history,
        metrics={
            "avg_reward": sum(rewards_history) / len(rewards_history) if rewards_history else 0.0
        },
    )
