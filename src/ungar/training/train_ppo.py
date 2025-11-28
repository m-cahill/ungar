"""PPO Training Runner.

Unified training loop for PPO agents.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from ungar.agents.adapters.gin_adapter import GinAdapter
from ungar.agents.adapters.high_card_adapter import HighCardAdapter
from ungar.agents.adapters.spades_adapter import SpadesAdapter
from ungar.agents.ppo_lite import PPOLiteAgent
from ungar.agents.unified_agent import Transition
from ungar.training.config import PPOConfig


@dataclass
class TrainingResult:
    rewards: List[float]
    metrics: Dict[str, float]


def get_adapter(game_name: str) -> HighCardAdapter | SpadesAdapter | GinAdapter:
    # Duplicated logic from train_dqn, consider moving to common utility if grows
    if game_name == "high_card_duel":
        return HighCardAdapter()
    elif game_name == "spades_mini":
        return SpadesAdapter()
    elif game_name == "gin_rummy":
        return GinAdapter()
    else:
        raise ValueError(f"Unknown game: {game_name}")


def train_ppo(
    game_name: str,
    config: PPOConfig | None = None,
    seed: int = 0,
) -> TrainingResult:
    """Train a PPO agent on the specified game."""
    if config is None:
        config = PPOConfig(
            algorithm="ppo",
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=64,
            total_episodes=50,
            max_steps_per_episode=1000,
            clip_coef=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            update_epochs=3,
            minibatch_size=16,
            gae_lambda=0.95,
            seed=seed,
        )

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    adapter = get_adapter(game_name)
    env = adapter.create_env()

    agent = PPOLiteAgent(
        input_dim=adapter.tensor_shape,
        action_space_size=adapter.action_space_size,
        config=config,
    )

    rewards_history = []

    for _ in range(config.total_episodes):
        # We don't set seed per episode here to allow variety, assuming global seed set
        # But environment might need reseeding if it's purely deterministic?
        # UNGAR envs use random module, which we seeded globally.
        state = env.reset()

        episode_reward = 0.0
        steps = 0
        done = False

        while not done and steps < config.max_steps_per_episode:
            current_player = state.current_player()
            obs_tensor = state.to_tensor(current_player)
            obs_flat = obs_tensor.data.flatten().astype(np.float32)

            legal_moves = state.legal_moves()
            if not legal_moves:
                break

            legal_indices = adapter.moves_to_indices(list(legal_moves))

            action_idx = agent.act(obs_flat, legal_indices)
            move = adapter.index_to_move(action_idx, list(legal_moves))

            next_state, rewards, done, _ = env.step(move)

            step_reward = 0.0
            if done:
                step_reward = rewards[current_player]
                episode_reward = step_reward

            # For PPO, we store transition now.
            # next_obs is needed only for value bootstrap if not done?
            # Our PPO Lite doesn't use next_obs for act/value in update loop except boundary.
            # We can pass zeros or real next obs.

            next_obs_flat = np.zeros_like(obs_flat)
            if not done:
                # Next player might be different, but we track "our" trajectory?
                # PPO assumes single agent perspective.
                # In self-play, next state IS the next observation for the agent (if it's their turn).
                # But in alternating turns, next state is opponent turn.
                # For this Lite implementation, we just store the sequence of states seen by ANY player.
                # Since we share the network, this is shared self-play training.
                next_player = next_state.current_player()
                next_tensor = next_state.to_tensor(next_player)
                next_obs_flat = next_tensor.data.flatten().astype(np.float32)

            transition = Transition(
                obs=obs_flat,
                action=action_idx,
                reward=step_reward,
                next_obs=next_obs_flat,
                done=done,
                legal_moves_next=[],  # PPO Lite doesn't strictly use this for masking next Q
            )

            agent.train_step(transition)
            state = next_state
            steps += 1

        # End of episode update
        agent.update()
        rewards_history.append(episode_reward)

    return TrainingResult(
        rewards=rewards_history,
        metrics={
            "avg_reward": sum(rewards_history) / len(rewards_history) if rewards_history else 0.0
        },
    )
