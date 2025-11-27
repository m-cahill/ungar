"""Minimal training loop for UNGAR games."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping

from ungar.game import GameEnv
from ungar.games.high_card_duel import make_high_card_duel_spec

from .rl_adapter import UngarGymEnv
from .xai_overlays import make_high_card_overlay

RewardComponents = Mapping[str, float]


@dataclass
class TrainingResult:
    """Results from a training session."""

    rewards: List[float]
    episode_lengths: List[int]
    config: Dict[str, Any]
    components: List[RewardComponents]


def train_high_card_duel(
    num_episodes: int = 1000,
    seed: int | None = None,
    epsilon: float = 0.1,
    record_overlays: bool = False,
) -> TrainingResult:
    """Run a simple bandit-style training loop for High Card Duel.

    Args:
        num_episodes: Number of episodes to run.
        seed: Random seed for reproducibility.
        epsilon: Epsilon-greedy exploration rate.
        record_overlays: If True, generate an XAI overlay for the last episode.

    Returns:
        TrainingResult containing rewards and episode lengths.
    """
    # 1. Setup Environment
    spec = make_high_card_duel_spec()
    game_env = GameEnv(spec)
    env = UngarGymEnv(game_env)

    # 2. Setup Agent (Simple Q-table for single action '0')
    # High Card Duel only has one legal move (reveal) for each player.
    # So a Q-table is trivial (1 state -> 1 action).
    # But to demonstrate the loop, we'll pretend there's something to learn or track.
    # We will just track average reward.

    # 3. Setup PRNG
    # We use python's random for action selection and env seeding
    rng = random.Random(seed)

    rewards_history: List[float] = []
    lengths_history: List[int] = []
    components_history: List[RewardComponents] = []
    last_overlay: Any = None

    for i in range(num_episodes):
        episode_seed = seed + i if seed is not None else None
        env.reset(seed=episode_seed)

        episode_reward = 0.0
        steps = 0

        # Simple rollout loop
        while True:
            legal = env.legal_actions()
            if not legal:
                break

            # Epsilon-greedy (trivial here since len(legal)==1)
            if rng.random() < epsilon:
                action = rng.choice(legal)
            else:
                # Greedy: pick first (trivial)
                action = legal[0]

            _, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                # If this is the last episode and we are recording overlays
                if record_overlays and i == num_episodes - 1:
                    # We need the state tensor. The environment is terminal.
                    # UngarGymEnv doesn't expose the raw state tensor easily via public properties
                    # other than via step return (which is observation), but that might be
                    # P0's observation.
                    # We can access the underlying GameEnv state directly.
                    if env.game_env.state:
                        # We want to explain the state from the perspective of Player 0 usually.
                        # High Card Duel state tensor has 'my_hand' etc. relative to the requested player.
                        tensor = env.game_env.state.to_tensor(player=0)
                        last_overlay = make_high_card_overlay(tensor)
                break

        rewards_history.append(episode_reward)
        lengths_history.append(steps)

        # Decompose reward
        components_history.append(
            {
                "win_loss": episode_reward,
                "baseline": 0.0,
            }
        )

    config: Dict[str, Any] = {
        "num_episodes": num_episodes,
        "seed": seed,
        "epsilon": epsilon,
        "game": "high_card_duel",
    }

    if last_overlay:
        config["last_overlay"] = last_overlay

    return TrainingResult(
        rewards=rewards_history,
        episode_lengths=lengths_history,
        config=config,
        components=components_history,
    )
