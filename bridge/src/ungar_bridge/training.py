"""Minimal training loop for UNGAR games."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List

from ungar.game import GameEnv
from ungar.games.high_card_duel import make_high_card_duel_spec

from .rl_adapter import UngarGymEnv


@dataclass
class TrainingResult:
    """Results from a training session."""

    rewards: List[float]
    episode_lengths: List[int]
    config: Dict[str, float | int | str | None]


def train_high_card_duel(
    num_episodes: int = 1000,
    seed: int | None = None,
    epsilon: float = 0.1,
) -> TrainingResult:
    """Run a simple bandit-style training loop for High Card Duel.

    Args:
        num_episodes: Number of episodes to run.
        seed: Random seed for reproducibility.
        epsilon: Epsilon-greedy exploration rate.

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
                break
        
        rewards_history.append(episode_reward)
        lengths_history.append(steps)

    return TrainingResult(
        rewards=rewards_history,
        episode_lengths=lengths_history,
        config={
            "num_episodes": num_episodes,
            "seed": seed,
            "epsilon": epsilon,
            "game": "high_card_duel",
        },
    )

