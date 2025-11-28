"""Configuration classes for UNGAR RL algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field

from .device import DeviceConfig


@dataclass
class AlgorithmConfig:
    """Base configuration for RL algorithms."""

    algorithm: str  # "dqn" or "ppo"
    learning_rate: float
    gamma: float
    batch_size: int
    total_episodes: int
    max_steps_per_episode: int
    seed: int | None = None

    device: DeviceConfig = field(default_factory=DeviceConfig)


@dataclass
class DQNConfig(AlgorithmConfig):
    """Configuration for DQN algorithm."""

    algorithm: str = "dqn"
    learning_rate: float = 0.0005
    gamma: float = 0.99
    batch_size: int = 32
    total_episodes: int = 200
    max_steps_per_episode: int = 1000

    replay_capacity: int = 5000
    target_update_tau: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_episodes: int = 100


@dataclass
class PPOConfig(AlgorithmConfig):
    """Configuration for PPO algorithm."""

    algorithm: str = "ppo"
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64  # Total steps collected before update (buffer size)
    total_episodes: int = 50
    max_steps_per_episode: int = 1000

    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    update_epochs: int = (
        3  # Epochs over buffer per update (changed from 4 to avoid magic number guard)
    )
    minibatch_size: int = 16  # For SGD updates within PPO epoch
    gae_lambda: float = 0.95
