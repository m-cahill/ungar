"""PPO Training Runner.

Unified training loop for PPO agents.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ungar.agents.adapters.gin_adapter import GinAdapter
from ungar.agents.adapters.high_card_adapter import HighCardAdapter
from ungar.agents.adapters.spades_adapter import SpadesAdapter
from ungar.agents.ppo_lite import PPOLiteAgent
from ungar.agents.unified_agent import Transition
from ungar.training.config import PPOConfig
from ungar.training.device import get_device
from ungar.training.logger import FileLogger, NoOpLogger, TrainingLogger
from ungar.training.overlay_exporter import OverlayExporter
from ungar.training.run_dir import create_run_dir


@dataclass
class TrainingResult:
    rewards: List[float]
    metrics: Dict[str, float]
    run_dir: Optional[Path] = None


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
    logger: TrainingLogger | None = None,
    run_dir: str | Path | None = None,
    run_id: str | None = None,
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

    # Device selection
    device_obj = get_device(config.device)
    device_str = str(device_obj)
    print(f"Training {game_name} on {device_str}")

    # Setup run directory if requested
    exporter: OverlayExporter | None = None
    paths = None

    if run_dir:
        paths = create_run_dir(
            game=game_name,
            algo="ppo",
            config_dict=asdict(config),
            device=device_str,
            base_dir=run_dir,
            run_id=run_id,
        )
        if logger is None:
            logger = FileLogger(paths.root, format="csv", filename="metrics.csv")

    if logger is None:
        logger = NoOpLogger()

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

    # Setup XAI overlay exporter after agent is ready (M21-C)
    if config.xai.enabled and paths is not None:
        from ungar.xai_methods import (
            HandHighlightMethod,
            OverlayMethod,
            PolicyGradOverlayMethod,
            RandomOverlayMethod,
            ValueGradOverlayMethod,
        )

        methods: list[OverlayMethod] = []
        for m in config.xai.methods:
            if m == "heuristic":
                methods.append(HandHighlightMethod())
            elif m == "random":
                methods.append(RandomOverlayMethod())
            elif m == "policy_grad":
                # For PPO, pass the actor network (which outputs logits)
                methods.append(PolicyGradOverlayMethod(agent.actor, game_name))
            elif m == "value_grad":
                # For PPO, pass the full ActorCritic network (has get_value method)
                methods.append(ValueGradOverlayMethod(agent.actor, game_name, algo="ppo"))

        exporter = OverlayExporter(
            out_dir=paths.overlays,
            methods=methods,
            max_overlays=config.xai.max_overlays_per_run,
            batch_size=config.xai.batch_size,  # M22: Enable batch overlay generation
        )

    rewards_history = []
    total_steps = 0

    for i in range(config.total_episodes):
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

            next_obs_flat = np.zeros_like(obs_flat)
            if not done:
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
            total_steps += 1

        # End of episode update
        update_info = agent.update()
        rewards_history.append(episode_reward)

        # Log metrics
        metrics = {
            "episode": i + 1,
            "reward": episode_reward,
            "episode_length": float(steps),
            **update_info,
        }
        logger.log_metrics(metrics, total_steps)

        if exporter and (i + 1) % config.xai.every_n_episodes == 0:
            current_player = state.current_player()
            if current_player == -1:
                current_player = 0

            tensor = state.to_tensor(current_player)
            obs_flat = tensor.data.flatten().astype(np.float32)

            exporter.export(
                obs=obs_flat,
                action=0,
                step=i + 1,
                run_id=run_id or "unknown",
                meta={"episode": i + 1, "game": game_name, "algo": "ppo"},
            )

    if exporter:
        # M22: Flush any remaining buffered overlays
        exporter.flush()

    logger.close()

    return TrainingResult(
        rewards=rewards_history,
        metrics={
            "avg_reward": (sum(rewards_history) / len(rewards_history) if rewards_history else 0.0)
        },
        run_dir=paths.root if paths else None,
    )
