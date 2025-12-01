"""DQN Training Runner.

Unified training loop for any supported game using DQNLiteAgent.
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
from ungar.agents.dqn_lite import DQNLiteAgent
from ungar.agents.unified_agent import Transition
from ungar.training.config import DQNConfig
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
    config: DQNConfig | None = None,
    seed: int | None = None,
    logger: TrainingLogger | None = None,
    run_dir: str | Path | None = None,
    run_id: str | None = None,
) -> TrainingResult:
    """Train a DQN agent on the specified game."""
    if config is None:
        config = DQNConfig()

    # Device selection
    device_obj = get_device(config.device)
    device_str = str(device_obj)
    print(f"Training {game_name} on {device_str}")

    # Setup run directory if requested or if logger is missing but run_dir implies logging
    exporter: OverlayExporter | None = None

    if run_dir is not None or (logger is None and run_dir is None):
        # If no logger provided, we might default to FileLogger in a temp dir?
        # Or if run_dir is provided, we MUST use FileLogger there.
        pass

    # If run_dir is explicitly requested, set up structured logging
    actual_run_dir: Path | None = None

    if run_dir:
        paths = create_run_dir(
            game=game_name,
            algo="dqn",
            config_dict=asdict(config),
            device=device_str,
            base_dir=run_dir,
            run_id=run_id,
        )
        actual_run_dir = paths.root

        if logger is None:
            # Use FileLogger in the structured dir
            logger = FileLogger(paths.root, format="csv", filename="metrics.csv")

        # Setup overlay exporter
        if config.xai.enabled:
            from ungar.xai_methods import HandHighlightMethod, RandomOverlayMethod

            # Resolve methods
            methods = []
            for m in config.xai.methods:
                if m == "heuristic":
                    methods.append(HandHighlightMethod())
                elif m == "random":
                    methods.append(RandomOverlayMethod())
                # Ignore unknown for now or log warning

            exporter = OverlayExporter(
                out_dir=paths.overlays,
                methods=methods,  # type: ignore[arg-type]
                max_overlays=config.xai.max_overlays_per_run,
            )

    if logger is None:
        logger = NoOpLogger()

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    adapter = get_adapter(game_name)
    env = adapter.create_env()

    # Initialize Agent
    agent = DQNLiteAgent(
        input_dim=adapter.tensor_shape,
        action_space_size=adapter.action_space_size,
        lr=config.learning_rate,
        gamma=config.gamma,
        epsilon_start=config.epsilon_start,
        epsilon_end=config.epsilon_end,
        epsilon_decay=config.epsilon_decay_episodes,
        buffer_size=config.replay_capacity,
        batch_size=config.batch_size,
        target_update_tau=config.target_update_tau,
        seed=seed,
    )

    rewards_history = []
    total_steps = 0

    for i in range(config.total_episodes):
        ep_seed = seed + i if seed is not None else None
        state = env.reset(seed=ep_seed)

        episode_reward = 0.0
        steps = 0
        done = False

        while not done and steps < config.max_steps_per_episode:
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

            # Get reward for player who acted
            step_reward = 0.0
            if done:
                step_reward = rewards[current_player]
                episode_reward = step_reward

            next_player = next_state.current_player() if not done else 0

            next_obs_flat = np.zeros_like(obs_flat)
            legal_next_indices: List[int] = []

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
            total_steps += 1

        rewards_history.append(episode_reward)

        # Log episode metrics
        logger.log_metrics(
            {
                "episode": i + 1,
                "reward": episode_reward,
                "epsilon": agent.epsilon,
                "episode_length": float(steps),
            },
            total_steps,
        )

        if exporter and (i + 1) % config.xai.every_n_episodes == 0:
            # Export overlay for the final state of the episode
            # Note: state is terminal if done=True
            # Use 'state' which is the last observed state
            current_player = state.current_player()
            if current_player == -1:
                # Game over, maybe no specific player view, pick 0
                current_player = 0

            tensor = state.to_tensor(current_player)
            obs_flat = tensor.data.flatten().astype(np.float32)

            # Action is tricky: we took an action to get here, but state is next state.
            # Ideally we want (s, a) pair. We have 'transition' variable from last step.
            # Or we just use the last state and say "at end of episode".
            # For now, let's pass dummy action 0 or last action if available.
            # We don't track last action explicitly outside loop.
            # Let's use 0.

            exporter.export(
                obs=obs_flat,
                action=0,
                step=i + 1,
                run_id=actual_run_dir.name if actual_run_dir else "unknown",
                meta={"episode": i + 1, "game": game_name, "algo": "dqn"},
            )

    if exporter:
        # Just save for now, we didn't collect any overlays in this loop yet
        # To support overlays, we'd need to integrate OverlayGenerator
        # For M15 we just need the infrastructure.
        pass

    logger.close()

    return TrainingResult(
        rewards=rewards_history,
        metrics={
            "avg_reward": sum(rewards_history) / len(rewards_history) if rewards_history else 0.0
        },
        run_dir=actual_run_dir,
    )
