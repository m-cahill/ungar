"""RediAI training adapter for Gin Rummy.

Wraps the Gin Rummy environment for RediAI training workflows.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, cast

from ungar.game import GameEnv
from ungar.games.gin_rummy import make_gin_rummy_spec
from ungar.xai import CardOverlay, zero_overlay

from .rediai_rewardlab import (
    build_reward_decomposition_payload,
    log_reward_decomposition,
)
from .rediai_training import TrainingResult, workflow_context
from .rediai_xai import log_overlays_as_artifact
from .rl_adapter import UngarGymEnv


def make_gin_overlay(state: Any) -> CardOverlay:
    """Generate a heuristic overlay for Gin Rummy.

    Highlights cards in hand and discard pile.
    """
    overlay = zero_overlay("gin_rummy_importance", meta={"game": "gin_rummy"})

    # Highlight current hand + discard top
    player = state.current_player()
    hand = state.hands[player]
    discard = state.discard_pile

    from ungar.enums import RANK_COUNT

    importance = overlay.importance

    # Hand = 1.0
    for card in hand:
        idx = card.to_index()
        suit_index, rank_index = divmod(idx, RANK_COUNT)
        importance[suit_index, rank_index] = 1.0

    # Discard = 0.5 (if present)
    if discard:
        top = discard[-1]
        idx = top.to_index()
        suit_index, rank_index = divmod(idx, RANK_COUNT)
        importance[suit_index, rank_index] = 0.5

    return overlay


async def train_gin_rummy_rediai(
    num_episodes: int = 100,
    seed: int | None = None,
    record_overlays: bool = False,
) -> TrainingResult:
    """Run Gin Rummy training within a RediAI workflow context.

    Args:
        num_episodes: Number of episodes to train.
        seed: Random seed.
        record_overlays: Whether to generate and log XAI overlays.

    Returns:
        The training result.
    """
    async with workflow_context() as recorder:
        spec = make_gin_rummy_spec()
        game_env = GameEnv(spec)
        env = UngarGymEnv(game_env)

        rng = random.Random(seed)

        rewards_history: List[float] = []
        lengths_history: List[int] = []
        components_history: List[Dict[str, float]] = []
        last_overlay: Any = None

        for i in range(num_episodes):
            ep_seed = seed + i if seed is not None else None
            env.reset(seed=ep_seed)

            ep_reward = 0.0
            steps = 0

            while steps < 1000:  # Safety break for random policy loops
                legal = env.legal_actions()
                if not legal:
                    break

                action = rng.choice(legal)
                _, reward, terminated, truncated, _ = env.step(action)

                steps += 1
                if terminated or truncated:
                    if game_env.state:
                        final_returns = game_env.state.returns()
                        ep_reward = final_returns[0]

                        if record_overlays and i == num_episodes - 1:
                            last_overlay = make_gin_overlay(game_env.state)
                    break
            else:
                # Force break if max steps reached
                # Capture overlay if this was the last episode
                if record_overlays and i == num_episodes - 1 and game_env.state:
                    last_overlay = make_gin_overlay(game_env.state)

            rewards_history.append(ep_reward)
            lengths_history.append(steps)
            components_history.append({"game_score": ep_reward, "baseline": 0.0})

        # Metrics
        avg_reward = sum(rewards_history) / len(rewards_history)
        await recorder.record_metric("ungar.gin.avg_reward", avg_reward, episodes=num_episodes)
        await recorder.record_metric("ungar.gin.last_reward", rewards_history[-1])

        # Artifacts
        components_as_mapping: List[Mapping[str, float]] = [
            cast(Mapping[str, float], comp) for comp in components_history
        ]

        result = TrainingResult(
            rewards=rewards_history,
            episode_lengths=lengths_history,
            config={"game": "gin_rummy", "episodes": num_episodes},
            components=components_as_mapping,
        )

        if record_overlays and last_overlay:
            await log_overlays_as_artifact(
                recorder, [last_overlay], artifact_name="ungar_gin_overlays.json"
            )

        if components_history:
            payload = build_reward_decomposition_payload(result, experiment_id="gin_rummy_demo")
            await log_reward_decomposition(recorder, payload)

        return result
