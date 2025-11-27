"""RediAI training adapter for Mini Spades.

Wraps the Mini Spades environment for RediAI training workflows.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, cast

from ungar.game import GameEnv
from ungar.games.spades_mini import make_spades_mini_spec
from ungar.xai import CardOverlay, zero_overlay

from .rediai_rewardlab import (
    build_reward_decomposition_payload,
    log_reward_decomposition,
)
from .rediai_training import TrainingResult, workflow_context
from .rediai_xai import log_overlays_as_artifact
from .rl_adapter import UngarGymEnv


def make_spades_overlay(state: Any) -> CardOverlay:
    """Generate a heuristic overlay for Mini Spades.

    Highlights cards in the current player's hand.
    """
    # For now, just reuse the logic similar to high card overlay but adapt if needed.
    # SpadesMiniState isn't easily accessible via public tensor unless we use to_tensor.
    # The training loop below will have access to the raw state.

    # We'll create a zero overlay and fill "my_hand" plane if available
    overlay = zero_overlay("spades_mini_importance", meta={"game": "spades_mini"})

    # Simple heuristic: highlight current hand
    # State is the raw SpadesMiniState
    player = state.current_player()
    hand = state.hands[player]

    from ungar.enums import RANK_COUNT

    importance = overlay.importance
    for card in hand:
        idx = card.to_index()
        suit_index, rank_index = divmod(idx, RANK_COUNT)
        importance[suit_index, rank_index] = 1.0

    return overlay


async def train_spades_mini_rediai(
    num_episodes: int = 200,
    seed: int | None = None,
    record_overlays: bool = False,
) -> TrainingResult:
    """Run Mini Spades training within a RediAI workflow context.

    Args:
        num_episodes: Number of episodes to train.
        seed: Random seed.
        record_overlays: Whether to generate and log XAI overlays.

    Returns:
        The training result.
    """
    async with workflow_context() as recorder:
        # Local training loop (embedded here for simplicity as per plan,
        # or we could extract to training.py like high_card)
        # Plan says: "Mirror train_high_card_duel_rediai"

        # 1. Setup
        spec = make_spades_mini_spec()
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

            # Track rewards for player 0
            ep_reward = 0.0
            steps = 0

            while True:
                legal = env.legal_actions()
                if not legal:
                    break

                # Random policy for now
                action = rng.choice(legal)

                _, reward, terminated, truncated, _ = env.step(action)

                # In 2-player zero sum, if I moved, I might get reward now if trick ended?
                # SpadesMini returns rewards at end of trick/game.
                # UngarGymEnv handles returning reward for player who just acted.
                # But we are tracking player 0.
                # If player 0 moved, reward is for player 0.
                # If player 1 moved, reward is for player 1.
                # We need to accumulate P0 reward.

                # Actually, UngarGymEnv.step returns reward for the player who acted.
                # We need to track P0 cumulative.
                # To do this rigorously in a multi-agent gym wrapper is tricky.
                # For this simple "train" loop which is really just "run random episodes",
                # let's just use the final game returns if terminal.

                steps += 1
                if terminated or truncated:
                    if game_env.state:
                        final_returns = game_env.state.returns()
                        ep_reward = final_returns[0]  # Player 0

                        if record_overlays and i == num_episodes - 1:
                            last_overlay = make_spades_overlay(game_env.state)
                    break

            rewards_history.append(ep_reward)
            lengths_history.append(steps)
            components_history.append({"game_score": ep_reward, "baseline": 0.0})

        # 2. Metrics
        avg_reward = sum(rewards_history) / len(rewards_history)
        await recorder.record_metric("ungar.spades.avg_reward", avg_reward, episodes=num_episodes)
        await recorder.record_metric("ungar.spades.last_reward", rewards_history[-1])

        # 3. Artifacts
        # Convert components_history to list of Mapping for type compatibility
        components_as_mapping: List[Mapping[str, float]] = [
            cast(Mapping[str, float], comp) for comp in components_history
        ]

        result = TrainingResult(
            rewards=rewards_history,
            episode_lengths=lengths_history,
            config={"game": "spades_mini", "episodes": num_episodes},
            components=components_as_mapping,
        )

        if record_overlays and last_overlay:
            # Attach to result to match high card pattern if needed, but we can log directly here
            await log_overlays_as_artifact(
                recorder, [last_overlay], artifact_name="ungar_spades_overlays.json"
            )

        if components_history:
            payload = build_reward_decomposition_payload(result, experiment_id="spades_mini_demo")
            await log_reward_decomposition(recorder, payload)

        return result
