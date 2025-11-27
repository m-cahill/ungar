"""CLI demo for RediAI-backed training of High Card Duel.

This script runs the training loop wrapped in a RediAI workflow context.
If RediAI is not installed, it runs in local-dummy mode with a warning.
"""

import argparse
import asyncio
import sys

from ungar_bridge.rediai_training import is_rediai_available, train_high_card_duel_rediai


async def main() -> None:
    """Parse args and run training."""
    parser = argparse.ArgumentParser(description="Train High Card Duel with RediAI integration.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Exploration rate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    if not is_rediai_available():
        print(
            "WARNING: RediAI not installed; running in local-dummy mode (no registry logging).",
            file=sys.stderr,
        )
    else:
        print("RediAI detected; metrics will be logged to workflow registry.")

    print(f"Starting training: episodes={args.episodes}, epsilon={args.epsilon}, seed={args.seed}")

    result = await train_high_card_duel_rediai(
        num_episodes=args.episodes, epsilon=args.epsilon, seed=args.seed
    )

    if result.rewards:
        avg_reward = sum(result.rewards) / len(result.rewards)
        print(f"Training complete. Avg Reward: {avg_reward:.4f}, Last Reward: {result.rewards[-1]}")
    else:
        print("Training complete (no rewards collected).")


if __name__ == "__main__":
    asyncio.run(main())
