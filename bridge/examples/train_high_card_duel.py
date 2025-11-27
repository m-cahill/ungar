"""CLI entry point for High Card Duel local training."""

import argparse
from statistics import mean

from ungar_bridge.training import train_high_card_duel


def main() -> None:
    """Run training via CLI args."""
    parser = argparse.ArgumentParser(
        description="Run local training loop for High Card Duel"
    )
    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of episodes to run"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1, help="Exploration rate"
    )
    args = parser.parse_args()

    print(f"Starting training: {args.episodes} episodes, seed={args.seed}")
    
    result = train_high_card_duel(
        num_episodes=args.episodes,
        seed=args.seed,
        epsilon=args.epsilon,
    )

    avg_reward = mean(result.rewards)
    min_reward = min(result.rewards)
    max_reward = max(result.rewards)
    avg_len = mean(result.episode_lengths)

    print("-" * 30)
    print("Training Complete")
    print(f"Episodes: {len(result.rewards)}")
    print(f"Avg Reward: {avg_reward:.3f}")
    print(f"Min/Max Reward: {min_reward:.1f} / {max_reward:.1f}")
    print(f"Avg Length: {avg_len:.2f}")
    print("-" * 30)


if __name__ == "__main__":
    main()

