"""Profile batch vs sequential overlay generation (M22).

This script measures the performance difference between sequential and batched
overlay generation for gradient-based XAI methods.

Usage:
    python scripts/profile_xai_batch.py [--batch-size BATCH_SIZE] [--num-overlays NUM]
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from ungar.xai_methods import PolicyGradOverlayMethod, ValueGradOverlayMethod


class SimplePolicy(nn.Module):
    """Simple policy network for profiling."""

    def __init__(self, input_dim: int = 168, output_dim: int = 52):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCritic(nn.Module):
    """Simple critic network for profiling."""

    def __init__(self, input_dim: int = 168):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_fake_observations(num_obs: int, input_dim: int = 168) -> list[np.ndarray]:
    """Generate fake observations for profiling."""
    np.random.seed(42)
    return [np.random.rand(input_dim).astype(np.float32) for _ in range(num_obs)]


def profile_sequential(method, observations: list[np.ndarray]) -> tuple[float, int]:
    """Profile sequential overlay generation."""
    start_time = time.perf_counter()

    overlays = []
    for i, obs in enumerate(observations):
        overlay = method.compute(obs=obs, action=i % 10, step=i, run_id="profile", meta=None)
        overlays.append(overlay)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    return elapsed, len(overlays)


def profile_batched(method, observations: list[np.ndarray], batch_size: int) -> tuple[float, int]:
    """Profile batched overlay generation."""
    # Create batch requests
    batch = [
        {"obs": obs, "action": i % 10, "step": i, "run_id": "profile", "meta": None}
        for i, obs in enumerate(observations)
    ]

    start_time = time.perf_counter()

    # Process in batches
    overlays = []
    for i in range(0, len(batch), batch_size):
        batch_chunk = batch[i : i + batch_size]
        chunk_overlays = method.compute_batch(batch_chunk)
        overlays.extend(chunk_overlays)

    end_time = time.perf_counter()
    elapsed = end_time - start_time

    return elapsed, len(overlays)


def main():
    parser = argparse.ArgumentParser(description="Profile batch vs sequential overlay generation")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for profiling (default: 8)"
    )
    parser.add_argument(
        "--num-overlays",
        type=int,
        default=100,
        help="Number of overlays to generate (default: 100)",
    )
    parser.add_argument(
        "--method",
        choices=["policy", "value", "both"],
        default="both",
        help="Which method to profile (default: both)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("UNGAR Batch Overlay Engine Profiling (M22)")
    print("=" * 60)
    print(f"Number of overlays: {args.num_overlays}")
    print(f"Batch size: {args.batch_size}")
    print(f"Method: {args.method}")
    print()

    # Generate fake observations (High Card Duel: 4×14×3 = 168)
    observations = generate_fake_observations(args.num_overlays, input_dim=168)

    results = []

    # Profile Policy Gradient
    if args.method in ["policy", "both"]:
        print("-" * 60)
        print("Policy Gradient Overlays")
        print("-" * 60)

        policy_model = SimplePolicy(input_dim=168, output_dim=52)
        policy_method = PolicyGradOverlayMethod(policy_model, "profiling")

        # Sequential
        seq_time, seq_count = profile_sequential(policy_method, observations)
        print(f"Sequential: {seq_time:.3f}s total, {seq_time/seq_count*1000:.2f}ms per overlay")

        # Batched
        batch_time, batch_count = profile_batched(policy_method, observations, args.batch_size)
        print(
            f"Batched ({args.batch_size}): {batch_time:.3f}s total, {batch_time/batch_count*1000:.2f}ms per overlay"
        )

        speedup = seq_time / batch_time if batch_time > 0 else 0
        print(f"Speedup: {speedup:.2f}×")
        print()

        results.append(("Policy Gradient", speedup))

    # Profile Value Gradient
    if args.method in ["value", "both"]:
        print("-" * 60)
        print("Value Gradient Overlays")
        print("-" * 60)

        critic_model = SimpleCritic(input_dim=168)
        value_method = ValueGradOverlayMethod(critic_model, "profiling", algo="ppo")

        # Sequential
        seq_time, seq_count = profile_sequential(value_method, observations)
        print(f"Sequential: {seq_time:.3f}s total, {seq_time/seq_count*1000:.2f}ms per overlay")

        # Batched
        batch_time, batch_count = profile_batched(value_method, observations, args.batch_size)
        print(
            f"Batched ({args.batch_size}): {batch_time:.3f}s total, {batch_time/batch_count*1000:.2f}ms per overlay"
        )

        speedup = seq_time / batch_time if batch_time > 0 else 0
        print(f"Speedup: {speedup:.2f}×")
        print()

        results.append(("Value Gradient", speedup))

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for method_name, speedup in results:
        print(f"{method_name}: {speedup:.2f}× faster with batching")

    if results:
        avg_speedup = sum(s for _, s in results) / len(results)
        print(f"\nAverage speedup: {avg_speedup:.2f}×")

    print("\nNote: Actual speedup depends on hardware, model size, and batch size.")
    print("GPU typically shows larger improvements than CPU.")


if __name__ == "__main__":
    main()
