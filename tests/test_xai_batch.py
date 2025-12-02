"""Tests for batch overlay engine (M22)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from ungar.enums import RANK_COUNT, SUIT_COUNT
from ungar.training.config import XAIConfig
from ungar.xai_methods import (
    HandHighlightMethod,
    PolicyGradOverlayMethod,
    RandomOverlayMethod,
    ValueGradOverlayMethod,
)


class ToyPolicyModel(nn.Module):
    """Simple policy network for testing."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ToyCritic(nn.Module):
    """Simple critic network for testing."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """PPO-style get_value method."""
        return self.linear(x)


# ============================================================================
# Config Validation Tests
# ============================================================================


def test_batch_size_validation_accepts_none() -> None:
    """Test that batch_size=None is accepted."""
    config = XAIConfig(batch_size=None)
    assert config.batch_size is None


def test_batch_size_validation_accepts_valid_range() -> None:
    """Test that batch_size in range 1-32 is accepted."""
    for size in [1, 4, 16, 32]:
        config = XAIConfig(batch_size=size)
        assert config.batch_size == size


def test_batch_size_validation_rejects_zero() -> None:
    """Test that batch_size=0 is rejected."""
    with pytest.raises(ValueError, match="must be between 1 and 32"):
        XAIConfig(batch_size=0)


def test_batch_size_validation_rejects_negative() -> None:
    """Test that negative batch_size is rejected."""
    with pytest.raises(ValueError, match="must be between 1 and 32"):
        XAIConfig(batch_size=-1)


def test_batch_size_validation_rejects_too_large() -> None:
    """Test that batch_size > 32 is rejected."""
    with pytest.raises(ValueError, match="must be between 1 and 32"):
        XAIConfig(batch_size=33)


# ============================================================================
# Protocol Fallback Tests
# ============================================================================


def test_heuristic_method_has_default_compute_batch() -> None:
    """Test that non-gradient methods have default compute_batch."""
    method = HandHighlightMethod()

    # Create batch of inputs
    batch = [
        {
            "obs": np.ones(56, dtype=np.float32),
            "action": 0,
            "step": i,
            "run_id": "test",
            "meta": None,
        }
        for i in range(3)
    ]

    # Should use default implementation (sequential)
    overlays = method.compute_batch(batch)

    assert len(overlays) == 3
    assert all(overlay.label == "heuristic" for overlay in overlays)


def test_random_method_has_default_compute_batch() -> None:
    """Test that random method uses default fallback."""
    method = RandomOverlayMethod()

    batch = [
        {
            "obs": np.ones(56, dtype=np.float32),
            "action": 0,
            "step": i,
            "run_id": "test",
            "meta": None,
        }
        for i in range(2)
    ]

    overlays = method.compute_batch(batch)

    assert len(overlays) == 2
    assert all(overlay.label == "random" for overlay in overlays)
    # Random values should differ between overlays
    assert not np.array_equal(overlays[0].importance, overlays[1].importance)


# ============================================================================
# Batch vs Sequential Equivalence Tests (Policy Gradient)
# ============================================================================


def test_policy_grad_batch_matches_sequential() -> None:
    """Test that batched policy gradient matches sequential computation."""
    model = ToyPolicyModel(56, 3)

    # Set deterministic weights
    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.weight[0, 0] = 5.0
        model.linear.weight[0, 10] = 3.0
        model.linear.bias.fill_(0.0)

    method = PolicyGradOverlayMethod(model, "test_game")

    # Create test observations
    np.random.seed(42)
    batch = [
        {
            "obs": np.random.rand(56).astype(np.float32),
            "action": i % 3,
            "step": i,
            "run_id": "test",
            "meta": None,
        }
        for i in range(4)
    ]

    # Compute sequentially
    sequential_overlays = [
        method.compute(
            obs=item["obs"],  # type: ignore[arg-type]
            action=item["action"],  # type: ignore[arg-type]
            step=item["step"],  # type: ignore[arg-type]
            run_id=item["run_id"],  # type: ignore[arg-type]
            meta=item["meta"],
        )
        for item in batch
    ]

    # Compute in batch
    batch_overlays = method.compute_batch(batch)

    # Compare results
    assert len(batch_overlays) == len(sequential_overlays)

    for batch_overlay, seq_overlay in zip(batch_overlays, sequential_overlays):
        # Importance should be very close (within numerical tolerance)
        assert np.allclose(
            batch_overlay.importance, seq_overlay.importance, atol=1e-6
        ), "Batch and sequential results differ"

        # Metadata should match
        assert batch_overlay.label == seq_overlay.label
        assert batch_overlay.step == seq_overlay.step
        assert batch_overlay.run_id == seq_overlay.run_id


# ============================================================================
# Batch vs Sequential Equivalence Tests (Value Gradient)
# ============================================================================


def test_value_grad_batch_matches_sequential() -> None:
    """Test that batched value gradient matches sequential computation."""
    model = ToyCritic(56)

    # Set deterministic weights
    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.weight[0, 5] = 7.0
        model.linear.weight[0, 20] = 4.0
        model.linear.bias.fill_(0.0)

    method = ValueGradOverlayMethod(model, "test_game", algo="ppo")

    # Create test observations
    np.random.seed(123)
    batch = [
        {
            "obs": np.random.rand(56).astype(np.float32),
            "action": 0,  # Ignored for value grad
            "step": i,
            "run_id": "test",
            "meta": None,
        }
        for i in range(3)
    ]

    # Compute sequentially
    sequential_overlays = [
        method.compute(
            obs=item["obs"],  # type: ignore[arg-type]
            action=item["action"],  # type: ignore[arg-type]
            step=item["step"],  # type: ignore[arg-type]
            run_id=item["run_id"],  # type: ignore[arg-type]
            meta=item["meta"],
        )
        for item in batch
    ]

    # Compute in batch
    batch_overlays = method.compute_batch(batch)

    # Compare results
    assert len(batch_overlays) == len(sequential_overlays)

    for batch_overlay, seq_overlay in zip(batch_overlays, sequential_overlays):
        # Importance should match (within tolerance)
        assert np.allclose(
            batch_overlay.importance, seq_overlay.importance, atol=1e-6
        ), "Value grad batch and sequential results differ"

        # Metadata should match
        assert batch_overlay.label == "value_grad"
        assert batch_overlay.meta["target_type"] == "state_value"
        assert batch_overlay.meta["algo"] == "ppo"


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_batch_size_one_matches_sequential() -> None:
    """Test that batch_size=1 produces same results as sequential."""
    model = ToyPolicyModel(56, 2)
    method = PolicyGradOverlayMethod(model, "test_game")

    obs = np.random.rand(56).astype(np.float32)
    action = 0
    step = 10
    run_id = "test"

    # Sequential
    seq_overlay = method.compute(obs=obs, action=action, step=step, run_id=run_id)

    # Batch of 1
    batch_overlays = method.compute_batch(
        [{"obs": obs, "action": action, "step": step, "run_id": run_id, "meta": None}]
    )

    assert len(batch_overlays) == 1
    assert np.allclose(batch_overlays[0].importance, seq_overlay.importance, atol=1e-6)


def test_empty_batch_returns_empty_list() -> None:
    """Test that compute_batch with empty input returns empty list."""
    model = ToyPolicyModel(56, 2)
    method = PolicyGradOverlayMethod(model, "test_game")

    overlays = method.compute_batch([])

    assert overlays == []


def test_batch_preserves_per_item_metadata() -> None:
    """Test that batch processing preserves individual metadata."""
    model = ToyPolicyModel(56, 2)
    method = PolicyGradOverlayMethod(model, "test_game")

    batch = [
        {
            "obs": np.random.rand(56).astype(np.float32),
            "action": i % 2,
            "step": i * 10,
            "run_id": f"run_{i}",
            "meta": {"episode": i},
        }
        for i in range(3)
    ]

    overlays = method.compute_batch(batch)

    for i, overlay in enumerate(overlays):
        assert overlay.step == i * 10
        assert overlay.run_id == f"run_{i}"
        assert overlay.meta["episode"] == i


# ============================================================================
# Normalization Tests
# ============================================================================


def test_batch_overlays_are_normalized() -> None:
    """Test that batch-computed overlays are L1-normalized."""
    model = ToyPolicyModel(56, 2)

    # Set non-zero weights to ensure gradients
    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.weight[0, 0] = 10.0
        model.linear.weight[0, 5] = 5.0
        model.linear.bias.fill_(0.0)

    method = PolicyGradOverlayMethod(model, "test_game")

    batch = [
        {
            "obs": np.ones(56, dtype=np.float32),
            "action": 0,
            "step": i,
            "run_id": "test",
            "meta": None,
        }
        for i in range(3)
    ]

    overlays = method.compute_batch(batch)

    for overlay in overlays:
        # Each overlay should sum to 1.0 (L1 normalization)
        assert np.isclose(overlay.importance.sum(), 1.0, atol=1e-6)


# ============================================================================
# Shape Tests
# ============================================================================


def test_batch_overlays_have_correct_shape() -> None:
    """Test that batch-computed overlays have correct 4x14 shape."""
    model = ToyPolicyModel(56, 2)
    method = PolicyGradOverlayMethod(model, "test_game")

    batch = [
        {
            "obs": np.random.rand(56).astype(np.float32),
            "action": 0,
            "step": i,
            "run_id": "test",
            "meta": None,
        }
        for i in range(5)
    ]

    overlays = method.compute_batch(batch)

    for overlay in overlays:
        assert overlay.importance.shape == (SUIT_COUNT, RANK_COUNT)
        assert overlay.importance.shape == (4, 14)
