"""Tests for gradient-based XAI helpers."""

import numpy as np
import torch
import torch.nn as nn
from ungar.enums import RANK_COUNT, SUIT_COUNT
from ungar.xai_grad import compute_policy_grad_importance, compute_value_grad_importance


class ToyModel(nn.Module):
    """Simple linear model for testing gradients."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_compute_policy_grad_shape() -> None:
    """Test that gradient helper returns correct shape."""
    # 4x14x1 = 56 inputs
    # 2 actions
    model = ToyModel(56, 2)
    obs = torch.zeros(56)

    importance = compute_policy_grad_importance(model, obs, action_index=0)

    assert importance.shape == (SUIT_COUNT, RANK_COUNT)
    assert np.all(importance >= 0)  # Magnitudes are positive


def test_compute_policy_grad_logic() -> None:
    """Test that gradients flow correctly from weights."""
    # 4x14x1 = 56 inputs
    model = ToyModel(56, 1)

    # Set weights manually:
    # W[0] = 10 (corresponds to (0,0) in 4x14 grid)
    # W[1] = 0
    # ...
    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.weight[0, 0] = 10.0  # High sensitivity for first pixel
        model.linear.bias.fill_(0.0)

    obs = torch.ones(56)  # Input doesn't strictly matter for linear gradient, but non-zero is safe

    importance = compute_policy_grad_importance(model, obs, action_index=0)

    # Expect (0,0) to have 1.0 (after normalization) and others 0.0
    assert np.isclose(importance[0, 0], 1.0)
    assert np.isclose(importance.sum(), 1.0)
    assert importance[0, 1] == 0.0


def test_compute_policy_grad_multi_plane() -> None:
    """Test reduction across planes."""
    # 4x14x2 = 112 inputs
    model = ToyModel(112, 1)

    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        # Weight for card (0,0) on plane 0
        model.linear.weight[0, 0] = 2.0
        # Weight for card (0,0) on plane 1
        # If shape is (4, 14, N=2), then flat index 1 is (0,0,1)
        model.linear.weight[0, 1] = 3.0

    obs = torch.randn(112)
    importance = compute_policy_grad_importance(model, obs, action_index=0, normalize=False)

    # Raw gradient sum should be |2| + |3| = 5
    assert np.isclose(importance[0, 0], 5.0)


# ============================================================================
# Value Gradient Tests (M21-A)
# ============================================================================


class ToyCritic(nn.Module):
    """Simple linear critic model for testing value gradients."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single value output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """PPO-style get_value method."""
        return self.linear(x)


def test_compute_value_grad_shape() -> None:
    """Test that value gradient helper returns correct shape."""
    # 4x14x1 = 56 inputs -> scalar value output
    model = ToyCritic(56)
    obs = torch.zeros(56)

    importance = compute_value_grad_importance(model, obs)

    assert importance.shape == (SUIT_COUNT, RANK_COUNT)
    assert np.all(importance >= 0)  # Magnitudes are positive


def test_compute_value_grad_normalization() -> None:
    """Test that value gradients are properly normalized."""
    model = ToyCritic(56)

    # Set non-zero weights to ensure gradients exist
    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.weight[0, 0] = 5.0
        model.linear.weight[0, 10] = 3.0
        model.linear.bias.fill_(0.0)

    obs = torch.ones(56)
    importance = compute_value_grad_importance(model, obs, normalize=True)

    # Should sum to 1.0 (L1 normalization)
    assert np.isclose(importance.sum(), 1.0, atol=1e-6)


def test_compute_value_grad_logic() -> None:
    """Test that value gradients flow correctly from critic weights."""
    model = ToyCritic(56)

    # Set weights manually: high weight for first input
    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.weight[0, 0] = 10.0  # High sensitivity for position (0,0)
        model.linear.bias.fill_(0.0)

    obs = torch.ones(56)
    importance = compute_value_grad_importance(model, obs)

    # Expect (0,0) to have 1.0 (after normalization) and others 0.0
    assert np.isclose(importance[0, 0], 1.0, atol=1e-6)
    assert np.isclose(importance.sum(), 1.0, atol=1e-6)
    assert np.isclose(importance[0, 1], 0.0, atol=1e-9)


def test_compute_value_grad_multi_plane() -> None:
    """Test reduction across multiple feature planes."""
    # 4x14x2 = 112 inputs
    model = ToyCritic(112)

    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        # Weight for card (0,0) on plane 0 (index 0)
        model.linear.weight[0, 0] = 2.0
        # Weight for card (0,0) on plane 1 (index 1 in flat representation)
        model.linear.weight[0, 1] = 3.0
        model.linear.bias.fill_(0.0)

    obs = torch.randn(112)
    importance = compute_value_grad_importance(model, obs, normalize=False)

    # Raw gradient sum should be |2| + |3| = 5 for position (0,0)
    assert np.isclose(importance[0, 0], 5.0, atol=1e-5)


def test_compute_value_grad_with_get_value() -> None:
    """Test that value gradient works with get_value() method (PPO-style)."""
    model = ToyCritic(56)

    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.weight[0, 5] = 7.0  # Arbitrary position
        model.linear.bias.fill_(0.0)

    obs = torch.ones(56)
    importance = compute_value_grad_importance(model, obs)

    # Should have all weight at position corresponding to index 5
    # Index 5 in flat (4, 14, 1) is (0, 5, 0) -> card position (0, 5)
    assert np.isclose(importance[0, 5], 1.0, atol=1e-6)
    assert np.isclose(importance.sum(), 1.0, atol=1e-6)
