"""Tests for XAI overlay methods (M21-B)."""

import numpy as np
import torch
import torch.nn as nn
from ungar.xai_methods import ValueGradOverlayMethod


class ToyCritic(nn.Module):
    """Simple critic for testing."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """PPO-style get_value method."""
        return self.linear(x)


def test_value_grad_overlay_method_shape() -> None:
    """Test that ValueGradOverlayMethod produces correct overlay shape."""
    model = ToyCritic(56)  # 4x14x1
    method = ValueGradOverlayMethod(model, game_name="high_card_duel", algo="ppo")

    obs = np.zeros(56, dtype=np.float32)
    overlay = method.compute(obs=obs, action=0, step=10, run_id="test_run")

    assert overlay.label == "value_grad"
    assert overlay.importance.shape == (4, 14)
    assert overlay.step == 10
    assert overlay.run_id == "test_run"


def test_value_grad_overlay_method_metadata() -> None:
    """Test that ValueGradOverlayMethod sets correct metadata."""
    model = ToyCritic(56)
    method = ValueGradOverlayMethod(model, game_name="high_card_duel", algo="ppo")

    obs = np.ones(56, dtype=np.float32)
    overlay = method.compute(obs=obs, action=0, step=5, run_id="meta_test")

    assert overlay.meta["method"] == "value_grad"
    assert overlay.meta["target_type"] == "state_value"
    assert overlay.meta["algo"] == "ppo"
    assert overlay.meta["game"] == "high_card_duel"


def test_value_grad_overlay_normalized() -> None:
    """Test that importance values are normalized."""
    model = ToyCritic(56)

    # Set non-zero weights to ensure gradients
    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.weight[0, 0] = 5.0
        model.linear.weight[0, 10] = 3.0
        model.linear.bias.fill_(0.0)

    method = ValueGradOverlayMethod(model, game_name="test_game", algo="ppo")
    obs = np.ones(56, dtype=np.float32)
    overlay = method.compute(obs=obs, action=0, step=1, run_id="norm_test")

    # Should sum to 1.0 (L1 normalization)
    assert np.isclose(overlay.importance.sum(), 1.0, atol=1e-6)


def test_value_grad_overlay_ignores_action() -> None:
    """Test that action parameter is ignored for value gradients."""
    model = ToyCritic(56)

    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.weight[0, 5] = 10.0
        model.linear.bias.fill_(0.0)

    method = ValueGradOverlayMethod(model, game_name="test_game", algo="ppo")
    obs = np.ones(56, dtype=np.float32)

    # Compute with different action indices - should produce identical results
    overlay1 = method.compute(obs=obs, action=0, step=1, run_id="test")
    overlay2 = method.compute(obs=obs, action=5, step=1, run_id="test")

    np.testing.assert_array_almost_equal(overlay1.importance, overlay2.importance)


def test_value_grad_overlay_custom_meta() -> None:
    """Test that custom metadata is merged correctly."""
    model = ToyCritic(56)
    method = ValueGradOverlayMethod(model, game_name="gin_rummy", algo="ppo")

    obs = np.ones(56, dtype=np.float32)
    custom_meta = {"episode": 42, "player": 1}

    overlay = method.compute(obs=obs, action=0, step=10, run_id="custom_test", meta=custom_meta)

    # Custom meta should be preserved
    assert overlay.meta["episode"] == 42
    assert overlay.meta["player"] == 1
    # Standard meta should still be present
    assert overlay.meta["method"] == "value_grad"
    assert overlay.meta["target_type"] == "state_value"
    assert overlay.meta["algo"] == "ppo"
    assert overlay.meta["game"] == "gin_rummy"
