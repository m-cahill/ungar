"""Tests for batch gradient overlay computation (M22-B)."""

import numpy as np
import torch
import torch.nn as nn
from ungar.xai_methods import PolicyGradOverlayMethod, ValueGradOverlayMethod


class TinyPolicyNet(nn.Module):
    """Minimal policy network for testing."""

    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TinyCritic(nn.Module):
    """Minimal critic network for testing."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_policy_grad_batch_matches_sequential() -> None:
    """Test that batch policy gradients match sequential computation (M22)."""
    model = TinyPolicyNet(56, 5)

    # Set deterministic weights
    with torch.no_grad():
        model.linear.weight.fill_(0.1)
        model.linear.bias.fill_(0.0)

    method = PolicyGradOverlayMethod(model, game_name="test_game")

    # Create 3 observations
    obs1 = np.random.rand(56).astype(np.float32)
    obs2 = np.random.rand(56).astype(np.float32)
    obs3 = np.random.rand(56).astype(np.float32)

    # Sequential computation
    overlay1 = method.compute(obs=obs1, action=0, step=1, run_id="test")
    overlay2 = method.compute(obs=obs2, action=1, step=2, run_id="test")
    overlay3 = method.compute(obs=obs3, action=2, step=3, run_id="test")

    # Batch computation
    batch = [
        {"obs": obs1, "action": 0, "step": 1, "run_id": "test"},
        {"obs": obs2, "action": 1, "step": 2, "run_id": "test"},
        {"obs": obs3, "action": 2, "step": 3, "run_id": "test"},
    ]
    batch_overlays = method.compute_batch(batch)

    # Compare results (should be very close, allowing for floating point errors)
    assert np.allclose(overlay1.importance, batch_overlays[0].importance, atol=1e-6)
    assert np.allclose(overlay2.importance, batch_overlays[1].importance, atol=1e-6)
    assert np.allclose(overlay3.importance, batch_overlays[2].importance, atol=1e-6)


def test_value_grad_batch_matches_sequential() -> None:
    """Test that batch value gradients match sequential computation (M22)."""
    model = TinyCritic(56)

    # Set deterministic weights
    with torch.no_grad():
        model.linear.weight.fill_(0.1)
        model.linear.bias.fill_(0.0)

    method = ValueGradOverlayMethod(model, game_name="test_game", algo="ppo")

    # Create 3 observations
    obs1 = np.random.rand(56).astype(np.float32)
    obs2 = np.random.rand(56).astype(np.float32)
    obs3 = np.random.rand(56).astype(np.float32)

    # Sequential computation
    overlay1 = method.compute(obs=obs1, action=0, step=1, run_id="test")
    overlay2 = method.compute(obs=obs2, action=0, step=2, run_id="test")
    overlay3 = method.compute(obs=obs3, action=0, step=3, run_id="test")

    # Batch computation
    batch = [
        {"obs": obs1, "action": 0, "step": 1, "run_id": "test"},
        {"obs": obs2, "action": 0, "step": 2, "run_id": "test"},
        {"obs": obs3, "action": 0, "step": 3, "run_id": "test"},
    ]
    batch_overlays = method.compute_batch(batch)

    # Compare results
    assert np.allclose(overlay1.importance, batch_overlays[0].importance, atol=1e-6)
    assert np.allclose(overlay2.importance, batch_overlays[1].importance, atol=1e-6)
    assert np.allclose(overlay3.importance, batch_overlays[2].importance, atol=1e-6)


def test_batch_gradient_shapes() -> None:
    """Test that batch gradients produce correct shapes."""
    model = TinyPolicyNet(56, 5)
    method = PolicyGradOverlayMethod(model, game_name="test_game")

    batch = [
        {"obs": np.random.rand(56).astype(np.float32), "action": i, "step": i, "run_id": "test"}
        for i in range(5)
    ]

    overlays = method.compute_batch(batch)

    assert len(overlays) == 5
    for overlay in overlays:
        assert overlay.importance.shape == (4, 14)
        # Should be normalized (L1 norm = 1.0 or all zeros)
        total = overlay.importance.sum()
        assert total == 0.0 or np.isclose(total, 1.0, atol=1e-6)


def test_batch_gradient_meta_preserved() -> None:
    """Test that metadata is correctly set in batch gradient overlays."""
    model = TinyCritic(56)
    method = ValueGradOverlayMethod(model, game_name="gin_rummy", algo="ppo")

    custom_meta = {"episode": 10, "player": 0}
    batch = [
        {
            "obs": np.random.rand(56).astype(np.float32),
            "action": 0,
            "step": 5,
            "run_id": "run_123",
            "meta": custom_meta,
        },
    ]

    overlays = method.compute_batch(batch)

    assert overlays[0].meta["game"] == "gin_rummy"
    assert overlays[0].meta["method"] == "value_grad"
    assert overlays[0].meta["target_type"] == "state_value"
    assert overlays[0].meta["algo"] == "ppo"
    assert overlays[0].meta["episode"] == 10
    assert overlays[0].meta["player"] == 0


def test_different_batch_sizes() -> None:
    """Test that different batch sizes all work correctly."""
    model = TinyPolicyNet(56, 3)
    method = PolicyGradOverlayMethod(model, game_name="test_game")

    for batch_size in [1, 2, 5, 10]:
        batch = [
            {"obs": np.random.rand(56).astype(np.float32), "action": 0, "step": i, "run_id": "test"}
            for i in range(batch_size)
        ]

        overlays = method.compute_batch(batch)

        assert len(overlays) == batch_size
        for i, overlay in enumerate(overlays):
            assert overlay.step == i
            assert overlay.importance.shape == (4, 14)


def test_batch_gradient_consistency() -> None:
    """Test that repeated batch calls give consistent results with same inputs."""
    model = TinyPolicyNet(56, 5)

    # Set deterministic weights and seed
    with torch.no_grad():
        model.linear.weight.normal_(0, 0.1)
        model.linear.bias.fill_(0.0)

    method = PolicyGradOverlayMethod(model, game_name="test_game")

    # Fixed observations
    obs = np.random.rand(56).astype(np.float32)
    batch = [
        {"obs": obs, "action": 0, "step": 1, "run_id": "test"},
    ]

    # Compute twice
    overlays1 = method.compute_batch(batch)
    overlays2 = method.compute_batch(batch)

    # Should be identical
    assert np.allclose(overlays1[0].importance, overlays2[0].importance, atol=1e-10)
