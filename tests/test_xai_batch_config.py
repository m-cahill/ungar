"""Tests for XAIConfig batch_size validation (M22-C)."""

import pytest
from ungar.training.config import XAIConfig


def test_batch_size_none_default() -> None:
    """Test that batch_size defaults to None (sequential mode)."""
    config = XAIConfig()
    assert config.batch_size is None


def test_batch_size_valid_values() -> None:
    """Test that valid batch_size values are accepted."""
    for size in [1, 2, 4, 8, 16, 32]:
        config = XAIConfig(batch_size=size)
        assert config.batch_size == size


def test_batch_size_validation_too_small() -> None:
    """Test that batch_size < 1 raises ValueError."""
    with pytest.raises(ValueError, match="must be between 1 and 32"):
        XAIConfig(batch_size=0)

    with pytest.raises(ValueError, match="must be between 1 and 32"):
        XAIConfig(batch_size=-1)


def test_batch_size_validation_too_large() -> None:
    """Test that batch_size > 32 raises ValueError."""
    with pytest.raises(ValueError, match="must be between 1 and 32"):
        XAIConfig(batch_size=33)

    with pytest.raises(ValueError, match="must be between 1 and 32"):
        XAIConfig(batch_size=100)


def test_batch_size_boundary_values() -> None:
    """Test boundary values (1 and 32) are accepted."""
    config_min = XAIConfig(batch_size=1)
    assert config_min.batch_size == 1

    config_max = XAIConfig(batch_size=32)
    assert config_max.batch_size == 32


def test_xai_config_with_other_fields() -> None:
    """Test that batch_size works alongside other XAIConfig fields."""
    config = XAIConfig(
        enabled=True,
        methods=["policy_grad", "value_grad"],
        every_n_episodes=5,
        max_overlays_per_run=100,
        batch_size=4,
    )

    assert config.enabled is True
    assert config.methods == ["policy_grad", "value_grad"]
    assert config.every_n_episodes == 5
    assert config.max_overlays_per_run == 100
    assert config.batch_size == 4
