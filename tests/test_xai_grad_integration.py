"""Integration test for gradient-based XAI overlays."""

import json
from pathlib import Path

import pytest
from ungar.training.config import DQNConfig, PPOConfig, XAIConfig
from ungar.training.train_dqn import train_dqn
from ungar.training.train_ppo import train_ppo


def test_training_emits_grad_overlays(tmp_path: Path) -> None:
    """Run a minimal training session with policy_grad enabled and verify output."""
    run_dir = tmp_path / "runs"

    xai_config = XAIConfig(
        enabled=True,
        methods=["heuristic", "policy_grad"],
        every_n_episodes=1,
        max_overlays_per_run=10,
    )

    dqn_config = DQNConfig(total_episodes=2, xai=xai_config)

    result = train_dqn(
        game_name="high_card_duel",
        config=dqn_config,
        run_dir=run_dir,
        run_id="grad_xai_test",
    )

    assert result.run_dir is not None
    overlays_dir = result.run_dir / "overlays"
    assert overlays_dir.exists()

    # Expect overlays for both methods
    files = list(overlays_dir.glob("*.json"))
    # 2 episodes * 2 methods = 4 files
    assert len(files) == 4

    # Check for policy_grad specific meta
    grad_files = list(overlays_dir.glob("policy_grad_*.json"))
    assert len(grad_files) == 2

    with open(grad_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["label"] == "policy_grad"
        assert data["meta"]["method"] == "policy_grad"
        # DQN integration doesn't explicitly set target_type yet in train_dqn (only in method default),
        # or we might want to check it.
        # The method implementation puts "target_type": "logit_or_q"
        assert data["meta"]["target_type"] == "logit_or_q"


def test_ppo_emits_value_grad_overlays(tmp_path: Path) -> None:
    """Test PPO training with value_grad enabled (M21-C)."""
    run_dir = tmp_path / "runs"

    xai_config = XAIConfig(
        enabled=True, methods=["value_grad"], every_n_episodes=1, max_overlays_per_run=5
    )

    ppo_config = PPOConfig(total_episodes=2, batch_size=32, xai=xai_config)

    result = train_ppo(
        game_name="high_card_duel",
        config=ppo_config,
        run_dir=run_dir,
        run_id="value_grad_test",
    )

    assert result.run_dir is not None
    overlays_dir = result.run_dir / "overlays"
    assert overlays_dir.exists()

    # Expect value_grad overlays
    value_grad_files = list(overlays_dir.glob("value_grad_*.json"))
    assert len(value_grad_files) == 2  # 2 episodes

    # Verify overlay metadata
    with open(value_grad_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["label"] == "value_grad"
        assert data["meta"]["method"] == "value_grad"
        assert data["meta"]["target_type"] == "state_value"
        assert data["meta"]["algo"] == "ppo"


def test_ppo_emits_both_policy_and_value_grad(tmp_path: Path) -> None:
    """Test PPO with both policy_grad and value_grad enabled (M21-C)."""
    run_dir = tmp_path / "runs"

    xai_config = XAIConfig(
        enabled=True,
        methods=["policy_grad", "value_grad"],
        every_n_episodes=1,
        max_overlays_per_run=10,
    )

    ppo_config = PPOConfig(total_episodes=2, batch_size=32, xai=xai_config)

    result = train_ppo(
        game_name="high_card_duel",
        config=ppo_config,
        run_dir=run_dir,
        run_id="both_grads_test",
    )

    assert result.run_dir is not None
    overlays_dir = result.run_dir / "overlays"
    assert overlays_dir.exists()

    # Expect both types
    policy_files = list(overlays_dir.glob("policy_grad_*.json"))
    value_files = list(overlays_dir.glob("value_grad_*.json"))

    assert len(policy_files) == 2
    assert len(value_files) == 2

    # Verify both are valid
    with open(policy_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["label"] == "policy_grad"
        assert data["meta"]["target_type"] == "logit_or_q"

    with open(value_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["label"] == "value_grad"
        assert data["meta"]["target_type"] == "state_value"


def test_dqn_rejects_value_grad(tmp_path: Path) -> None:
    """Test that DQN training rejects value_grad config (M21-C)."""
    run_dir = tmp_path / "runs"

    xai_config = XAIConfig(
        enabled=True,
        methods=["value_grad"],  # This should be rejected
        every_n_episodes=1,
        max_overlays_per_run=5,
    )

    dqn_config = DQNConfig(total_episodes=2, xai=xai_config)

    # Should raise ValueError
    with pytest.raises(
        ValueError, match="value_grad overlays are currently only supported for PPO"
    ):
        train_dqn(
            game_name="high_card_duel",
            config=dqn_config,
            run_dir=run_dir,
            run_id="should_fail",
        )


def test_ppo_batch_overlay_engine(tmp_path: Path) -> None:
    """Test PPO with batch overlay engine (M22)."""
    run_dir = tmp_path / "runs"

    xai_config = XAIConfig(
        enabled=True,
        methods=["policy_grad", "value_grad"],
        every_n_episodes=1,
        max_overlays_per_run=10,
        batch_size=3,  # M22: Enable batching
    )

    ppo_config = PPOConfig(total_episodes=3, batch_size=32, xai=xai_config)

    result = train_ppo(
        game_name="high_card_duel",
        config=ppo_config,
        run_dir=run_dir,
        run_id="batch_test",
    )

    assert result.run_dir is not None
    overlays_dir = result.run_dir / "overlays"
    assert overlays_dir.exists()

    # Expect overlays from both methods (3 episodes × 2 methods = 6 files)
    policy_files = list(overlays_dir.glob("policy_grad_*.json"))
    value_files = list(overlays_dir.glob("value_grad_*.json"))

    assert len(policy_files) == 3
    assert len(value_files) == 3

    # Verify overlays are valid
    for overlay_file in policy_files + value_files:
        with open(overlay_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert "importance" in data
            assert len(data["importance"]) == 4  # 4 suits
            assert len(data["importance"][0]) == 14  # 14 ranks
