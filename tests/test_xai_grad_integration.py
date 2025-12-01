"""Integration test for gradient-based XAI overlays."""

import json
from pathlib import Path

from ungar.training.config import DQNConfig, XAIConfig
from ungar.training.train_dqn import train_dqn


def test_training_emits_grad_overlays(tmp_path: Path) -> None:
    """Run a minimal training session with policy_grad enabled and verify output."""
    run_dir = tmp_path / "runs"

    xai_config = XAIConfig(
        enabled=True,
        methods=["heuristic", "policy_grad"],
        every_n_episodes=1,
        max_overlays_per_run=10
    )
    
    dqn_config = DQNConfig(
        total_episodes=2,
        xai=xai_config
    )
    
    result = train_dqn(
        game_name="high_card_duel",
        config=dqn_config,
        run_dir=run_dir,
        run_id="grad_xai_test"
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

