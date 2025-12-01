"""Integration test for XAI overlay generation in training."""

from pathlib import Path

from ungar.training.config import DQNConfig, XAIConfig


def test_training_emits_overlays(tmp_path: Path) -> None:
    """Run a minimal training session with XAI enabled and verify output."""
    run_dir = tmp_path / "runs"

    # We can't easily modify the config class defaults globally for cmd_train via CLI args yet
    # because CLI parsing for nested config isn't implemented.
    # However, cmd_train creates default configs.
    # We need to invoke train_dqn directly to pass a custom config,
    # OR we can update cmd_train to support hidden/advanced args,
    # OR we can mock the config creation inside cmd_train.

    # Actually, the plan M19-B Task 4 says "In your existing analytics integration test... add XAIConfig".
    # Let's call train_dqn directly to have full control.

    from ungar.training.train_dqn import train_dqn

    xai_config = XAIConfig(
        enabled=True, methods=["heuristic", "random"], every_n_episodes=1, max_overlays_per_run=10
    )

    dqn_config = DQNConfig(total_episodes=2, xai=xai_config)

    result = train_dqn(
        game_name="high_card_duel", config=dqn_config, run_dir=run_dir, run_id="xai_test"
    )

    assert result.run_dir is not None
    overlays_dir = result.run_dir / "overlays"
    assert overlays_dir.exists()

    # We expect 2 episodes * 2 methods = 4 overlays
    files = list(overlays_dir.glob("*.json"))
    assert len(files) == 4

    # Validate one
    import json

    from ungar.analysis.schema import validate_overlay

    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
        validate_overlay(data)
        assert data["label"] in ["heuristic", "random"]
