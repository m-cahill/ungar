"""Tests for run directory and manifest management."""

import json
from pathlib import Path

from ungar.training.config import DQNConfig
from ungar.training.run_dir import RunManifest, create_run_dir
from ungar.training.train_dqn import train_dqn


def test_create_run_dir(tmp_path: Path) -> None:
    """Test creating a run directory structure."""
    config = {"learning_rate": 0.01, "gamma": 0.99}

    paths = create_run_dir(
        game="test_game",
        algo="dqn",
        config_dict=config,
        device="cpu",
        base_dir=tmp_path,
        run_id="test_123",
        notes="test run",
    )

    # Check directories
    assert paths.root.exists()
    assert paths.overlays.exists()
    assert paths.root.parent == tmp_path
    assert "test_game_dqn_test_123" in paths.root.name

    # Check config
    assert paths.config.exists()
    with open(paths.config) as f:
        loaded_config = json.load(f)
    assert loaded_config == config

    # Check manifest
    assert paths.manifest.exists()
    with open(paths.manifest) as f:
        data = json.load(f)

    manifest = RunManifest.from_dict(data)
    assert manifest.run_id == "test_123"
    assert manifest.game == "test_game"
    assert manifest.algo == "dqn"
    assert manifest.device == "cpu"
    assert manifest.notes == "test run"
    assert manifest.metrics_path == "metrics.csv"


def test_train_dqn_with_run_dir(tmp_path: Path) -> None:
    """Test that training runner correctly populates run directory."""
    config = DQNConfig(total_episodes=2)

    result = train_dqn(
        game_name="high_card_duel", config=config, run_dir=tmp_path, run_id="dqn_test"
    )

    assert result.run_dir is not None
    assert result.run_dir.exists()

    # Check for manifest
    manifest_path = result.run_dir / "manifest.json"
    assert manifest_path.exists()

    # Check for metrics
    metrics_path = result.run_dir / "metrics.csv"
    assert metrics_path.exists()

    # Check content
    content = metrics_path.read_text()
    assert "episode_reward" in content
