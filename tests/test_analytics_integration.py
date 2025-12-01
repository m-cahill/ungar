"""Integration test ensuring training produces valid analytics artifacts."""

import json
from argparse import Namespace
from pathlib import Path

from ungar.analysis.schema import validate_manifest, validate_metrics_file
from ungar.cli import cmd_train


def make_args(**kwargs: object) -> Namespace:
    """Helper to create argparse.Namespace objects."""
    ns = Namespace()
    for key, value in kwargs.items():
        setattr(ns, key, value)
    return ns


def test_training_produces_valid_schema(tmp_path: Path) -> None:
    """Run a minimal training session and validate output artifacts."""
    run_dir = tmp_path / "runs"

    args = make_args(
        game="high_card_duel",
        algo="dqn",
        episodes=2,
        run_dir=str(run_dir),
        device="cpu",
    )

    # 1. Run training
    cmd_train(args)

    # 2. Verify run directory creation
    assert run_dir.exists()
    runs = [d for d in run_dir.iterdir() if d.is_dir()]
    assert len(runs) == 1
    run_path = runs[0]

    # 3. Validate Manifest
    manifest_path = run_path / "manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # This checks required fields including created_at and analytics_schema_version
    validate_manifest(manifest)

    # 4. Validate Metrics
    metrics_path = run_path / "metrics.csv"
    assert metrics_path.exists()

    # This checks headers and content (including episode column)
    validate_metrics_file(metrics_path)
