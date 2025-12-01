"""Tests for the UNGAR CLI."""

import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from ungar.cli import main
from ungar.training.run_dir import RunManifest


@pytest.fixture
def mock_runs_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a mock runs directory with one run."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Create a run
    run_id = "test_123"
    run_path = runs_dir / f"123456_game_algo_{run_id}"
    run_path.mkdir()

    manifest = RunManifest(
        run_id=run_id,
        timestamp=123456.0,
        created_at="2025-01-01T00:00:00Z",
        analytics_schema_version=1,
        game="test_game",
        algo="dqn",
        config={"lr": 0.01},
        device="cpu",
        metrics_path="metrics.csv",
        overlays_path="overlays",
        notes="test run",
    )
    manifest.save(run_path / "manifest.json")

    with patch("ungar.cli._get_runs_dir", return_value=runs_dir):
        yield runs_dir


def run_cli(args: list[str]) -> None:
    """Helper to run CLI with arguments."""
    with patch.object(sys, "argv", ["ungar"] + args):
        main()


def test_list_runs(mock_runs_dir: Path, capsys: pytest.CaptureFixture) -> None:
    """Test listing runs."""
    run_cli(["list-runs"])
    captured = capsys.readouterr()
    assert "test_123" in captured.out
    assert "test_game" in captured.out


def test_show_run(mock_runs_dir: Path, capsys: pytest.CaptureFixture) -> None:
    """Test showing a run."""
    run_cli(["show-run", "test_123"])
    captured = capsys.readouterr()
    assert '"run_id": "test_123"' in captured.out
    assert '"lr": 0.01' in captured.out


def test_train_smoke(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test training CLI (smoke test)."""
    # We mock the actual training function to avoid long runtime
    with patch("ungar.cli.train_dqn") as mock_train:
        mock_train.return_value = MagicMock(run_dir=tmp_path / "new_run", metrics={})

        run_cli(
            [
                "train",
                "--game",
                "high_card_duel",
                "--algo",
                "dqn",
                "--episodes",
                "10",
                "--run-dir",
                str(tmp_path),
            ]
        )

        mock_train.assert_called_once()
        assert "Training complete" in capsys.readouterr().out


def test_plot_curves_smoke(tmp_path: Path) -> None:
    """Test plotting CLI (smoke test)."""
    # Because we moved imports inside the function, we must patch where they are IMPORTED from,
    # or patch sys.modules before the function runs if we want to mock the module itself.
    # But cmd_plot_curves does: from ungar.analysis.plots import plot_learning_curve
    # So we should patch ungar.analysis.plots.plot_learning_curve

    with patch("ungar.analysis.plots.plot_learning_curve") as mock_plot:
        # Mock matplotlib import check inside CLI
        with patch.dict(sys.modules, {"matplotlib": MagicMock()}):
            run_cli(["plot-curves", "--run", "some/path", "--out", str(tmp_path / "curve.png")])
            mock_plot.assert_called_once()


def test_summarize_overlays_smoke(tmp_path: Path) -> None:
    """Test overlay summary CLI (smoke test)."""
    # Similarly, imports are local now.
    # We need a real-ish overlay for load_overlays return value, because compute_mean_overlay checks shape.
    from ungar.xai import zero_overlay

    real_overlay = zero_overlay("test")

    with patch("ungar.analysis.overlays.load_overlays", return_value=[real_overlay]), patch(
        "ungar.analysis.overlays.save_aggregation"
    ), patch("ungar.analysis.plots.plot_overlay_heatmap"), patch.dict(
        sys.modules, {"matplotlib": MagicMock()}
    ):
        run_cli(["summarize-overlays", "--run", "some/path", "--out-dir", str(tmp_path)])


def test_compare_overlays_smoke(tmp_path: Path) -> None:
    """Test compare-overlays CLI (smoke test)."""
    from ungar.xai import zero_overlay
    
    o1 = zero_overlay("A")
    o2 = zero_overlay("B")
    
    with patch("ungar.analysis.overlays.load_overlays", return_value=[o1, o2]), patch(
        "ungar.analysis.overlays.compare_overlays", return_value=zero_overlay("diff")
    ), patch("ungar.analysis.plots.plot_overlay_heatmap"), patch.dict(
        sys.modules, {"matplotlib": MagicMock()}
    ):
        run_cli([
            "compare-overlays",
            "--run", "some/path",
            "--label-a", "A",
            "--label-b", "B",
            "--out", str(tmp_path / "diff.json")
        ])
        assert (tmp_path / "diff.json").exists()
