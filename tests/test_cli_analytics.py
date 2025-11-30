"""Tests for CLI analytics commands."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from ungar.cli import cmd_export_run, cmd_list_runs
from ungar.training.run_dir import create_run_dir


@pytest.fixture
def runs_dir(tmp_path):
    """Create a temporary runs directory with one valid run."""
    d = tmp_path / "runs"
    d.mkdir()
    
    # Create a valid run
    paths = create_run_dir(
        game="high_card_duel",
        algo="dqn",
        config_dict={"lr": 0.01},
        device="cpu",
        base_dir=d,
        run_id="test_run",
    )
    
    # Create valid metrics
    with open(paths.metrics, "w", encoding="utf-8") as f:
        f.write("step,episode,reward\n1,1,0.5\n")
        
    return d


@pytest.mark.smoke
def test_list_runs_json(runs_dir, capsys):
    class Args:
        format = "json"

    with patch("ungar.cli._get_runs_dir", return_value=runs_dir):
        cmd_list_runs(Args())

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    
    assert isinstance(output, list)
    assert len(output) == 1
    assert output[0]["run_id"] == "test_run"
    assert output[0]["game"] == "high_card_duel"
    assert "path" in output[0]


def test_export_run(runs_dir, tmp_path):
    out_dir = tmp_path / "export"
    
    class Args:
        pass
    
    Args.run_id = "test_run"
    Args.out_dir = str(out_dir)

    with patch("ungar.cli._get_runs_dir", return_value=runs_dir):
        cmd_export_run(Args())

    assert out_dir.exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "metrics.csv").exists()


def test_export_run_not_found(runs_dir, tmp_path):
    class Args:
        run_id = "missing_run"
        out_dir = str(tmp_path / "export")

    with patch("ungar.cli._get_runs_dir", return_value=runs_dir):
        with pytest.raises(SystemExit):
            cmd_export_run(Args())


def test_export_run_invalid_schema(runs_dir, tmp_path):
    # Corrupt the manifest
    run_path = runs_dir / [x for x in runs_dir.iterdir()][0]
    manifest_path = run_path / "manifest.json"
    
    with open(manifest_path, "r") as f:
        data = json.load(f)
    
    del data["algo"]  # Make it invalid
    
    with open(manifest_path, "w") as f:
        json.dump(data, f)

    class Args:
        run_id = "test_run"
        out_dir = str(tmp_path / "export")

    with patch("ungar.cli._get_runs_dir", return_value=runs_dir):
        with pytest.raises(SystemExit):
            cmd_export_run(Args())


@pytest.mark.smoke
def test_cli_help_runs_without_viz_deps() -> None:
    """Ensure CLI runs without matplotlib/pandas installed."""
    # We can't easily uninstall deps in a test, but we can check that --help returns 0
    # and doesn't crash on imports.
    # The 'ungar' command might not be in path during test if not installed in venv,
    # so we invoke via python -m ungar.cli or similar if needed.
    # But project structure has ungar script. Let's try python -m ungar.cli
    
    result = subprocess.run(
        [sys.executable, "-m", "ungar.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "UNGAR" in result.stdout
