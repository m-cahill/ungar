"""Tests for backend logging and overlay export."""

import json
from pathlib import Path

from ungar.training.logger import FileLogger
from ungar.training.overlay_exporter import OverlayExporter


def test_file_logger_csv(tmp_path: Path) -> None:
    """Test CSV logging."""
    logger = FileLogger(log_dir=tmp_path, format="csv")

    logger.log_metrics({"loss": 0.5, "reward": 1.0}, step=1)
    logger.log_metrics({"loss": 0.3, "reward": 2.0}, step=2)
    logger.close()

    # Check file exists
    log_files = list(tmp_path.glob("metrics_*.csv"))
    assert len(log_files) == 1

    # Check content
    content = log_files[0].read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert len(lines) == 3  # Header + 2 rows
    assert "step,loss,reward" in lines[0] or "step,reward,loss" in lines[0]
    assert "1,0.5,1.0" in lines[1] or "1,1.0,0.5" in lines[1]


def test_overlay_exporter(tmp_path: Path) -> None:
    """Test overlay export to JSON."""
    # M19 updated OverlayExporter signature
    import numpy as np
    from ungar.xai_methods import RandomOverlayMethod

    exporter = OverlayExporter(out_dir=tmp_path, methods=[RandomOverlayMethod()], max_overlays=10)

    # export() computes and saves directly
    obs = np.zeros(56)  # 4x14 flattened
    exporter.export(obs=obs, action=0, step=1, run_id="test_run")
    exporter.export(obs=obs, action=0, step=2, run_id="test_run")

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 2

    # Check content of one
    with open(files[0], "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["label"] == "random"
        assert data["run_id"] == "test_run"
