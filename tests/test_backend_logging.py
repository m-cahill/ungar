"""Tests for backend logging and overlay export."""

import json
from pathlib import Path

from ungar.training.logger import FileLogger
from ungar.training.overlay_exporter import OverlayExporter
from ungar.xai import zero_overlay


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
    exporter = OverlayExporter(export_dir=tmp_path)

    overlay1 = zero_overlay("test1")
    overlay2 = zero_overlay("test2")

    exporter.add(overlay1)
    exporter.add(overlay2)

    output_path = exporter.save("test_overlays.json")

    assert output_path.exists()

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["label"] == "test1"
    assert data[1]["label"] == "test2"

    exporter.clear()
    assert len(exporter.overlays) == 0
