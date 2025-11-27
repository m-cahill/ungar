"""Tests for RediAI XAI bridge."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ungar.xai import CardOverlay
from ungar_bridge.rediai_xai import log_overlays_as_artifact


class FakeRecorder:
    """Mock recorder to capture artifact calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def record_artifact(self, name: str, path: str, **_: Any) -> None:
        """Store artifact call arguments."""
        self.calls.append((name, path))


@pytest.mark.asyncio
async def test_log_overlays_as_artifact(tmp_path: Path) -> None:
    """Test serializing and logging overlays."""
    recorder = FakeRecorder()

    # Create sample overlays
    overlays = [
        CardOverlay(importance=np.zeros((4, 14)), label="test_1", meta={"frame": 1}),
        CardOverlay(importance=np.zeros((4, 14)), label="test_2", meta={"frame": 2}),
    ]

    # Convert tmp_path to str for the bridge function
    tmp_dir = str(tmp_path)

    # Call bridge
    artifact_path = await log_overlays_as_artifact(
        recorder,
        overlays,
        artifact_name="test_overlays.json",
        tmp_dir=tmp_dir,
    )

    # Assertions
    assert len(recorder.calls) == 1
    name, path = recorder.calls[0]

    assert name == "test_overlays.json"
    assert path == artifact_path
    assert str(tmp_path) in path

    # Verify content
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["label"] == "test_1"
    assert data[1]["label"] == "test_2"
    assert data[0]["meta"]["frame"] == 1
