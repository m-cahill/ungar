"""Bridge XAI Tests."""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
from ungar.xai import CardOverlay
from ungar_bridge.rediai_xai import RediAIXAIBridge


class FakeRecorder:
    """Mock for RediAI WorkflowRecorder."""

    def __init__(self) -> None:
        self.artifacts: Dict[str, Path] = {}

    def log_artifact(self, name: str, path: str | Path) -> None:
        self.artifacts[name] = Path(path)


@pytest.mark.asyncio
async def test_log_overlays_as_artifact(tmp_path: Path) -> None:
    """Test serializing and logging overlays."""
    recorder = FakeRecorder()

    # Create sample overlays
    overlays = [
        CardOverlay(
            run_id="test",
            agg="none",
            step=1,
            importance=np.zeros((4, 14)),
            label="test_1",
            meta={"frame": 1},
        ),
        CardOverlay(
            run_id="test",
            agg="none",
            step=2,
            importance=np.zeros((4, 14)),
            label="test_2",
            meta={"frame": 2},
        ),
    ]

    bridge = RediAIXAIBridge(recorder)  # type: ignore
    bridge.log_overlays(overlays, "test_overlays")

    assert "test_overlays" in recorder.artifacts
    path = recorder.artifacts["test_overlays"]
    assert path.exists()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["label"] == "test_1"
    assert data[1]["label"] == "test_2"
