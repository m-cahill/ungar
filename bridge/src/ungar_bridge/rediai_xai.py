"""RediAI XAI Integration Bridge.

This module bridges UNGAR's XAI overlays to RediAI's artifact logging system.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ungar.xai import CardOverlay, overlay_to_dict

from .types import WorkflowRecorder

HAS_REDAI_XAI = True

try:
    # Adjust import paths based on actual RediAI repo
    # Assuming the same import path as rediai_training.py, or similar
    from RediAI.registry.recorder import WorkflowRecorder as _RealRecorder  # noqa: F401
except ImportError:
    HAS_REDAI_XAI = False


def is_rediai_xai_available() -> bool:
    """Return True if RediAI XAI features are available."""
    return HAS_REDAI_XAI


async def log_overlays_as_artifact(
    recorder: WorkflowRecorder,
    overlays: Sequence[CardOverlay],
    artifact_name: str = "ungar_card_overlays.json",
    tmp_dir: str = ".",
) -> str:
    """Serialize overlays to JSON and log as a RediAI artifact.

    Args:
        recorder: The RediAI workflow recorder.
        overlays: Sequence of CardOverlay objects.
        artifact_name: Name of the artifact file.
        tmp_dir: Directory to write the temporary file.

    Returns:
        The path to the written artifact file.
    """
    path = Path(tmp_dir) / artifact_name
    data = [overlay_to_dict(o) for o in overlays]
    path.write_text(json.dumps(data), encoding="utf-8")

    await recorder.record_artifact(artifact_name, str(path))
    return str(path)
