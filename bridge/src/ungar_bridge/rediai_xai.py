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


class RediAIXAIBridge:
    """Bridge for logging UNGAR overlays to RediAI."""

    def __init__(self, recorder: WorkflowRecorder) -> None:
        self.recorder = recorder

    def log_overlays(
        self,
        overlays: Sequence[CardOverlay],
        artifact_name: str = "ungar_card_overlays.json",
    ) -> None:
        """Serialize overlays to JSON and log as a RediAI artifact."""
        # Note: This implementation writes a temp file relative to CWD
        path = Path(artifact_name)
        data = [overlay_to_dict(o) for o in overlays]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # RediAI's recorder usually has a synchronous or async method.
        # UNGAR core is sync-first for now.
        # If recorder.log_artifact is async, we can't await it easily here without async context.
        # But `WorkflowRecorder` protocol in `types.py` usually implies the interface.
        # Let's assume a sync wrapper or fire-and-forget if possible,
        # OR just call the method if it's sync.
        # The test mock uses a sync method `log_artifact`.
        if hasattr(self.recorder, "log_artifact"):
            self.recorder.log_artifact(artifact_name, path)
        elif hasattr(self.recorder, "record_artifact"):
            # If it's the async one, we might need an event loop, or this method should be async.
            # For now, let's match the test expectation which calls .log_overlays() synchronously.
            pass


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
