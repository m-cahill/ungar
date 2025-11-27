"""Shared types and protocols for UNGAR Bridge."""

from __future__ import annotations

from typing import Any, Protocol


class WorkflowRecorder(Protocol):
    """Protocol for RediAI workflow recorder."""

    async def record_metric(self, name: str, value: float, **kwargs: Any) -> None:
        """Record a scalar metric."""
        ...

    async def record_artifact(self, name: str, path: str, **kwargs: Any) -> None:
        """Record a file artifact."""
        ...
