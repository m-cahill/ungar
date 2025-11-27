# mypy: ignore-errors
import importlib
import importlib.util
from typing import Any
from unittest.mock import patch

import pytest

# Try to locate the demo module in either layout:
#   - In-repo:   bridge.examples.demo_rediai
#   - Installed: ungar_bridge.examples.demo_rediai
spec = importlib.util.find_spec("bridge.examples.demo_rediai") or importlib.util.find_spec(
    "ungar_bridge.examples.demo_rediai"
)

if spec is None:
    pytest.skip(
        "RediAI demo not available (no bridge.examples.demo_rediai or "
        "ungar_bridge.examples.demo_rediai); skipping RediAI smoke test.",
        allow_module_level=True,
    )

# Dynamically import whichever module was found.
demo_rediai = importlib.import_module(spec.name)


class FakeRecorder:
    """Mock RediAI recorder for testing."""

    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, dict[str, Any]]] = []

    async def record_metric(self, name: str, value: float, **kwargs: Any) -> None:
        self.metrics.append((name, value, kwargs))


@pytest.mark.asyncio
async def test_rediai_smoke_demo() -> None:
    """Smoke test for the RediAI demo script.

    Verifies that demo_rediai.main() runs to completion when RediAI checks are bypassed.
    """

    # We patch is_rediai_available in the adapter module to allow make_rediai_env to succeed
    with patch("ungar_bridge.rediai_adapter.is_rediai_available", return_value=True):
        # We also need to patch workflow_context in the demo script to capture metrics
        # (Though the demo script already has a dummy fallback, for testing we might want to inspect calls)
        # But for smoke test "runs through without runtime errors" is the main goal.
        # The demo script's internal fallback prints to stdout.
        # Let's try to capture it or just run it.

        # To be strict about "Acceptance Criteria: One metric logged", we should inspect it.
        # The demo script uses `from RediAI... import workflow_context` or defines a dummy.
        # We can patch `demo_rediai.workflow_context` directly.

        from contextlib import asynccontextmanager

        captured_metrics: list[tuple[str, float]] = []

        @asynccontextmanager
        async def mock_context() -> Any:
            class _Mock:
                async def record_metric(self, name: str, value: float, **kwargs: Any) -> None:
                    captured_metrics.append((name, value))

            yield _Mock()

        with patch("bridge.examples.demo_rediai.workflow_context", side_effect=mock_context):
            await demo_rediai.main()

    # Assertions
    assert len(captured_metrics) > 0
    names = [m[0] for m in captured_metrics]
    assert "step_reward" in names or "total_steps" in names
