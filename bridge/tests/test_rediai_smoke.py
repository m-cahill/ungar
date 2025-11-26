from unittest.mock import patch

import pytest

# Import the demo script module
from bridge.examples import demo_rediai


class FakeRecorder:
    """Mock RediAI recorder for testing."""
    def __init__(self):
        self.metrics = []

    async def record_metric(self, name: str, value: float, **kwargs):
        self.metrics.append((name, value, kwargs))


@pytest.mark.asyncio
async def test_rediai_smoke_demo():
    """Smoke test for the RediAI demo script.
    
    Verifies that demo_rediai.main() runs to completion when RediAI checks are bypassed.
    """
    
    # We patch HAS_REDAI in the adapter module to allow make_rediai_env to succeed
    with patch("ungar_bridge.rediai_adapter.HAS_REDAI", True):
        # We also need to patch workflow_context in the demo script to capture metrics
        # (Though the demo script already has a dummy fallback, for testing we might want to inspect calls)
        # But for smoke test "runs through without runtime errors" is the main goal.
        # The demo script's internal fallback prints to stdout.
        # Let's try to capture it or just run it.
        
        # To be strict about "Acceptance Criteria: One metric logged", we should inspect it.
        # The demo script uses `from RediAI... import workflow_context` or defines a dummy.
        # We can patch `demo_rediai.workflow_context` directly.
        
        from contextlib import asynccontextmanager
        
        captured_metrics = []

        @asynccontextmanager
        async def mock_context():
            class _Mock:
                async def record_metric(self, name, value, **kwargs):
                    captured_metrics.append((name, value))
            yield _Mock()

        with patch("bridge.examples.demo_rediai.workflow_context", side_effect=mock_context):
            await demo_rediai.main()

    # Assertions
    assert len(captured_metrics) > 0
    names = [m[0] for m in captured_metrics]
    assert "step_reward" in names or "total_steps" in names

