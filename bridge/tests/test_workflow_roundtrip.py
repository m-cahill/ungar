from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_workflow_execution() -> None:
    """Test that run_demo_workflow calls the RediAI CLI runner correctly."""

    # Mock is_rediai_available to True (not used in run_workflow yet, but good practice if refactored)
    # run_workflow uses HAS_REDAI directly currently.

    # Mock run_workflow
    mock_runner = AsyncMock(return_value={"status": "completed", "metrics": {"demo_reward": 10.0}})

    from ungar_bridge import run_workflow as run_workflow_module

    # Patch HAS_REDAI to allow the check inside run_demo_workflow to pass
    # Also patch run_workflow function
    with patch.object(run_workflow_module, "HAS_REDAI", True):
        with patch.object(run_workflow_module, "run_workflow", new=mock_runner, create=True):
            from ungar_bridge.run_workflow import run_demo_workflow

            result = await run_demo_workflow("dummy_path.yaml")

            assert result["status"] == "completed"
            mock_runner.assert_called_once_with("dummy_path.yaml")


@pytest.mark.asyncio
async def test_workflow_guard() -> None:
    """Test that run_demo_workflow raises if RediAI is missing."""
    from ungar_bridge import run_workflow as run_workflow_module

    # Ensure HAS_REDAI is False (it should be by default in this env)
    # But explicitly setting/patching to be sure.
    with patch.object(run_workflow_module, "HAS_REDAI", False):
        from ungar_bridge.run_workflow import run_demo_workflow

        with pytest.raises(RuntimeError, match="RediAI is not installed"):
            await run_demo_workflow()
