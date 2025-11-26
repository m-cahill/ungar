from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_workflow_execution():
    """Test that run_demo_workflow calls the RediAI CLI runner correctly."""
    
    # Mock HAS_REDAI to True
    with patch("ungar_bridge.run_workflow.HAS_REDAI", True):
        # Mock run_workflow
        mock_runner = AsyncMock(return_value={"status": "completed", "metrics": {"demo_reward": 10.0}})
        
        # We need to patch where run_workflow is IMPORTED in ungar_bridge.run_workflow
        # But wait, run_workflow imports it inside the module scope if HAS_REDAI is True.
        # Since HAS_REDAI is False in reality, the import inside ungar_bridge.run_workflow failed or wasn't executed.
        # So ungar_bridge.run_workflow.run_workflow (the symbol) might not exist or be None.
        
        # We need to patch sys.modules or patch the function in the module if it was imported.
        # Easier: Patch the function where it is used. But it's used inside the function `run_demo_workflow`.
        # `from RediAI.cli import run_workflow` happens at top level if HAS_REDAI.
        
        # Strategy:
        # 1. Patch `ungar_bridge.run_workflow.HAS_REDAI` to True.
        # 2. But `ungar_bridge.run_workflow` has already been imported? 
        #    If imported, the top-level `if HAS_REDAI:` block already ran with False.
        #    So `run_workflow` is not imported.
        #    So `run_demo_workflow` will crash with NameError if it tries to call `run_workflow`?
        #    Wait, `run_demo_workflow` calls `run_workflow(workflow_path)`.
        #    If `run_workflow` was not imported, it's not in the global namespace of that module.
        
        # Correct. We need to reload the module or patch `ungar_bridge.run_workflow.run_workflow`.
        # But if it wasn't defined, we can't patch it easily on the module object if it doesn't exist.
        # Actually `patch.object` can create it if using `create=True`.
        
        from ungar_bridge import run_workflow as run_workflow_module
        
        with patch.object(run_workflow_module, "HAS_REDAI", True):
             with patch.object(run_workflow_module, "run_workflow", new=mock_runner, create=True):
                 from ungar_bridge.run_workflow import run_demo_workflow
                 
                 result = await run_demo_workflow("dummy_path.yaml")
                 
                 assert result["status"] == "completed"
                 mock_runner.assert_called_once_with("dummy_path.yaml")

@pytest.mark.asyncio
async def test_workflow_guard():
    """Test that run_demo_workflow raises if RediAI is missing."""
    from ungar_bridge import run_workflow as run_workflow_module
    
    # Ensure HAS_REDAI is False (it should be by default in this env)
    # But explicitly setting/patching to be sure.
    with patch.object(run_workflow_module, "HAS_REDAI", False):
        from ungar_bridge.run_workflow import run_demo_workflow
        
        with pytest.raises(RuntimeError, match="RediAI is not installed"):
            await run_demo_workflow()

