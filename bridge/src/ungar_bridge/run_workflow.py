"""Helper to run the UNGAR demo workflow via RediAI."""

from typing import Any

from .rediai_adapter import HAS_REDAI

if HAS_REDAI:
    try:
        from RediAI.cli import run_workflow  # type: ignore[import-not-found]
    except ImportError:
        # Should not happen if HAS_REDAI is True, unless RediAI structure changed
        HAS_REDAI = False


async def run_demo_workflow(
    workflow_path: str = "bridge/examples/workflows/ungar_demo.yaml",
) -> Any:
    """Run the specified workflow using RediAI.

    Args:
        workflow_path: Path to the workflow YAML file.

    Returns:
        The result of the workflow execution.

    Raises:
        RuntimeError: If RediAI is not installed.
    """
    if not HAS_REDAI:
        raise RuntimeError(
            "RediAI is not installed. Install with `pip install ungar-bridge[rediai]`."
        )

    # In a real scenario, this calls RediAI's runner
    return await run_workflow(workflow_path)
