"""Demo script for running UNGAR within a RediAI workflow context.

This script demonstrates how to wrap a UNGAR game and run a simple loop
logging metrics to RediAI.
"""

import asyncio
from typing import Any

# Handle RediAI import gracefully for CI/testing without RediAI installed
try:
    from RediAI.registry.recorder import workflow_context
except ImportError:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def workflow_context() -> Any:
        """Dummy context manager when RediAI is not available."""
        class _Dummy:
            async def record_metric(self, name: str, value: float, **kwargs: Any) -> None:
                print(f"[Mock Recorder] metric={name} value={value}")
        yield _Dummy()

from ungar_bridge.rediai_adapter import make_rediai_env


async def main() -> None:
    """Run a simple episode."""
    # Create the environment adapter
    # Note: In a real environment, this requires RediAI installed.
    # In CI smoke tests, we patch the availability check.
    env = make_rediai_env("high_card_duel")

    async with workflow_context() as recorder:
        print("Starting episode...")
        obs = await env.reset()
        print(f"Initial observation shape: {obs.shape}")
        
        # Simple loop
        done = False
        step_count = 0
        while not done and step_count < 10:
            legal = env.legal_moves()
            if not legal:
                break
                
            action = legal[0]  # Just pick the first move
            obs, reward, done, info = await env.step(action)
            step_count += 1
            
            await recorder.record_metric("step_reward", reward, step=step_count)
            print(f"Step {step_count}: reward={reward}, done={done}")

        await recorder.record_metric("total_steps", float(step_count))
        print("Episode complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError as e:
        # Graceful exit if RediAI is missing and we are running manually
        print(f"Skipping execution: {e}")

