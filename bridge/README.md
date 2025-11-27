# UNGAR Bridge

This package provides the integration layer between **UNGAR** core games and the **RediAI** platform. It allows UNGAR environments to be wrapped and used within RediAI workflows for training, evaluation, and tournaments.

See [VISION.md](../../VISION.md) and [ungar.md](../../ungar.md) for project context.

## Installation

### Development
From the repository root:
```bash
pip install -e .
pip install -e bridge
```

### With RediAI Support
To enable the RediAI integration (requires RediAI access):
```bash
pip install "ungar-bridge[rediai]"
```

## Quickstart

This example shows how to wrap `HighCardDuel` and run a simple step using the RediAI adapter.

```python
import asyncio
from ungar_bridge.rediai_adapter import make_rediai_env

async def main():
    # 1. Create the adapter
    #    (Raises RuntimeError if RediAI is not installed/available)
    env = make_rediai_env("high_card_duel")

    # 2. Reset (returns 4x14xN tensor)
    obs = await env.reset()
    print(f"Observation shape: {obs.shape}")

    # 3. Step
    moves = env.legal_moves()
    next_obs, reward, done, info = await env.step(moves[0])
    print(f"Reward: {reward}, Done: {done}")

if __name__ == "__main__":
    asyncio.run(main())
```

See `bridge/examples/demo_rediai.py` for a more complete example with workflow recording.

## Local RL Training (No RediAI Required)

The bridge includes a Gymnasium-style RL adapter (`UngarGymEnv`) that supports multi-agent play and can be used for local training loops without RediAI.

A simple bandit-style training script for `HighCardDuel` is provided:

```bash
make train-high-card
```

Or manually:
```bash
python bridge/examples/train_high_card_duel.py --episodes 1000 --seed 42 --epsilon 0.1
```

This runs a deterministic training loop using the 4x14xN tensor observations.

## Optional Dependency

The bridge package is designed to be **safe to import** even without RediAI.
It checks for `RediAI` at runtime. If missing:
*   Importing `ungar_bridge.rediai_adapter` works (using protocol stubs).
*   Calling `make_rediai_env` or instantiating `RediAIUngarAdapter` raises a clear `RuntimeError`.

## Documentation

*   [RediAI Integration](bridge_rediai.md)
*   [Local Training & RL Adapter](../../docs/training_high_card.md)
