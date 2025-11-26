# UNGAR RediAI Bridge

The `ungar-bridge` package provides integration between UNGAR core games and the RediAI platform.

## Overview

The bridge implements a RediAI `EnvAdapter` that wraps standard UNGAR `GameEnv` instances. This allows UNGAR games to be used within RediAI workflows for:
- Large-scale training
- XAI analysis
- Tournaments
- Reward decomposition

## Installation

The bridge comes with UNGAR, but RediAI integration is optional.

To install with RediAI support:

```bash
pip install "ungar-bridge[rediai]"
```

If RediAI is not installed, the bridge code is still importable but will raise `RuntimeError` if you try to instantiate RediAI-specific components.

## Usage

### Creating an Adapter

```python
from ungar_bridge.rediai_adapter import make_rediai_env

# Creates an adapter for High Card Duel
env = make_rediai_env("high_card_duel")

# Use as a standard RediAI EnvAdapter
obs = await env.reset()
action = env.legal_moves()[0]
obs, reward, done, info = await env.step(action)
```

### Running the Demo Workflow

A demonstration workflow is provided in `bridge/examples/demo_rediai.py`.

To run it (requires RediAI):

```bash
python bridge/examples/demo_rediai.py
```

Or programmatically:

```python
from ungar_bridge.run_workflow import run_demo_workflow

await run_demo_workflow()
```

## extending

To add support for a new game:
1. Ensure the game implements the UNGAR `GameSpec` and `GameState` protocols.
2. Update `make_rediai_env` in `ungar_bridge/rediai_adapter.py` to register the new game name.

## Architecture

- **Core Agnostic**: `ungar` core package does not depend on `RediAI`.
- **Bridge**: `ungar-bridge` depends on `ungar` and optionally `RediAI`.
- **Guardrails**: All RediAI imports are guarded. If RediAI is missing, the bridge falls back to stub protocols to allow type checking and local tests to pass.

