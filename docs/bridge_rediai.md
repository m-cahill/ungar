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

### Training Workflow

You can run training loops that hook into RediAI's `WorkflowRecorder`. This allows you to log metrics, artifacts, and events to the RediAI registry.

```python
from ungar_bridge.rediai_training import train_high_card_duel_rediai

# Run training (async)
# Logs metrics to RediAI if installed
result = await train_high_card_duel_rediai(num_episodes=5000)
print(f"Final Avg Reward: {sum(result.rewards)/len(result.rewards)}")
```

Under the hood, this uses RediAI's `workflow_context` to capture metrics. If RediAI is not available, a dummy context is used so your code remains portable.

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

## Performance

The bridge includes a micro-benchmark to ensure the adapter overhead remains minimal.
Run it via:
```bash
python bridge/benchmarks/benchmark_rediai_adapter.py
```
This script measures `encode_state` latency and is run in CI (non-gating) to track performance trends.
