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

You can run training loops that hook into RediAI's `WorkflowRecorder`.

**High Card Duel:**
```python
from ungar_bridge.rediai_training import train_high_card_duel_rediai

result = await train_high_card_duel_rediai(num_episodes=5000, record_overlays=True)
```

**Mini Spades (New in M10):**
```python
from ungar_bridge.rediai_spades import train_spades_mini_rediai

result = await train_spades_mini_rediai(num_episodes=200, record_overlays=True)
```

### XAI & RewardLab Integration

The training loop also supports advanced artifacts for Explainable AI (XAI) and Reward Decomposition.

#### XAI Overlays
When `record_overlays=True` is passed to the training function:
1. An XAI overlay (attribution map) is generated for the final episode.
2. It is serialized to JSON and attached as an artifact named `ungar_high_card_overlays.json` or `ungar_spades_overlays.json`.
3. This artifact can be visualized in RediAI dashboards.

#### Reward Decomposition
High Card Duel training automatically logs reward decomposition data:
1. Each episode's reward is broken down into components (e.g., `win_loss`, `baseline`).
2. This data is serialized to `ungar_reward_decomposition.json`.
3. It enables granular analysis via RediAI's RewardLab tools.

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
