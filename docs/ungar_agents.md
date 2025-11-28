# UNGAR Agents

M12 introduces a unified agent system to support both random baselines and RL agents (DQN) across all supported games.

## Architecture

The agent system is built on a few core abstractions:

1.  **UnifiedAgent Protocol** (`ungar.agents.unified_agent`):
    *   Defines the standard interface: `act(obs, legal_moves)` and `train_step(transition)`.
    *   Ensures all agents (Random, DQN, PPO future) are pluggable.

2.  **GameAdapter** (`ungar.agents.adapters`):
    *   Bridges the gap between raw game state (Move objects) and RL needs (integer action IDs, tensor shapes).
    *   One adapter per game (`HighCardAdapter`, `SpadesAdapter`, `GinAdapter`).

3.  **DQNLiteAgent** (`ungar.agents.dqn_lite`):
    *   A clean, dependency-free (PyTorch only) implementation of Deep Q-Networks.
    *   Features: Target networks, Replay Buffer, Epsilon-greedy exploration, Legal move masking.

## Supported Agents

### RandomUnifiedAgent
A baseline that selects legal moves uniformly at random. Useful for testing environment mechanics.

### DQNLiteAgent
A lightweight DQN implementation suitable for learning card game strategies. It handles the flattened `4x14xn` card tensor directly.

## Usage

Agents are typically instantiated via the training runner, but can be used standalone:

```python
from ungar.agents.dqn_lite import DQNLiteAgent

agent = DQNLiteAgent(input_dim=168, action_space_size=1)
action = agent.act(observation, legal_moves=[0])
```

