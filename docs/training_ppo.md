# Training with PPO

UNGAR provides a built-in training runner for Proximal Policy Optimization (PPO) that works out-of-the-box for all implemented games.

## Quickstart

To train a PPO agent on a specific game:

```python
from ungar.training.train_ppo import train_ppo
from ungar.training.config import PPOConfig

# Train with defaults
result = train_ppo("high_card_duel")

# Train with custom config
config = PPOConfig(
    total_episodes=100,
    learning_rate=1e-4,
    clip_coef=0.2,
    gae_lambda=0.95,
)
result = train_ppo("gin_rummy", config=config)
print(f"Avg Reward: {result.metrics['avg_reward']}")
```

## Configuration

The `PPOConfig` dataclass supports:

*   `total_episodes`: Number of games to play.
*   `learning_rate`: Adam optimizer learning rate (default 3e-4).
*   `gamma`: Discount factor (default 0.99).
*   `batch_size`: Total steps collected before update (default 64).
*   `clip_coef`: PPO clipping coefficient (default 0.2).
*   `value_coef`: Value loss weight (default 0.5).
*   `entropy_coef`: Entropy bonus weight (default 0.01).
*   `update_epochs`: Epochs over the buffer per update (default 4).
*   `minibatch_size`: SGD minibatch size (default 16).
*   `gae_lambda`: GAE lambda for advantage estimation (default 0.95).
*   `device`: `DeviceConfig` for GPU selection (default "auto").

## Logging

Metrics (episode reward, length, loss) can be logged to CSV or TensorBoard:

```python
from ungar.training.logger import FileLogger, TensorBoardLogger

# CSV logging
logger = FileLogger("logs")
train_ppo("spades_mini", logger=logger)

# TensorBoard logging
logger = TensorBoardLogger("runs/spades_experiment")
train_ppo("spades_mini", logger=logger)
```

## How it Works

1.  **Adapter Selection:** The runner selects the appropriate `GameAdapter` for the requested game.
2.  **Agent Initialization:** A `PPOLiteAgent` is created with Actor-Critic architecture.
3.  **Training Loop:**
    *   The environment runs in a self-play context.
    *   The agent observes states and acts via its policy network.
    *   Transitions `(s, a, logprob, r, done, value)` are collected into a rollout buffer.
    *   At the end of each episode, the agent performs PPO updates (multiple epochs over the buffer).

## Supported Games

*   `high_card_duel`
*   `spades_mini`
*   `gin_rummy`

## PPO vs DQN

*   **PPO:** On-policy, Actor-Critic. Better for games with complex action spaces or delayed rewards.
*   **DQN:** Off-policy, value-based. Faster for simple games with discrete actions.

For High Card Duel, both work. For Gin Rummy (complex state), PPO may learn faster.

## Analysis

See [docs/analytics_overview.md](analytics_overview.md) for how to visualize training logs and XAI overlays.
