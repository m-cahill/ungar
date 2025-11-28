# Training with DQN

UNGAR provides a built-in training runner for Deep Q-Learning (DQN) that works out-of-the-box for all implemented games.

## Quickstart

To train a DQN agent on a specific game:

```python
from ungar.training.train_dqn import train_dqn

# Train High Card Duel
result = train_dqn("high_card_duel", episodes=200)
print(f"Avg Reward: {result.metrics['avg_reward']}")

# Train Gin Rummy
result = train_dqn("gin_rummy", episodes=500)
```

## Configuration

The `train_dqn` function supports basic hyperparameters:

*   `episodes`: Number of games to play.
*   `lr`: Learning rate (default 0.0005).
*   `buffer_size`: Replay buffer capacity (default 5000).
*   `seed`: Random seed for reproducibility.

## How it Works

1.  **Adapter Selection:** The runner selects the appropriate `GameAdapter` for the requested game to determine input shapes and action spaces.
2.  **Agent Initialization:** A `DQNLiteAgent` is created with the correct network architecture.
3.  **Training Loop:**
    *   The environment runs in a loop (self-play context).
    *   The agent observes the state (masked by legal moves) and acts.
    *   Transitions `(s, a, r, s', done)` are pushed to the replay buffer.
    *   The agent performs a training step (sampling from buffer, backprop) at every step.

## Supported Games

*   `high_card_duel`
*   `spades_mini`
*   `gin_rummy`

