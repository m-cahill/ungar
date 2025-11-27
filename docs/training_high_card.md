# Local Training with UngarGymEnv

The `ungar-bridge` package provides a Gymnasium-style environment wrapper (`UngarGymEnv`) that enables standard Reinforcement Learning (RL) loops on top of UNGAR core games.

## Architecture

The RL adapter bridges the gap between UNGAR's event-driven, multi-agent core and the standard step-based RL interface.

### 1. UngarGymEnv (The Wrapper)

*   **Wraps:** `ungar.game.GameEnv`
*   **Implements:** `reset()`, `step()`, `legal_actions()`
*   **State:** Tracks `current_player` and handles the mapping from game moves to RL actions.

### 2. Observation Space

The environment exposes the raw 4x14xN boolean tensor (as float32) from the perspective of the *current player*.

*   **Shape:** `(4, 14, N)`
*   **Planes (N):** Game-dependent. For High Card Duel, N=3 (`my_hand`, `opponent_hand`, `unseen`).

### 3. Multi-Agent Handling

Since most card games are turn-based multi-agent:

1.  `reset()` returns the observation for the starting player (usually P0).
2.  `step(action)` applies the move for the *current* player.
3.  The return tuple `(obs, reward, terminated, truncated, info)` provides:
    *   `obs`: Observation for the *next* player to act.
    *   `reward`: Reward for the player who *just acted*.
4.  The loop must check `env.current_player` to know whose turn it is (though for simple self-play training, we often treat the agent as "the current player").

## High Card Duel Training

We provide a reference implementation of a local training loop for High Card Duel.

*   **File:** `bridge/src/ungar_bridge/training.py`
*   **Algorithm:** Simple epsilon-greedy bandit (since the game is 1-step).
*   **Goal:** Demonstrate the E2E flow from `GameEnv` -> `UngarGymEnv` -> Training -> Results.

### Running the Training

```bash
make train-high-card
```

This runs 1000 episodes and reports average reward. Since High Card Duel is zero-sum and random, the average reward usually converges near 0.0 against a random opponent (or self-play).

