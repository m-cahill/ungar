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

### XAI Overlay Scaffold

The training loop supports generating XAI (Explainable AI) overlays for game states.

*   **Structure:** `CardOverlay` (defined in `ungar.xai`).
*   **Shape:** 4×14 grid of importance values (floats).
*   **Heuristic:** For High Card Duel, we implement a simple placeholder overlay that highlights the current player's hand with importance `1.0`.

This infrastructure allows the training loop to produce "explanations" alongside rewards, which will be used in future milestones for integration with RediAI's interpretability tools.

> **Note:** The XAI pipeline also supports **Mini Spades** (added in M10). See `ungar_bridge/rediai_spades.py` for the implementation.

#### Enabling Overlays

Pass `record_overlays=True` to the training function:

```python
result = train_high_card_duel(..., record_overlays=True)
# Access overlay for the last episode
overlay = result.config["last_overlay"]
```

### Reward Decomposition (RewardLab)

The training loop also decomposes rewards into components for finer-grained analysis:
*   `win_loss`: The raw game outcome.
*   `baseline`: A baseline component (currently 0.0).

These components are stored in `result.components` and can be exported to RediAI RewardLab.
