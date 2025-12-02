# XAI Overlays in UNGAR

**Status:** v1 (M19)
**Schema:** [Analytics Schema v1](analytics_schema.md)

UNGAR provides a built-in engine for generating, logging, and analyzing Explainable AI (XAI) overlays. These overlays are 4x14 heatmaps that map importance/saliency to specific cards in the UNGAR canonical representation.

## 1. Enabling Overlays

By default, overlay generation is **disabled** to keep training fast. You can enable it via `XAIConfig` in your training script.

### Configuration

```python
from ungar.training.config import DQNConfig, XAIConfig

config = DQNConfig(
    # ... other params ...
    xai=XAIConfig(
        enabled=True,
        methods=["heuristic", "random"],  # List of method labels
        every_n_episodes=10,              # Logging frequency
        max_overlays_per_run=200          # Safety cap
    )
)
```

### Available Methods

*   **`heuristic`**: Highlights cards currently in the agent's hand (normalized). Useful for sanity checking observation encoding.
*   **`random`**: Generates a random heatmap. Useful for testing the pipeline.
*   **`policy_grad`**: Saliency from policy gradient.
    *   **DQN**: Gradient of Q-value for the chosen action w.r.t. input.
    *   **PPO**: Gradient of the chosen action's logit w.r.t. input.
*   **`value_grad`**: Saliency from value/critic gradient (M21).
    *   **PPO**: Gradient of the state-value V(s) w.r.t. input.
    *   **Note**: Only supported for PPO-style actor-critic agents. Not available for DQN.

## 2. Artifacts

Overlays are saved as JSON files in the run directory:

```
runs/<run_id>/
  ├── manifest.json
  ├── metrics.csv
  └── overlays/
      ├── heuristic_10.json   # Method 'heuristic' at step 10
      ├── heuristic_20.json
      ├── random_10.json
      └── ...
```

Each file follows the [Analytics Schema](analytics_schema.md) for overlays.

## 3. Analysis CLI

You can inspect and aggregate overlays using the `ungar` CLI.

### Summarize Overlays

Compute the mean (or max) overlay across a run, optionally filtering by label.

```bash
# Average all heuristic overlays
ungar summarize-overlays --run runs/<id> --label heuristic --agg mean --out-dir analysis_output/

# Compute max saliency for random method
ungar summarize-overlays --run runs/<id> --label random --agg max --out-dir analysis_output/
```

**Outputs:**
*   `overlay_mean.json`: The aggregated overlay object.
*   `overlay_mean_heatmap.png`: A visualization of the 4x14 grid.

### Compare Overlays

Compute the difference between two sets of overlays (e.g., gradient vs heuristic).

```bash
ungar compare-overlays \
    --run runs/<id> \
    --label-a policy_grad \
    --label-b heuristic \
    --agg mean \
    --out comparison.json
```

**Outputs:**
*   `comparison.json`: A valid overlay object representing `(mean(A) - mean(B))`, normalized.
*   `comparison.png`: Heatmap of the difference.

## 4. Integration Guide

### Adding a New Method

1.  Create a class implementing the `OverlayMethod` protocol:

    ```python
    class MyMethod:
        label = "my_method"

        def compute(self, obs, action, *, step, run_id, meta=None) -> CardOverlay:
            # ... logic to produce 4x14 importance matrix ...
            return CardOverlay(...)
## 5. Example Walkthrough

1.  **Train with gradient XAI:**

    ```bash
    # (Assuming config is set in code or passed via a future CLI flag)
    # Enable "policy_grad" in your DQNConfig
    python bridge/examples/train_high_card_duel.py
    ```

2.  **Inspect outputs:**

    ```bash
    ls runs/<latest_run>/overlays/
    # Should see policy_grad_*.json files
    ```

3.  **Compare:**

    ```bash
    ungar compare-overlays \
        --run runs/<latest_run> \
        --label-a policy_grad \
        --label-b heuristic \
        --agg mean \
        --out diff.json
    ```

4.  **Result:** `diff.png` shows where the agent's attention (gradient) differs from the ground truth (heuristic).

## 6. Comparing Policy vs Value Gradients (M21)

After training a PPO agent with both gradient methods enabled, you can compare them:

```bash
# Train PPO with both methods (in code)
# xai = XAIConfig(enabled=True, methods=["policy_grad", "value_grad"])

# Compare the two gradient types
ungar compare-overlays \
    --run runs/<ppo_run_id> \
    --label-a policy_grad \
    --label-b value_grad \
    --agg mean \
    --out diff_policy_vs_value.json

# Outputs:
# - diff_policy_vs_value.json: Difference map (normalized)
# - diff_policy_vs_value.png: Heatmap visualization
```

**Interpretation:**

*   **Positive values** (red): Cards where **policy gradient** is higher → actor focuses more on these cards for action selection.
*   **Negative values** (blue): Cards where **value gradient** is higher → critic focuses more on these cards for state valuation.
*   **Near-zero values** (white): Both networks assign similar importance.

This comparison can reveal **actor-critic misalignment** — cases where the policy network and value network have learned to focus on different features.

### Gradient Method Details

*   **Scalar Target:**
    *   **DQN:** $Q(s, a_{taken})$
    *   **PPO (policy):** Logit of $a_{taken}$
    *   **PPO (value):** State-value $V(s)$
*   **Aggregation:** Gradients are summed across feature planes (channels) to produce a single scalar per card position.
*   **Normalization:** $L_1$ norm (sum to 1). Magnitudes are absolute.

### Value Gradient Overlays (M21)

**Value gradients** (`value_grad`) reveal which cards the critic/value network considers most important for estimating the state value, independent of which action is chosen.

**Key Differences from Policy Gradients:**

| Feature | `policy_grad` | `value_grad` |
|---------|---------------|--------------|
| **Target** | Action logit or Q-value | State-value V(s) |
| **Question Answered** | "What matters for choosing this action?" | "What matters for evaluating this state?" |
| **Action-Dependent** | Yes (varies per action) | No (pure state attribution) |
| **Supported Algorithms** | DQN, PPO | PPO only (actor-critic) |

**Use Cases:**

*   **Critic Introspection**: Understanding what the value network has learned about state importance.
*   **Comparison**: Comparing `policy_grad` vs `value_grad` can reveal whether the actor and critic focus on the same cards.
*   **Debugging**: Identifying if the critic is attending to relevant state features.

**Enabling in PPO:**

```python
from ungar.training.config import PPOConfig, XAIConfig

config = PPOConfig(
    # ... other params ...
    xai=XAIConfig(
        enabled=True,
        methods=["policy_grad", "value_grad"],  # Both gradients
        every_n_episodes=10,
        max_overlays_per_run=200
    )
)
```

**Limitation:**

*   `value_grad` is **not supported** for DQN because DQN does not have an explicit state-value function (only action-value Q(s,a)).
*   Attempting to use `value_grad` with DQN will raise a `ValueError` during training setup.

## 7. Batch Overlay Engine (M22)

**Performance Optimization for Gradient Overlays**

M22 introduces an optional **batch overlay engine** that improves the performance of gradient-based overlay generation by processing multiple overlay requests together.

### What is Batching?

By default, UNGAR generates overlays one at a time (sequential mode). With batching enabled, the system:
1. Accumulates overlay requests in a buffer
2. Processes them together in batches
3. Automatically flushes partial batches at training end

This reduces gradient computation overhead, especially when generating many overlays per episode.

### Enabling Batching

```python
from ungar.training.config import PPOConfig, XAIConfig

# Sequential (default, M21 behavior)
xai = XAIConfig(
    enabled=True,
    methods=["value_grad"],
    batch_size=None,  # Sequential overlay generation
)

# Batched (M22 optimization)
xai = XAIConfig(
    enabled=True,
    methods=["policy_grad", "value_grad"],
    batch_size=4,  # Process up to 4 overlays per batch
)

config = PPOConfig(xai=xai)
```

### Configuration

**`batch_size`** (`int | None`):
- **`None`** (default): Sequential overlay generation (M21 behavior)
- **`1-32`**: Batch size for overlay generation
- Values outside this range will raise a `ValueError`

### Performance Impact

Batching overlay generation can significantly reduce per-overlay computation time:
- Sequential: Each overlay requires a separate forward/backward pass
- Batched: Multiple overlays processed together, reducing overhead

Typical speedup depends on hardware and batch size, but users have observed **several-times faster** overlay generation on GPU-enabled systems.

### Implementation Details

- **Per-method buffers**: Each overlay method (e.g., `policy_grad`, `value_grad`) maintains its own buffer
- **Partial batch flushing**: Incomplete batches are automatically processed at training end
- **Backward compatibility**: Non-gradient methods (heuristic, random) use sequential fallback automatically
- **PPO-only**: Batching is currently only available for PPO training

### Example: Batched PPO Training

```python
from ungar.training.train_ppo import train_ppo
from ungar.training.config import PPOConfig, XAIConfig

xai = XAIConfig(
    enabled=True,
    methods=["policy_grad", "value_grad"],
    every_n_episodes=5,
    batch_size=8,  # Batch up to 8 overlays
)

config = PPOConfig(
    total_episodes=100,
    xai=xai,
)

result = train_ppo("high_card_duel", config=config, run_dir="runs/")
```

### Limitations

- **PPO-only**: DQN integration is not yet supported (future milestone)
- **Gradient methods**: Batching only benefits gradient-based methods (`policy_grad`, `value_grad`)
- **Memory**: Larger batch sizes require more GPU memory

### When to Use Batching

**Use batching (`batch_size > 1`) when:**
- Generating many overlays per training run
- Using gradient-based methods on GPU
- Training time is a concern

**Keep sequential (`batch_size = None`) when:**
- Generating few overlays (`max_overlays_per_run < 10`)
- Debugging overlay generation
- Maximum transparency is needed




