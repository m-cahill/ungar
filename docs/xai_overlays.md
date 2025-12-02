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




