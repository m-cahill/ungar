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

### Gradient Method Details

*   **Scalar Target:**
    *   **DQN:** $Q(s, a_{taken})$
    *   **PPO:** Logit of $a_{taken}$
*   **Aggregation:** Gradients are summed across feature planes (channels) to produce a single scalar per card position.
*   **Normalization:** $L_1$ norm (sum to 1). Magnitudes are absolute.




