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
*   *(Future)* `policy_grad`: Saliency from policy gradient.
*   *(Future)* `value_grad`: Saliency from value function.

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

## 4. Integration Guide

### Adding a New Method

1.  Create a class implementing the `OverlayMethod` protocol:

    ```python
    class MyMethod:
        label = "my_method"

        def compute(self, obs, action, *, step, run_id, meta=None) -> CardOverlay:
            # ... logic to produce 4x14 importance matrix ...
            return CardOverlay(...)
    ```

2.  Register it in the training loop (currently hardcoded in `train_dqn.py` / `train_ppo.py` factory logic, to be dynamic in M20).



