# Analytics & Visualization

M15 introduces tools for analyzing training runs, plotting learning curves, and aggregating XAI overlays. M16 wraps these in the unified `ungar` CLI.

## Run Directory Structure

Training runs create structured output directories:

```text
runs/
  <timestamp>_<game>_<algo>_<id>/
    manifest.json      # Metadata (config, device, etc.)
    config.json        # Full algorithm configuration
    metrics.csv        # Training log (steps, rewards, losses)
    overlays/          # XAI overlay artifacts (JSON)
```

## Usage

### Training with Logging

```bash
# Will create a run directory in runs/
ungar train --game high_card_duel --algo dqn --run-dir runs/my_experiment
```

### Analyzing Metrics

You can load metrics programmatically:

```python
from ungar.analysis.metrics import load_metrics

df = load_metrics("runs/1700000000_gin_rummy_dqn_a1b2c3")
print(df.rewards)
```

### Plotting

Requires `matplotlib` (`pip install ungar[viz]`).

**CLI:**
```bash
ungar plot-curves --run runs/my_run --out curve.png
```

**Python:**
```python
from ungar.analysis.plots import plot_learning_curve

# Plot single run
plot_learning_curve(["runs/my_run"], out_path="curve.png")

# Compare runs
plot_learning_curve(["runs/dqn_run", "runs/ppo_run"], out_path="comparison.png")
```

### Aggregating Overlays

Summarize card importance over an entire training run:

**CLI:**
```bash
ungar summarize-overlays --run runs/my_run --out-dir heatmaps/
```

**Python:**
```python
from ungar.analysis.overlays import load_overlays, aggregate_overlays
from ungar.analysis.plots import plot_overlay_heatmap

overlays = load_overlays("runs/my_run")
mean_map = aggregate_overlays(overlays, method="mean")

plot_overlay_heatmap(mean_map, out_path="heatmap.png")
```
