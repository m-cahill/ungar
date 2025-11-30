# UNGAR CLI Quickstart

UNGAR provides a unified command-line interface for training, analysis, and visualization.

## Installation

```bash
# Install with visualization dependencies
pip install -e ".[viz]"
```

## 1. Train an Agent

Train a DQN agent on **Gin Rummy** for 1000 episodes.

```bash
ungar train --game gin_rummy --algo dqn --episodes 1000
```

Output:
```text
Starting DQN training on gin_rummy...
Training complete.
Run directory: runs/1732000000_gin_rummy_dqn_a1b2c3
```

## 2. Inspect Runs

List all recorded experiments:

```bash
ungar list-runs
```

Show details for a specific run:

```bash
ungar show-run a1b2c3
```

## 3. Visualize Learning

Plot the learning curve for your run:

```bash
ungar plot-curves --run runs/1732000000_gin_rummy_dqn_a1b2c3 --out learning_curve.png
```

## 4. Analyze XAI Overlays

Generate an aggregated heatmap of card importance:

```bash
ungar summarize-overlays --run runs/1732000000_gin_rummy_dqn_a1b2c3 --out-dir analysis/
```

This produces:
*   `analysis/overlay_aggregated.json`: Raw data
*   `analysis/overlay_heatmap.png`: Visual heatmap

