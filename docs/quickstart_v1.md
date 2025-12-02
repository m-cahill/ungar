# UNGAR v1 Quickstart Guide

**Goal:** Get training, analyzing, and visualizing card game AI in 15-30 minutes.

**Prerequisites:**
- Python 3.10+ 
- Git
- 5GB disk space

---

## 1. Installation (5 minutes)

### Clone and Install

```bash
# Clone repository
git clone https://github.com/m-cahill/ungar.git
cd ungar

# Install development dependencies
python -m pip install -r requirements-dev.txt

# Install UNGAR (editable mode)
python -m pip install -e .

# Install visualization dependencies (optional)
python -m pip install -e ".[viz]"

# Verify installation
python -c "import ungar; print(f'UNGAR {ungar.__version__} installed')"
ungar --version
```

**Expected output:**
```
UNGAR installed
UNGAR 0.1.0 (commit ...)
```

---

## 2. First Training Run (5 minutes)

### Train Without XAI (Fast Baseline)

```bash
# Train PPO on High Card Duel for 10 episodes
ungar train \
    --game high_card_duel \
    --algo ppo \
    --episodes 10 \
    --seed 42
```

**What happens:**
- Training runs for ~10 seconds
- Creates `runs/{timestamp}_high_card_duel_ppo_{run_id}/`
- Generates `manifest.json` and `metrics.csv`

**Check results:**

```bash
# List all runs
ungar list-runs

# Show run details
ungar show-run {run_id}
```

---

## 3. Run M22 XAI Demo (10 minutes)

### Full Cycle Validation

```bash
# Run comprehensive M22 demo with batched XAI
python scripts/demo_m22_full_cycle.py --clean
```

**What it does:**
1. **Training** — PPO with batched gradient overlays (20 episodes)
2. **Validation** — Verifies batch == sequential (< 1e-6 tolerance)
3. **CLI** — Tests list-runs, show-run, summarize-overlays
4. **Visualization** — Generates heatmap PNG
5. **Summary** — Outputs validation JSON

**Expected output:**
```
======================================================================
M22 BATCH OVERLAY ENGINE — FULL CYCLE DEMO
======================================================================

PHASE 1: Training PPO with Batched XAI
✅ Training complete
✅ Generated 40 overlay files

PHASE 2: Numerical Equivalence Validation
✅ Numerical equivalence validated (max diff: 0.00e+00 < 1.00e-06)

PHASE 3: CLI Commands Validation
✅ Loaded 40 overlays (20 policy, 20 value)

PHASE 4: Heatmap Visualization
✅ Heatmap generated

DEMO COMPLETE ✅
```

**Inspect results:**
- Summary: `runs/{run_dir}/demo_m22_summary.json`
- Heatmap: `runs/{run_dir}/mean_heatmap.png`

See **[docs/demo_m22.md](demo_m22.md)** for complete documentation.

---

## 4. Train With XAI (Your First Real Run)

### Enable Gradient Overlays

```bash
ungar train \
    --game high_card_duel \
    --algo ppo \
    --episodes 100 \
    --xai-enabled \
    --xai-methods policy_grad value_grad \
    --xai-batch-size 4 \
    --xai-every-n-episodes 10 \
    --seed 1234
```

**What's new:**
- `--xai-enabled` — Activates XAI overlay generation
- `--xai-methods policy_grad value_grad` — Gradient-based introspection
- `--xai-batch-size 4` — Batched processing (M22, 5-10× faster)

**Training time:** ~1-2 minutes (100 episodes)

---

## 5. Analyze Results (5 minutes)

### Load and Visualize Overlays

```bash
# Find your run
ungar list-runs

# Aggregate overlays
ungar summarize-overlays \
    --run runs/{your_run_id} \
    --out-dir analysis/ \
    --label policy_grad \
    --agg mean
```

**Output:**
- `analysis/overlay_aggregated.json` — Raw aggregated data
- `analysis/overlay_heatmap.png` — Visual heatmap

**Open the heatmap** to see which cards your agent considers important!

### Compare Actor vs Critic

```bash
# Compare policy vs value gradients
ungar compare-overlays \
    --run runs/{your_run_id} \
    --label-a policy_grad \
    --label-b value_grad \
    --out actor_vs_critic.json
```

---

## Troubleshooting

### Import Error: "No module named 'ungar'"

**Fix:**
```bash
python -m pip install -e .
```

### CLI Error: "Unknown method 'foo'"

**Fix:** Use valid XAI methods: `heuristic`, `random`, `policy_grad`, `value_grad`

```bash
ungar train --xai-methods heuristic policy_grad value_grad
```

### Demo Fails: "Matplotlib not available"

**Fix:** Install visualization dependencies:
```bash
python -m pip install 'ungar[viz]'
```

**Note:** Demo still validates data without matplotlib (visualization skipped).

### Training Slow on CPU

**Tip:** Enable batch processing for faster XAI:
```bash
--xai-batch-size 8  # Process 8 overlays at once
```

**GPU:** Use `--device cuda` for ~10× speedup:
```bash
ungar train --game high_card_duel --algo ppo --device cuda
```

---

## Next Steps

### Learn More

**Core Concepts:**
- [API Contracts v1](api_contracts_v1.md) — Stable interfaces and guarantees
- [XAI Overlays Guide](xai_overlays.md) — Complete XAI documentation
- [Analytics Schema](analytics_schema.md) — Data formats and validation

**Advanced Usage:**
- [Training DQN](training_dqn.md) — Deep Q-Network training
- [Training PPO](training_ppo.md) — Proximal Policy Optimization
- [CLI Reference](cli_reference.md) — Complete command documentation

**Architecture:**
- [ADR-001](adr/ADR-001-analytics_schema_v1_freeze.md) — Analytics design
- [ADR-002](adr/ADR-002-xai_overlay_engine.md) — XAI architecture
- [ADR-003](adr/ADR-003-opt_in_batch_processing.md) — Batch engine design

### Try Different Games

```bash
# Mini Spades (trick-taking game)
ungar train --game spades_mini --algo ppo --episodes 50

# Gin Rummy (complex meld-based game)
ungar train --game gin_rummy --algo dqn --episodes 200
```

### Profile Performance

```bash
# Benchmark batch vs sequential overlay generation
python scripts/profile_xai_batch.py \
    --batch-size 8 \
    --num-overlays 100
```

---

## Summary

**You've learned:**
✅ How to install UNGAR  
✅ How to run training with and without XAI  
✅ How to validate with the M22 demo  
✅ How to analyze and visualize results  
✅ Where to find documentation

**Total time:** ~30 minutes

**Ready for:** Production training runs, custom XAI experiments, multi-game research

---

**Questions?** Check [GitHub Issues](https://github.com/m-cahill/ungar/issues) or create a new one.

**Version:** 1.0  
**UNGAR:** 0.1.0

