

# M22 Batch Overlay Engine — Full Cycle Demo

**Status:** Production Ready  
**Milestone:** M22 (Batch Overlay Engine for PPO Gradient XAI)  
**Purpose:** End-to-end validation and demonstration of UNGAR's batched XAI overlay generation

---

## Overview

This demo validates the complete M22 batch overlay engine implementation through a 6-phase workflow:

1. **Training** — PPO on High Card Duel with batched XAI enabled
2. **Numerical Equivalence** — Verify batch vs sequential overlay correctness
3. **CLI Validation** — Test command-line interface end-to-end
4. **Visualization** — Generate and validate heatmap outputs
5. **Performance Profiling** — Measure batch speedup (optional)
6. **Summary Report** — JSON output with all validation results

---

## Quick Start

### Basic Demo

```bash
# Run the full demo (Phases 1-4 + 6)
python scripts/demo_m22_full_cycle.py
```

**Expected output:**
- Training runs for 20 episodes (~30 seconds)
- Overlays generated in `runs/demo_m22/overlays/`
- Heatmap PNG in `runs/demo_m22/mean_heatmap.png`
- Summary JSON in `runs/demo_m22/demo_m22_summary.json`

### With Profiling

```bash
# Include performance benchmarking (Phase 5)
python scripts/demo_m22_full_cycle.py --with-profiling
```

###  Ephemeral Mode

```bash
# Clean up run directory after completion
python scripts/demo_m22_full_cycle.py --clean
```

---

## Phase Breakdown

### Phase 1: Training with Batched XAI

**Goal:** Train PPO agent with M22 batch overlay generation enabled.

**Configuration:**
- Game: High Card Duel (fast, deterministic)
- Algorithm: PPO (supports both policy_grad and value_grad)
- Episodes: 20
- XAI Methods: `policy_grad`, `value_grad`
- Batch Size: 4
- Overlay Frequency: Every episode (`every_n_episodes=1`)
- Max Overlays: 50
- Seed: 1234 (reproducible)

**Validates:**
- M22 batch buffer logic
- `compute_batch()` override for gradient methods
- Exporter flush behavior (partial batches)
- Per-method buffer management

**Success Criteria:**
- ✅ Training completes without errors
- ✅ Overlays directory created
- ✅ Overlay count > 0

---

### Phase 2: Numerical Equivalence Validation

**Goal:** Prove batch and sequential overlay generation produce identical results.

**Method:**
1. Create fresh PPO agent
2. Generate sample observation from High Card Duel
3. Compute overlay via `method.compute()` (sequential)
4. Compute overlay via `method.compute_batch([...])` (batched)
5. Compare importance maps with strict tolerance

**Validates:**
- Core M22 promise: numerical equivalence
- Batch gradient computation correctness
- Normalization consistency

**Success Criteria:**
- ✅ `max(|batch - sequential|) < 1e-6` for policy_grad
- ✅ `max(|batch - sequential|) < 1e-6` for value_grad

**Failure Mode:**
- `AssertionError` if difference exceeds tolerance

---

### Phase 3: CLI Commands Validation

**Goal:** Ensure UNGAR's CLI works end-to-end with M22 outputs.

**Tests:**
1. **`list-runs`** — Verify run appears in listing
2. **`show-run`** — Load and display manifest.json
3. **`summarize-overlays`** — Aggregate policy/value overlays programmatically

**Validates:**
- Overlay schema v1 compliance
- Analytics loader compatibility
- Run directory structure

**Success Criteria:**
- ✅ Manifest loads successfully
- ✅ Overlays load via `load_overlays()`
- ✅ Mean aggregation produces correct shape (4×14)

---

### Phase 4: Heatmap Visualization

**Goal:** Generate visual confirmation of overlay correctness.

**Method:**
1. Load all overlays from run directory
2. Compute mean overlay across all methods
3. Validate shape (4×14) and L1 normalization
4. Plot heatmap using `plot_overlay_heatmap()`

**Validates:**
- Visualization pipeline
- Overlay data integrity
- Matplotlib integration (graceful degradation if unavailable)

**Success Criteria:**
- ✅ Mean overlay shape == (4, 14)
- ✅ L1 normalization: 0.9 < sum < 1.1
- ✅ PNG file generated (if matplotlib available)

**Output:**
- `runs/demo_m22/mean_heatmap.png`

---

### Phase 5: Performance Profiling (Optional)

**Goal:** Empirically measure batch speedup.

**Method:**
- Runs `scripts/profile_xai_batch.py`
- Batch size: 4
- Overlays: 100
- Methods: Both policy and value gradients

**Validates:**
- M22 performance claims
- Batch engine efficiency

**Expected Speedup:**
- CPU: ~1.2-1.5×
- GPU: ~3-5×

**Note:** Excluded from CI (manual dev tool).

---

### Phase 6: Summary Report

**Goal:** Programmatic validation output for CI/testing.

**Output:** `demo_m22_summary.json`

```json
{
  "demo": "M22 Full Cycle",
  "run_dir": "runs/demo_m22",
  "num_overlays": 37,
  "max_diff_seq_vs_batch": 3.1e-7,
  "numerical_equivalence_tolerance": 1e-6,
  "numerical_equivalence_passed": true,
  "heatmap_path": "runs/demo_m22/mean_heatmap.png",
  "profiling": null
}
```

---

## CLI Training with XAI (Manual)

The demo script uses the Python API, but you can also train with XAI via CLI:

```bash
ungar train \
    --game high_card_duel \
    --algo ppo \
    --episodes 20 \
    --xai-enabled \
    --xai-methods policy_grad value_grad \
    --xai-batch-size 4 \
    --xai-every-n-episodes 1 \
    --xai-max-overlays 50 \
    --seed 1234
```

Then analyze:

```bash
# List all runs
ungar list-runs

# Show specific run
ungar show-run demo_m22

# Summarize overlays
ungar summarize-overlays --run runs/demo_m22 --out-dir analysis/
```

---

## Validation Criteria

### All Phases Must Pass

| Phase | Criterion | Tolerance |
|-------|-----------|-----------|
| 1 | Training completes | Hard fail |
| 1 | Overlays generated | Count > 0 |
| 2 | Numerical equivalence | < 1e-6 |
| 3 | Manifest loadable | Hard fail |
| 3 | Overlays loadable | Count > 0 |
| 4 | Mean overlay shape | (4, 14) |
| 4 | L1 normalization | 0.9 < sum < 1.1 |
| 6 | Summary JSON | Valid format |

---

## CI Integration

The demo is integrated into the **full test suite** (not smoke tests):

```yaml
# .github/workflows/test.yml (example)
- name: Run M22 Demo
  run: python scripts/demo_m22_full_cycle.py --clean
```

**Why full tier, not smoke?**
- Training takes ~30 seconds (too slow for smoke)
- Validates entire XAI pipeline (thorough check)
- Profiling excluded from CI (manual only)

---

## Troubleshooting

### Demo Fails: "No overlays were generated!"

**Cause:** XAI config not properly passed to training.

**Fix:**
- Check `xai.enabled = True`
- Verify `xai.methods` is non-empty
- Ensure `xai.batch_size` is valid (1-32 or None)

### Demo Fails: "Numerical equivalence failed"

**Cause:** Batch and sequential produce different results.

**Debug:**
1. Check gradient computation in `xai_grad.py`
2. Verify `compute_batch()` implementation
3. Ensure no random state differences

### Demo Fails: "Matplotlib not available"

**Cause:** Visualization dependencies not installed.

**Fix:**
```bash
pip install 'ungar[viz]'
```

**Note:** Demo still passes without viz (Phase 4 skipped).

---

## Performance Expectations

### Training Time

| Hardware | Time (20 episodes) |
|----------|-------------------|
| Laptop CPU | ~30-40 seconds |
| Desktop CPU | ~20-30 seconds |
| GPU | ~10-15 seconds |

### Overlay Generation

| Batch Size | Overlays | Time (CPU) | Speedup |
|-----------|----------|------------|---------|
| None (seq) | 100 | ~1.3 seconds | 1.0× |
| 4 | 100 | ~1.0 seconds | 1.3× |
| 8 | 100 | ~0.9 seconds | 1.4× |

**GPU:** Expect 3-5× speedup with larger batch sizes.

---

## What This Demo Proves

✅ **M22 Batch Engine Works:**  
Overlays generated via batching match sequential (< 1e-6 difference).

✅ **Full Pipeline Integration:**  
Training → Exporter → CLI → Analytics → Visualization all functional.

✅ **Performance Improvement:**  
Batch processing provides measurable speedup (CPU: ~1.2-1.5×, GPU: 3-5×).

✅ **Production Ready:**  
Zero breaking changes, backward compatible, comprehensive validation.

---

## Next Steps

After running the demo:

1. **Inspect heatmap:** `runs/demo_m22/mean_heatmap.png`
2. **Review summary:** `runs/demo_m22/demo_m22_summary.json`
3. **Try different configs:** Modify batch size, methods, games
4. **Profile on GPU:** Run with `--with-profiling` on GPU hardware
5. **Integrate into workflow:** Use demo as template for custom XAI experiments

---

## Related Documentation

- [XAI Overlays Guide](xai_overlays.md) — Complete XAI system documentation
- [Analytics Schema](analytics_schema.md) — Overlay file format specification
- [CLI Quickstart](cli_quickstart.md) — Command-line interface guide
- [M22 Audit](../Project Files/Milestones/Batch3/M_22_audit.md) — Implementation quality review
- [M22 Summary](../Project Files/Milestones/Batch3/M_22_summary.md) — Milestone completion details

---

## Feedback & Issues

Found a bug or have suggestions?
1. Check existing issues: https://github.com/m-cahill/ungar/issues
2. Create new issue with `[M22 Demo]` prefix
3. Include `demo_m22_summary.json` output

---

**Demo Version:** 1.0  
**Last Updated:** 2025-12-02  
**Milestone:** M22 Complete

