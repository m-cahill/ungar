# UNGAR v1 Core API Contracts

**Version:** 1.0  
**Status:** Stable  
**Last Updated:** 2025-12-02  
**Schema Version:** `analytics_schema_version=1`

---

## Overview

This document defines the **v1 core contract** for UNGAR (Universal Neural Grid for Analysis and Research). These interfaces are considered **stable** and **production-ready**. Future changes to these contracts will be treated as potentially breaking and will require appropriate version bumps according to Semantic Versioning.

**Contract Guarantee:**  
Within the `v1.x` series:
- **Breaking changes** will increment the MAJOR version (v1 → v2)
- **Backward-compatible additions** may increment the MINOR version (v1.0 → v1.1)
- **Bug fixes** will increment the PATCH version (v1.0.0 → v1.0.1)

**Covered Systems:**
1. **Representation** — 4×14×n card tensor
2. **XAI (Explainable AI)** — CardOverlay structure and methods
3. **Analytics** — Run directory structure and file formats
4. **CLI** — Command-line interface

---

## 1. Representation Contract

###  1.1 Card Tensor (4×14×n)

**Core Invariant:**  
All card states in UNGAR are represented as a **4×14×n** NumPy tensor:

- **Axis 0 (Suits):** 4 suits (Spades, Hearts, Diamonds, Clubs)
- **Axis 1 (Ranks):** 14 ranks (2-10, J, Q, K, A, Joker)
- **Axis 2 (Planes):** N feature planes (game-defined)

**Python API:**

```python
from ungar.tensor import CardTensor
from ungar.cards import Card
from ungar.enums import Suit, Rank

# Create tensor from cards
hand = [Card(Suit.SPADES, Rank.ACE), Card(Suit.HEARTS, Rank.KING)]
tensor = CardTensor.from_plane_card_map({"my_hand": hand})

# Access shape
assert tensor.data.shape[0] == 4  # suits
assert tensor.data.shape[1] == 14  # ranks
# tensor.data.shape[2] varies by game

# Query cards in a plane
cards_in_hand = tensor.cards_in_plane("my_hand")
assert set(cards_in_hand) == set(hand)
```

**Standard Deck (4×13):**  
The first 13 columns (indices 0-12) represent the standard 52-card deck. Column 13 (index 13) is reserved for jokers.

**Contract:**
- Suit order: `[Spades=0, Hearts=1, Diamonds=2, Clubs=3]`
- Rank order: `[Two=0, Three=1, ..., King=11, Ace=12, Joker=13]`
- Dtype: `bool` or `float32` (bool for presence, float for features)

---

## 2. XAI (Explainable AI) Contract

### 2.1 CardOverlay Structure

**Definition:**  
A `CardOverlay` is a 4×14 importance map representing the saliency or attribution of each card position for a model's decision.

**Python API:**

```python
from ungar.xai import CardOverlay
import numpy as np

overlay = CardOverlay(
    run_id="my_run",
    label="policy_grad",
    agg="none",
    step=100,
    importance=np.random.rand(4, 14),  # Must be (4, 14) shape
    meta={"game": "high_card_duel", "algo": "ppo"}
)
```

**Required Fields:**
- `run_id` (str): Identifier for the training run
- `label` (str): Overlay method (e.g., "policy_grad", "value_grad", "heuristic")
- `agg` (str): Aggregation type ("none", "mean", "max", "diff")
- `step` (int): Training step number (-1 for aggregated overlays)
- `importance` (np.ndarray): Shape **(4, 14)** importance values
- `meta` (dict | None): Additional metadata

**JSON Wire Format:**

```json
{
  "run_id": "my_run",
  "label": "policy_grad",
  "agg": "none",
  "step": 100,
  "importance": [[0.1, 0.2, ...], ...],  // 4×14 nested array
  "meta": {
    "game": "high_card_duel",
    "algo": "ppo",
    "method": "policy_grad",
    "target_type": "q_value"
  }
}
```

**Contract:**
- `importance` shape MUST be `(4, 14)`
- For gradient-based methods, values are typically L1-normalized (sum ≈ 1.0)
- File naming: `{label}_{step}.json`

### 2.2 Standard XAI Methods

**v1 Core Methods:**
1. **`heuristic`** — Hand-based importance (simple baseline)
2. **`random`** — Random importance values (control baseline)
3. **`policy_grad`** — Gradient of policy/Q-value w.r.t. input
4. **`value_grad`** — Gradient of state-value V(s) w.r.t. input (PPO only)

**Method Labels Contract:**  
These labels are **reserved** in v1.x. Custom methods should use descriptive prefixes (e.g., `custom_attention`, `shap_policy`).

**Adding New Methods:**  
Implement the `OverlayMethod` protocol:

```python
from ungar.xai import CardOverlay
import numpy as np

class CustomOverlayMethod:
    label = "custom_method"
    
    def compute(
        self,
        obs: np.ndarray,
        action: int,
        *,
        step: int,
        run_id: str,
        meta: dict | None = None,
    ) -> CardOverlay:
        # Your logic here
        importance = np.zeros((4, 14))
        return CardOverlay(
            run_id=run_id,
            label=self.label,
            agg="none",
            step=step,
            importance=importance,
            meta=meta,
        )
    
    def compute_batch(self, batch):
        # Optional: batch processing
        return [self.compute(**item) for item in batch]
```

### 2.3 XAIConfig Contract

**Configuration:**

```python
from ungar.training.config import XAIConfig

xai = XAIConfig(
    enabled=True,
    methods=["policy_grad", "value_grad"],
    every_n_episodes=10,
    max_overlays_per_run=200,
    batch_size=4,  # M22: Batch overlay generation (1-32 or None)
)
```

**Contract:**
- `enabled` (bool): Enable XAI overlay generation
- `methods` (list[str]): List of overlay method labels
- `every_n_episodes` (int): Generate overlays every N episodes
- `max_overlays_per_run` (int): Maximum overlays per training run
- `batch_size` (int | None): Batch size for gradient methods (1-32 or None)

**Validation:**  
Invalid method names MUST raise `ValueError` with clear message listing valid methods.

---

## 3. Analytics Contract

### 3.1 Run Directory Structure

**Standard Layout:**

```
runs/
└── {timestamp}_{game}_{algo}_{run_id}/
    ├── manifest.json          # Run metadata
    ├── metrics.csv            # Training metrics
    ├── overlays/              # XAI overlays (optional)
    │   ├── policy_grad_0.json
    │   ├── policy_grad_100.json
    │   ├── value_grad_0.json
    │   └── ...
    └── checkpoints/           # Model checkpoints (future)
```

### 3.2 manifest.json Contract

**Required Fields:**

```json
{
  "analytics_schema_version": 1,
  "run_id": "demo_m22",
  "timestamp": 1732000000,
  "created_at": "2025-12-02T12:00:00Z",
  "game": "high_card_duel",
  "algo": "ppo",
  "device": "cpu",
  "config": {
    "total_episodes": 100,
    "batch_size": 32,
    "xai": {
      "enabled": true,
      "methods": ["policy_grad", "value_grad"],
      "batch_size": 4
    }
  }
}
```

**Contract:**
- `analytics_schema_version` MUST equal `1` (for v1.x)
- `run_id`, `timestamp`, `game`, `algo` are REQUIRED
- `config` SHOULD include full training configuration
- `created_at` SHOULD be ISO 8601 timestamp

**Version Guarantee:**  
Within v1.x, the `analytics_schema_version` will remain `1`. Schema changes will bump the version number and may require migration tools.

### 3.3 metrics.csv Contract

**Required Columns:**

```csv
step,episode,reward,loss,value
0,0,1.0,0.5,0.3
1,0,0.0,0.4,0.2
...
```

**Contract:**
- `step` (int): Global training step
- `episode` (int): Episode number
- `reward` (float): Reward for current step/episode

**Optional Columns:**  
Additional columns (e.g., `loss`, `value`, `entropy`) are ALLOWED and will not break schema v1.

**Format:**  
CSV with headers, UTF-8 encoding.

### 3.4 Overlay Files Contract

**Location:** `{run_dir}/overlays/`

**Naming:** `{label}_{step}.json`

**Format:** JSON following CardOverlay wire format (see section 2.1)

**Contract:**
- Each file is a single CardOverlay JSON object
- Files are independent (no ordering dependency)
- Missing overlay files are allowed (e.g., due to sampling)

---

## 4. CLI Contract

### 4.1 Core Commands

#### 4.1.1 `ungar train`

**Purpose:** Start a training run

**Required Arguments:**
- `--game`: Game name (`high_card_duel`, `spades_mini`, `gin_rummy`)
- `--algo`: Algorithm (`dqn`, `ppo`)

**Optional Arguments:**
- `--episodes`: Number of episodes (default: algorithm-specific)
- `--seed`: Random seed for reproducibility
- `--run-dir`: Base directory for runs (default: `./runs`)
- `--device`: Device (`auto`, `cpu`, `cuda`, `mps`)

**XAI Arguments:**
- `--xai-enabled`: Enable XAI overlay generation
- `--xai-methods`: Space-separated list of methods (e.g., `policy_grad value_grad`)
- `--xai-batch-size`: Batch size for overlay generation (1-32)
- `--xai-every-n-episodes`: Generate overlays every N episodes
- `--xai-max-overlays`: Maximum overlays per run

**Example:**

```bash
ungar train \
    --game high_card_duel \
    --algo ppo \
    --episodes 100 \
    --xai-enabled \
    --xai-methods policy_grad value_grad \
    --xai-batch-size 4 \
    --seed 1234
```

**Contract:**
- MUST create a run directory following section 3.1 structure
- MUST generate `manifest.json` with schema v1
- MUST generate `metrics.csv` with required columns
- IF `--xai-enabled`, MUST generate overlays in `overlays/` directory

#### 4.1.2 `ungar list-runs`

**Purpose:** List all training runs

**Optional Arguments:**
- `--format`: Output format (`table`, `json`)

**Contract:**
- MUST list all directories in runs folder
- MUST show `run_id`, `timestamp`, `game`, `algo`, `device`

#### 4.1.3 `ungar show-run`

**Purpose:** Show details for a specific run

**Required Arguments:**
- `run_id`: Run ID or partial directory name

**Contract:**
- MUST load and display `manifest.json`
- MUST support partial ID matching (unique prefix)

#### 4.1.4 `ungar summarize-overlays`

**Purpose:** Aggregate XAI overlays

**Required Arguments:**
- `--run`: Run ID or path
- `--out-dir`: Output directory

**Optional Arguments:**
- `--agg`: Aggregation method (`mean`, `max`)
- `--label`: Filter by overlay label

**Contract:**
- MUST load overlays from `{run}/overlays/`
- MUST output aggregated overlay JSON + heatmap PNG
- MUST preserve (4, 14) shape

#### 4.1.5 `ungar --version`

**Purpose:** Show UNGAR version

**Contract:**
- MUST print version string in format: `UNGAR {version} (commit {hash})`
- Version MUST match `ungar.__version__`

**Example Output:**

```
UNGAR 0.1.0 (commit 2f3eedc)
```

### 4.2 CLI Error Handling

**Invalid XAI Method:**

```
Unknown method 'foo'. Available: heuristic, random, policy_grad, value_grad. 
See docs/xai_overlays.md for details.
```

**Invalid Batch Size:**

```
XAIConfig.batch_size must be between 1 and 32 (inclusive). Got 50.
```

**Run Not Found:**

```
No run found matching 'abc123'
```

**Contract:**  
Error messages MUST be clear, actionable, and include:
- What went wrong
- Valid options (if applicable)
- Where to find more info

---

## 5. Extension Points

### 5.1 Adding New Games

Implement `GameSpec` and register with adapter:

```python
from ungar.game import GameSpec

def make_my_game_spec() -> GameSpec:
    # Define game rules, tensor planes, etc.
    ...
```

**Contract:** Games MUST use the 4×14×n tensor representation.

### 5.2 Adding New XAI Methods

Implement `OverlayMethod` protocol (see section 2.2).

**Contract:** Custom methods SHOULD use unique, descriptive labels.

### 5.3 Adding Metrics Columns

Add columns to `metrics.csv` during training.

**Contract:** New columns are ALLOWED. Required columns (`step`, `episode`, `reward`) MUST remain.

### 5.4 Adding Metadata to Overlays

Add keys to `meta` dictionary.

**Contract:** Custom metadata is ALLOWED. Standard keys (`game`, `algo`, `method`) SHOULD be preserved.

---

## 6. Breaking Change Policy

### 6.1 What Constitutes a Breaking Change?

**Breaking changes** include:
- Changing `analytics_schema_version`
- Removing required fields from `manifest.json` or `metrics.csv`
- Changing CardOverlay shape from (4, 14)
- Removing CLI commands or required arguments
- Changing standard XAI method labels

**Non-breaking changes** include:
- Adding optional fields to manifest or metrics
- Adding new CLI arguments (optional)
- Adding new XAI methods
- Performance improvements

### 6.2 Migration Path

When breaking changes are necessary:
1. Increment MAJOR version (v1 → v2)
2. Provide migration tools or documentation
3. Support both versions during transition period (if feasible)
4. Update `analytics_schema_version` in manifests

---

## 7. Validation & Testing

### 7.1 Contract Tests

UNGAR includes contract tests to prevent accidental breaking changes:

- `tests/test_analytics_schema_contract.py` — Schema version validation
- `tests/test_overlay_contract.py` — Overlay shape and label validation
- `tests/test_cli_contracts.py` — CLI command and flag validation

**CI Enforcement:**  
These tests run in the smoke tier and will **block merges** if contracts are violated.

### 7.2 Validating Your Code

To ensure compliance with v1 contracts:

```bash
# Run contract tests
pytest tests/test_*_contract.py -v

# Run full test suite
pytest

# Validate a run directory
python -c "
from ungar.analysis.schema import validate_manifest, validate_metrics_file
validate_manifest('runs/my_run/manifest.json')
validate_metrics_file('runs/my_run/metrics.csv')
"
```

---

## 8. References

**Related Documentation:**
- [XAI Overlays Guide](xai_overlays.md) — Complete XAI system documentation
- [Analytics Schema](analytics_schema.md) — Detailed schema specification
- [M22 Demo](demo_m22.md) — Full cycle validation demo
- [Quickstart v1](quickstart_v1.md) — Getting started guide
- [CLI Reference](cli_reference.md) — Complete CLI documentation

**Architecture Decision Records:**
- [ADR-001: Analytics Schema v1 Freeze](adr/ADR-001-analytics_schema_v1_freeze.md)
- [ADR-002: XAI Overlay Engine](adr/ADR-002-xai_overlay_engine.md)
- [ADR-003: Opt-in Batch Processing](adr/ADR-003-opt_in_batch_processing.md)
- [ADR-004: Per-Method Buffers](adr/ADR-004-per_method_buffers.md)

---

## 9. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-02 | Initial v1 core contract (M23) |

---

**Questions or Issues?**  
- Check existing GitHub issues: https://github.com/m-cahill/ungar/issues
- Create new issue with `[Contract]` prefix

**Contract Status:** ✅ **STABLE** (v1.0)

