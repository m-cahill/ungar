# UNGAR Analytics Schema v1

**Version:** 1.0
**Status:** Frozen (M17)

This document defines the contract for UNGAR v1 analytics artifacts. All downstream tools (frontends, analysis scripts, RediAI bridges) can rely on these invariants.

## 1. Run Manifest (`manifest.json`)

Located at the root of every run directory.

### Schema

```json
{
  "run_id": "string (unique identifier)",
  "game": "string (e.g. 'high_card_duel', 'gin_rummy')",
  "algo": "string (e.g. 'dqn', 'ppo')",
  "created_at": "string (ISO 8601 timestamp, e.g. '2025-11-30T12:34:56Z')",
  "analytics_schema_version": 1,
  "config": { ... },
  "device": "string (e.g. 'cpu', 'cuda:0')",
  "metrics_path": "string (relative path, usually 'metrics.csv')",
  "overlays_path": "string (relative path, usually 'overlays')",
  "timestamp": "float (unix timestamp, optional but recommended)",
  "notes": "string | null (optional)",
  "tags": ["string", ...] (optional)
}
```

### Invariants

1.  `analytics_schema_version` must be present. If missing, assume version 0 (pre-M17) or invalid.
2.  `created_at` is the authoritative human-readable timestamp.
3.  `config` must be a dictionary (contents are algo-specific).

## 2. Metrics (`metrics.csv`)

A CSV file containing time-series training data.

### Required Columns

| Column | Type | Description |
| :--- | :--- | :--- |
| `step` | int | Global training step (monotonically increasing) |
| `episode` | int | Episode number (1-based) |
| `reward` | float | Episode return (total reward) |

### Optional Standard Columns

| Column | Type | Description |
| :--- | :--- | :--- |
| `loss` | float | Training loss (may be empty/NaN for evaluation steps) |
| `epsilon` | float | Current exploration rate |
| `episode_length` | float | Number of steps in the episode |

### Invariants

1.  Header row must be present.
2.  Rows must be sorted by `step` ascending.
3.  Extra columns are allowed and should be ignored by strict parsers.

## 3. Overlays (`overlays/*.json`)

XAI attribution maps stored as JSON files in the `overlays/` subdirectory.

### Schema

```json
{
  "run_id": "string",
  "label": "string (e.g. 'policy', 'value', 'grad', 'integrated_gradients')",
  "agg": "string (e.g. 'mean', 'max', 'none')",
  "step": "int (optional, if specific to a step)",
  "importance": [
    [... 14 floats ...],
    [... 14 floats ...],
    [... 14 floats ...],
    [... 14 floats ...]
  ],
  "meta": {
    "game": "string",
    "algo": "string",
    "version": 1,
    "comment": "string (optional)"
  }
}
```

### Invariants

1.  `importance` is strictly a 4x14 matrix (list of 4 lists, each containing 14 floats).
2.  Rows correspond to suits: [SPADES, HEARTS, DIAMONDS, CLUBS].
3.  Columns correspond to ranks: [ACE, TWO, ..., KING, JOKER].
4.  Values are normalized (interpretations vary by method, but structure is fixed).

### Standard Labels

*   `heuristic`: Rule-based importance (e.g., cards in hand).
*   `random`: Noise (for baseline/testing).
*   `policy_grad`: Gradient/saliency w.r.t policy output (or Q-value). Supported for DQN and PPO.
*   `value_grad`: Gradient/saliency w.r.t critic/value output V(s). Supported for PPO only (M21).

### Aggregation Semantics

*   `agg`: Describes how this overlay was derived from others.
    *   `none`: Raw single-step overlay.
    *   `mean`: Pixel-wise average of multiple overlays.
    *   `max`: Pixel-wise maximum.

## 4. Versioning Policy

*   **Current Version:** 1
*   **Backwards Compatibility:** Minor additions (new optional fields) do not require a version bump.
*   **Breaking Changes:** Any change to required fields, shapes (e.g., changing 4x14 to 4x15), or file formats requires incrementing `analytics_schema_version`.

