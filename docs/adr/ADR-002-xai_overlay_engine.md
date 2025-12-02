# ADR-002: XAI Overlay Engine with Heuristic & Random Baselines

**Status:** Accepted  
**Date:** 2024-11 (M19)  
**Deciders:** Core Team  
**Related:** M19, M20, M21, M22

---

## Context

UNGAR needed an Explainable AI (XAI) system to understand what card game agents learn. Requirements:

- **Interpretability:** Visualize which cards matter for decisions
- **Comparison:** Compare different XAI methods (gradient vs heuristic)
- **Performance:** Fast enough for training-time generation
- **Extensibility:** Easy to add new XAI methods

We needed to choose an architecture and baseline methods.

---

## Decision

We will implement a **plug-and-play overlay engine** with:

### Core Abstraction

**CardOverlay:** 4×14 importance map matching card tensor shape

```python
@dataclass
class CardOverlay:
    run_id: str
    label: str              # Method identifier
    agg: str                # Aggregation type
    step: int               # Training step
    importance: np.ndarray  # (4, 14) float array
    meta: dict | None       # Extensible metadata
```

### Initial Methods (M19)

1. **Heuristic (`heuristic`):**
   - Highlights cards in player's hand
   - Simple, interpretable baseline
   - No model required

2. **Random (`random`):**
   - Uniform random importance
   - Control baseline for comparison
   - Tests visualization pipeline

### Design Principles

- **Protocol-based:** Methods implement `OverlayMethod` protocol
- **Stateless:** Methods are pure functions of (observation, action)
- **Normalized:** Importance maps are L1-normalized (sum ≈ 1.0)
- **Serializable:** JSON wire format for storage

---

## Consequences

### Positive

✅ **Extensible** — New methods added via protocol implementation  
✅ **Testable** — Baselines provide comparison ground truth  
✅ **Portable** — JSON format enables cross-tool analysis  
✅ **Fast** — Heuristic/random methods have negligible overhead

### Negative

⚠️ **Fixed shape** — All overlays must be 4×14 (no per-game customization yet)  
⚠️ **Storage cost** — JSON per overlay can grow large

### Future Enhancements (Delivered)

- **M20:** Gradient-based methods (`policy_grad`)
- **M21:** Value gradients (`value_grad` for PPO)
- **M22:** Batch processing for gradient methods

---

## Alternatives Considered

### Alternative 1: Attention Mechanisms
❌ Rejected — Requires model architecture changes

### Alternative 2: SHAP/LIME
❌ Deferred — Too slow for training-time generation

### Alternative 3: Custom per-game overlays
❌ Rejected — Breaks cross-game comparison

---

## Implementation

- **Core:** `src/ungar/xai.py` (CardOverlay dataclass)
- **Methods:** `src/ungar/xai_methods.py`
- **Exporter:** `src/ungar/training/overlay_exporter.py`
- **Tests:** `tests/test_xai_methods.py`
- **Documentation:** `docs/xai_overlays.md`

---

## References

- M19 Milestone: XAI Overlay Engine v1
- [xai_overlays.md](../xai_overlays.md)
- [API Contracts v1](../api_contracts_v1.md)

