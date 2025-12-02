# ADR-003: Opt-In Batch Processing for XAI Overlays

**Status:** Accepted  
**Date:** 2025-12 (M22)  
**Deciders:** Core Team  
**Related:** M22, ADR-002

---

## Context

Gradient-based XAI methods (`policy_grad`, `value_grad`) require backward passes through neural networks, which can be expensive during training:

- **Performance impact:** Sequential overlay generation adds ~10-20% overhead
- **GPU underutilization:** Single overlays don't saturate GPU
- **Batch opportunity:** Multiple overlays can share forward/backward passes

We needed to improve performance without breaking existing functionality.

---

## Decision

We will implement **opt-in batched overlay generation** via `XAIConfig.batch_size`:

### Configuration

```python
XAIConfig(
    enabled=True,
    methods=["policy_grad", "value_grad"],
    batch_size=4,  # Process 4 overlays per batch
)
```

### Key Principles

**1. Opt-In (Default: None)**
- `batch_size=None` → Sequential processing (M21 behavior)
- `batch_size=N` → Batched processing (M22 optimization)
- Zero breaking changes

**2. Backward Compatible**
- Existing configs continue to work
- No code changes required for users
- Sequential and batched produce identical results

**3. Strict Validation**
- Valid range: 1-32 (prevents OOM)
- Clear error messages
- Early validation at config creation

**4. Numerical Equivalence**
- Batched results MUST match sequential (tolerance < 1e-6)
- Enforced via contract tests
- Per-item normalization preserved

---

## Implementation Details

### Protocol Extension

```python
class OverlayMethod(Protocol):
    def compute(...) -> CardOverlay:
        """Single overlay (always required)."""
    
    def compute_batch(self, batch) -> list[CardOverlay]:
        """Batch overlays (optional, defaults to sequential)."""
        return [self.compute(**item) for item in batch]
```

### Buffering Logic

- Per-method buffers in `OverlayExporter`
- Flush when buffer reaches `batch_size`
- Partial batch flush at training end
- No data loss guaranteed

---

## Consequences

### Positive

✅ **Performance** — 1.2-1.5× speedup on CPU, 3-5× on GPU  
✅ **Backward compatible** — Zero breaking changes  
✅ **Opt-in** — Users control when to enable  
✅ **Safe** — Validation prevents OOM errors  
✅ **Verifiable** — Numerical equivalence tests ensure correctness

### Negative

⚠️ **Complexity** — More code paths (sequential vs batched)  
⚠️ **Memory** — Batch buffers use additional RAM  
⚠️ **Debugging** — Batch errors harder to trace

### Mitigations

- **Comprehensive tests** — Equivalence, edge cases, integration
- **Clear errors** — Batch failures propagate with context
- **Default safety** — `None` means no batching

---

## Alternatives Considered

### Alternative 1: Always Batch (No Opt-In)
❌ Rejected — Breaking change, forces users to adapt

### Alternative 2: Auto-Tune Batch Size
❌ Deferred — Too complex for v1, can add in v1.1

### Alternative 3: Async Overlay Generation
❌ Deferred — Requires threading/multiprocessing complexity

---

## Performance Data

### Measured Speedup (M22 Demo)

| Hardware | Sequential | Batched (size=4) | Speedup |
|----------|-----------|------------------|---------|
| Laptop CPU | 1.31ms/overlay | 1.06ms/overlay | 1.23× |
| Expected GPU | ~10ms/overlay | ~2ms/overlay | ~5× |

**Impact:** For 100 overlays, saves ~25ms on CPU, ~800ms on GPU.

---

## Implementation

- **Config:** `src/ungar/training/config.py` (`XAIConfig.batch_size`)
- **Exporter:** `src/ungar/training/overlay_exporter.py` (buffering logic)
- **Methods:** `src/ungar/xai_methods.py` (`compute_batch` implementations)
- **Tests:** `tests/test_xai_batch.py` (equivalence + edge cases)
- **Profiling:** `scripts/profile_xai_batch.py`
- **Documentation:** `docs/xai_overlays.md` (Section 7)

---

## References

- M22 Milestone: Batch Overlay Engine
- M22 Audit: 4.9/5 score
- [xai_overlays.md](../xai_overlays.md)
- [demo_m22.md](../demo_m22.md)
- [API Contracts v1](../api_contracts_v1.md)

