# ADR-004: Per-Method Buffers for Batch Processing

**Status:** Accepted  
**Date:** 2025-12 (M22)  
**Deciders:** Core Team  
**Related:** M22, ADR-003

---

## Context

With M22's batch processing (ADR-003), we needed to decide how to buffer overlay requests before flushing them to `compute_batch()`:

**Options:**
1. **Single global buffer** — All methods share one buffer
2. **Per-method buffers** — Each method has independent buffer
3. **Per-label buffers** — Group by label (e.g., all "policy_grad" together)

**Constraints:**
- Different methods may have different batch performance characteristics
- Methods should not block each other
- Partial batch flushing must work cleanly

---

## Decision

We will use **independent per-method buffers** in `OverlayExporter`:

### Implementation

```python
class OverlayExporter:
    def __init__(self, methods, batch_size=None):
        self.buffers: dict[str, list[OverlayInput]] = {
            method.label: [] for method in methods
        }
    
    def export(self, obs, action, step, run_id, meta):
        for method in self.methods:
            self.buffers[method.label].append(...)
            if len(self.buffers[method.label]) >= self.batch_size:
                self._flush_buffer(method)
```

### Key Characteristics

- **Independent:** Each method's buffer fills/flushes independently
- **No blocking:** Fast methods don't wait for slow ones
- **Clean flush:** `flush()` processes all buffers at training end
- **Simple:** Easy to reason about, test, and debug

---

## Consequences

### Positive

✅ **Isolation** — Methods don't interfere with each other  
✅ **Flexibility** — Can extend to per-method batch sizes in future  
✅ **Clear semantics** — Each buffer has single responsibility  
✅ **Easy testing** — Can test methods independently

### Negative

⚠️ **Memory overhead** — N buffers instead of 1 (negligible: ~50KB total)  
⚠️ **Code complexity** — More bookkeeping than single buffer

### Trade-offs Accepted

- **Memory cost:** Acceptable (buffers are small: max 32 items × ~5KB each ≈ 160KB per method)
- **Complexity:** Worth it for clean separation of concerns

---

## Alternatives Considered

### Alternative 1: Single Global Buffer

```python
self.buffer: list[tuple[OverlayMethod, OverlayInput]] = []
```

**Pros:**
- Simpler code
- Less memory

**Cons:**
❌ Methods block each other  
❌ Harder to flush partial batches cleanly  
❌ Can't optimize per-method in future

**Verdict:** Rejected — Coupling outweighs simplicity

### Alternative 2: Per-Label Buffers

```python
self.buffers: dict[str, list[OverlayInput]] = {}
# Multiple methods with same label share buffer
```

**Pros:**
- Groups semantically similar overlays

**Cons:**
❌ Assumes methods with same label have same performance  
❌ Breaks if two methods accidentally share labels

**Verdict:** Rejected — Per-method is safer

---

## Implementation Details

### Buffer Lifecycle

1. **Initialize:** Empty buffer per method
2. **Accumulate:** Add items to method's buffer on `export()`
3. **Flush (full batch):** When buffer reaches `batch_size`
4. **Flush (partial):** At training end via `exporter.flush()`

### Memory Analysis

**Worst case:**
- 4 methods × 32 batch_size × 5KB per item ≈ **640KB total**
- Negligible compared to model memory (~10-100MB)

**Typical case:**
- 2 methods × 4 batch_size × 5KB ≈ **40KB total**

---

## Future Enhancements

**Per-Method Batch Sizes (v1.1+):**
```python
XAIConfig(
    methods={
        "policy_grad": {"batch_size": 8},
        "value_grad": {"batch_size": 16},
        "heuristic": {"batch_size": None},  # Always sequential
    }
)
```

This is **forward-compatible** with per-method buffers.

---

## Implementation

- **Core:** `src/ungar/training/overlay_exporter.py`
  - `self.buffers` dictionary
  - `_flush_buffer(method)` helper
  - `flush()` public method
- **Tests:**
  - `tests/test_xai_batch.py` (buffer behavior)
  - `tests/test_xai_grad_integration.py` (end-to-end)
- **Documentation:** `docs/xai_overlays.md`

---

## References

- M22 Milestone: Batch Overlay Engine
- ADR-003: Opt-In Batch Processing
- M22 Audit: Section 3 (Modularity & Coupling)
- [xai_overlays.md](../xai_overlays.md)
- [API Contracts v1](../api_contracts_v1.md)

