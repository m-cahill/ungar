# ADR-001: Analytics Schema v1 Freeze

**Status:** Accepted  
**Date:** 2024-11 (M17)  
**Deciders:** Core Team  
**Related:** M17, M18, M19

---

## Context

Prior to M17, UNGAR's run directory structure and analytics file formats were evolving rapidly with each milestone. This created compatibility issues:
- Frontend couldn't rely on stable manifest structure
- Analysis scripts broke with schema changes
- Cross-run comparisons were unreliable

We needed to **freeze** a stable schema to enable frontend development and long-term experiment tracking.

---

## Decision

We will define and freeze **Analytics Schema v1** with the following guarantees:

### Frozen Components

1. **Run Directory Structure:**
   ```
   runs/{timestamp}_{game}_{algo}_{run_id}/
   ├── manifest.json
   ├── metrics.csv
   └── overlays/  (if XAI enabled)
   ```

2. **manifest.json Required Fields:**
   - `analytics_schema_version: 1`
   - `run_id`, `timestamp`, `created_at`
   - `game`, `algo`, `device`
   - `config` (full training configuration)

3. **metrics.csv Required Columns:**
   - `step`, `episode`, `reward`

### Compatibility Guarantees

- **Within v1.x:** Schema version remains `1`
- **Backward compatibility:** New optional fields allowed
- **Breaking changes:** Require schema version bump (`2`, `3`, etc.)

### Versioning

All manifests MUST include:
```json
{"analytics_schema_version": 1}
```

---

## Consequences

### Positive

✅ **Frontend can freeze** — RediAI UI development can proceed with confidence  
✅ **Cross-run compatibility** — Experiments remain comparable over time  
✅ **Clear upgrade path** — Schema changes require explicit version bumps  
✅ **Validation possible** — Can write schema validators and contract tests

### Negative

⚠️ **Flexibility reduced** — New metrics require schema evolution  
⚠️ **Migration burden** — Future schema bumps need migration tools

### Mitigations

- Allow **optional fields** for extensibility
- Provide **migration scripts** when bumping schema version
- Document **extension points** clearly

---

## Implementation

- **Validation:** `src/ungar/analysis/schema.py`
- **Manifest generation:** `src/ungar/training/run_dir.py`
- **Tests:** `tests/test_analytics_schema.py`
- **Documentation:** `docs/analytics_schema.md`

---

## References

- M17 Milestone: Analytics Contracts & Frontend Freeze
- [analytics_schema.md](../analytics_schema.md)
- [API Contracts v1](../api_contracts_v1.md)

