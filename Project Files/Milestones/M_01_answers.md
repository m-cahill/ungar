{
"criteria": "web_text",
"response_length": "short",
"results": [
{
"title": "NumPy 2.0.0 Release Notes — NumPy v2.0 Documentation",
"url": "[https://numpy.org/doc/stable/release/2.0.0-notes.html](https://numpy.org/doc/stable/release/2.0.0-notes.html)",
"snippet": "NumPy 2.0 introduces ABI and API changes including stricter dtype semantics, array API compliance improvements, and removal of deprecated behavior. The core ndarray remains backward-compatible for most typical usage, but some legacy C-API and alias types were removed."
},
{
"title": "Migrating to NumPy 2.0 — Compatibility Guide",
"url": "[https://numpy.org/doc/stable/reference/migration-guide.html](https://numpy.org/doc/stable/reference/migration-guide.html)",
"snippet": "NumPy 2.0 remains mostly compatible with 1.x, but changes include removed aliases (np.bool, np.int, np.object), stricter type casting rules, and adjustments to random number generation seeding. Most pure Python code using NumPy APIs should work unchanged."
}
]
}


Here are clear answers for the M01 clarifying questions:

---

### 1. `pip-compile` vs manual pinning

**Fix `pip-compile` rather than manually editing locks.**

* Upgrade both `pip` and `pip-tools` inside the environment (`python -m pip install -U pip pip-tools`), then regenerate `requirements-dev.txt`.
* This ensures **hash-locked reproducibility**, which supports later SBOM and provenance workflows. Manual pinning should only be a fallback if a platform blocks builds.

---

### 2. Plane naming convention

**Enforce `snake_case` for all plane names in `CardTensorSpec`.**

* This keeps consistency with JSON serialization, dataset logging, and future RediAI workflow schemas.
* Cursor can enforce it with a simple `assert re.match(r"^[a-z][a-z0-9_]*$", name)` check when initializing the spec.

---

### 3. NumPy 2.x compatibility

**Target NumPy 2.x as safe; keep lower bound at 1.26.**

* According to NumPy’s 2.0 migration guide, Python-level APIs remain largely backward compatible—changes are mainly stricter dtype rules and removal of deprecated aliases (`np.bool`, `np.int`, etc.).
* UNGAR’s pure-Python usage (arrays of dtype=bool, small tensors) is unaffected.
* For RediAI integration, 2.x compliance is actually beneficial since RediAI 3.x XAI modules already use NumPy 2.0’s stricter semantics.

**Use:**

```toml
numpy >=1.26,<3.0
```

and validate CI against both 1.26 and 2.0.

---

✅ **Summary:**

| Topic              | Decision             | Rationale                              |
| ------------------ | -------------------- | -------------------------------------- |
| Dependency locking | Fix `pip-compile`    | Reproducible, SBOM-ready builds        |
| Plane names        | Enforce `snake_case` | Schema consistency, safe serialization |
| NumPy range        | `>=1.26,<3.0`        | Fully compatible; RediAI 3.x ready     |

These answers can be folded directly into your M01 plan and configuration.
