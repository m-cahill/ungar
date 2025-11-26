# QA & Evidence

## Quality Gates (M00)

The following gates are enforced in CI (`make ci`):

| Check | Tool | Threshold |
| :--- | :--- | :--- |
| **Linting** | `ruff` | Zero errors |
| **Formatting** | `ruff format` | Zero differences |
| **Types** | `mypy` | Strict mode, zero errors |
| **Docstrings** | `pydocstyle` | Google convention |
| **Tests** | `pytest` | 100% pass rate |
| **Coverage** | `coverage.py` | ≥ 85% lines & branches |

## M01 – Card Tensor

M01 introduces the core data structures with strict invariants:

*   **Bijections:** 56-card mapping (`Card` ↔ Index ↔ Tensor Position) is verified via exhaustive unit tests and Hypothesis property tests.
*   **Immutability:** `CardTensor` data arrays are read-only to prevent accidental mutation.
*   **Guardrails:** `validate_exclusive_planes` and `validate_partition` helpers ensure logical consistency of game states.

## M02 – Game Definitions & Runner

M02 adds the game interface layer and reference implementation.

### Invariants
*   **Protocol Compliance:** `HighCardDuelState` fully implements `GameState`. Verified by type checkers and protocol tests.
*   **Partitioning:** `to_tensor()` is guaranteed to produce a valid partition of the deck (my hand + opponent hand + unseen = full deck).
*   **Termination:** Random walks on `HighCardDuel` are proven to terminate in ≤ 2 steps.

### Performance Baseline
A micro-benchmark (`scripts/benchmark_tensor.py`) measures the cost of creating a `CardTensor` from card sets and validating its partition.

**Baseline (Nov 2025):**
*   **Op:** Create tensor + `validate_partition`
*   **Time:** ~160 µs per iteration (Python 3.10, Windows)

Run the benchmark:
```bash
python scripts/benchmark_tensor.py
```

## Artifacts

CI runs currently produce:
*   `coverage.xml`: Coverage report.
*   `results.sarif`: OpenSSF Scorecard analysis (in GitHub Actions).
