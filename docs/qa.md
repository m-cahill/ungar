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

*   **Bijections:** 52-card mapping (`Card` ↔ Index ↔ Tensor Position) is verified via exhaustive unit tests and Hypothesis property tests.
*   **Immutability:** `CardTensor` data arrays are read-only to prevent accidental mutation.
*   **Guardrails:** `validate_exclusive_planes` and `validate_partition` helpers ensure logical consistency of game states.

## Artifacts

CI runs currently produce:
*   `coverage.xml`: Coverage report.
*   `results.sarif`: OpenSSF Scorecard analysis (in GitHub Actions).
