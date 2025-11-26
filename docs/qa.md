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

## Artifacts

CI runs currently produce:
* `coverage.xml`: Coverage report.
* `results.sarif`: OpenSSF Scorecard analysis (in GitHub Actions).

