Here’s a concrete **M00 plan** you can drop straight into Cursor as an implementation brief for the empty `m-cahill/ungar` repo.

M00 is **pure scaffolding + CI harness**: we don’t implement real card logic yet, just enough structure that future milestones can safely plug in the UNGAR core/bridge design and RediAI integration.

---

## M00 – Repo & CI Bootstrap for UNGAR

### 🎯 M00 Goals

By the end of M00, the `ungar` repo should:

1. Be a **valid Python package skeleton** with `src/ungar` and `tests/`.
2. Have a **minimal but serious quality toolchain** (Ruff, mypy, pytest, coverage, pydocstyle, etc.) wired through **pre-commit** and **GitHub Actions**.
3. Provide a **single command to run “full CI” locally** (`make ci`), mirroring what GitHub Actions does. 
4. Include **vision + security docs** that set expectations for later: UNGAR stays RediAI-agnostic, with a future bridge package, and will eventually adopt NIST SSDF / OpenSSF Scorecard / SLSA provenance as it grows.

Code-wise, we only add **tiny stubs**: enough to prove imports, types, tests, and coverage all work.

---

## Target Layout After M00

Cursor should shape the repo to roughly:

```text
ungar/
  README.md
  VISION.md
  SECURITY.md
  LICENSE
  pyproject.toml
  requirements-dev.in
  requirements-dev.txt
  .gitignore
  .pydocstyle
  .pre-commit-config.yaml
  .coveragerc
  Makefile
  src/
    ungar/
      __init__.py
      version.py      # trivial function + __version__ constant
  tests/
    test_sanity.py
  docs/
    index.md
    qa.md            # “QA & Evidence” landing page (stub for now)
  .github/
    workflows/
      ci.yml
      dependency-review.yml
      scorecard.yml  # non-blocking, warn-only for now
```

This aligns with your UNGAR vision doc and “implementation agent” enhancement prompts, while still being small enough for an empty repo.

---

## Phase M00-A – Repo Skeleton & Vision

**Objective:** Turn the empty repo into a clearly scoped, documented project that matches the UNGAR vision and the “core + bridge” architecture.

### Tasks for Cursor

1. **Create basic Python package layout**

   * Add:

     ```text
     src/
       ungar/
         __init__.py
         version.py
     tests/
       test_sanity.py
     ```

   * `src/ungar/version.py`:

     ```python
     __all__ = ["__version__", "get_version"]

     __version__ = "0.0.0"


     def get_version() -> str:
         """Return the UNGAR package version string."""
         return __version__
     ```

   * `tests/test_sanity.py`:

     ```python
     from ungar.version import get_version


     def test_get_version_returns_non_empty_string() -> None:
         assert isinstance(get_version(), str)
         assert get_version() != ""
     ```

2. **Add `pyproject.toml` (project metadata only for now)**

   Minimal example (no build backend change yet):

   ```toml
   [project]
   name = "ungar"
   version = "0.0.0"
   description = "Universal Neural Grid for Analysis and Research (UNGAR)"
   authors = [{ name = "Michael Cahill" }]
   requires-python = ">=3.10"
   readme = "README.md"

   [project.optional-dependencies]
   dev = ["-r requirements-dev.txt"]
   ```

   (Tool sections for Ruff/mypy/pytest will be added in later phases.)

3. **Bring in the UNGAR vision**

   * Add `VISION.md` using the existing UNGAR vision document as content. 
   * Link it from `README.md` (e.g., “See [VISION.md](./VISION.md) for the high-level goals and principles of UNGAR.”).

4. **`README.md`**

   Short but clear, something like:

   * What UNGAR is (“all card games, one brain”, 4×13×n tensor as core invariant). 
   * Relationship to RediAI & the future **bridge package**, explicitly calling out that **UNGAR core will remain RediAI-agnostic** and the bridge will live in a separate repo.
   * Quickstart (“clone, `pip install -r requirements-dev.txt`, `make ci`”).

5. **LICENSE**

   * Add MIT or similar OSI license (MIT is consistent with your other work and the RediAI ecosystem). 

6. **`.gitignore`**

   * Standard Python + tooling patterns: `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.venv/`, `dist/`, `build/`, `.coverage`, `coverage.xml`, `.hypothesis/`, `*.egg-info/`.

### Acceptance for M00-A

* `python -m pip install -r requirements-dev.txt` **does not exist yet** (that’s next phase), but:

  * `python -m pip install .` (editable or not) should at least be possible once a backend is chosen later.
* `pytest` from repo root imports `ungar.version` and runs `test_sanity.py` successfully once dev deps are installed in later phase.

---

## Phase M00-B – Dev Tooling & Pre-commit

**Objective:** Install the **minimum dev tool stack** and wiring to enforce basic quality (lint, types, docstrings) consistently across local dev and CI. This is an early, light slice of the broader SSDF/ASVS-aligned plan you used on RediAI.

### Tasks for Cursor

1. **`requirements-dev.in` + `requirements-dev.txt`**

   * Create `requirements-dev.in` with core tools only for M00:

     ```text
     ruff
     mypy
     pytest
     pytest-cov
     coverage
     pydocstyle
     hypothesis
     ```

     (Bandit, pip-audit, radon, cyclonedx-bom, gitleaks, Sphinx, etc. will be added in a later milestone, not M00.)

   * Run:

     ```bash
     python -m pip install -U pip pip-tools
     pip-compile --generate-hashes -o requirements-dev.txt requirements-dev.in
     ```

   * Commit both files.

2. **`pyproject.toml` tool config**

   Extend `pyproject.toml` with tool sections (merge safely if it already exists):

   ```toml
   [tool.ruff]
   line-length = 100
   target-version = "py310"
   fix = true
   extend-select = ["I"]  # import sorting

   [tool.ruff.lint]
   # keep defaults; we can refine ignores later

   [tool.mypy]
   python_version = "3.10"
   check_untyped_defs = true
   disallow_incomplete_defs = true
   disallow_untyped_defs = true
   warn_unused_ignores = true
   no_implicit_optional = true
   strict_equality = true

   [tool.pytest.ini_options]
   addopts = "-q --maxfail=1 --disable-warnings --cov=src/ungar --cov-report=xml"
   testpaths = ["tests"]
   ```

3. **`.pydocstyle`**

   ```ini
   [pydocstyle]
   convention = google
   add-ignore = D105,D107
   ```

4. **`.coveragerc`**

   ```ini
   [run]
   branch = True
   source = src/ungar

   [report]
   fail_under = 85
   omit =
     */tests/*
     **/__init__.py
   ```

   With the tiny `version.py` + test, 85% is trivially satisfied, so we can enable the gate from day one. This is consistent with your full-stack reliability plan. 

5. **`.pre-commit-config.yaml`**

   Minimal but useful hooks (pin latest tags when Cursor actually edits the file):

   ```yaml
   repos:
     - repo: https://github.com/astral-sh/ruff-pre-commit
       rev: <pin-latest>
       hooks:
         - id: ruff
           args: ["--fix"]
         - id: ruff-format

     - repo: https://github.com/pre-commit/mirrors-mypy
       rev: <pin-latest>
       hooks:
         - id: mypy

     - repo: https://github.com/PyCQA/pydocstyle
       rev: <pin-latest>
       hooks:
         - id: pydocstyle
   ```

   (We’ll add gitleaks and additional hooks later, when the repo has more substance.)

6. **Install and run pre-commit once**

   * `pre-commit install`
   * `pre-commit run --all-files` (CI in M00 will also do this once it exists.)

### Acceptance for M00-B

From repo root, after `pip install -r requirements-dev.txt`:

* `ruff check .` passes.
* `ruff format --check .` passes.
* `mypy .` passes.
* `pydocstyle .` passes.
* `pytest` passes and `coverage.xml` is produced with ≥85% coverage.

---

## Phase M00-C – CI Workflow + Local CI Harness

**Objective:** Add a **single main CI workflow** that runs on GitHub (matrix Python 3.10–3.12) *and* a **local “full CI” command** that mirrors those checks. This follows modern GitHub Actions guidance for Python projects and prepares for future enhancements like Scorecard and SLSA. ([Real Python][1])

### Tasks for Cursor

1. **`Makefile` with CI-parity targets** 

   ```makefile
   .PHONY: lint typecheck test ci

   lint:
   	ruff check .
   	ruff format --check .

   typecheck:
   	mypy .

   test:
   	pytest
   	coverage report --fail-under=85

   ci: lint typecheck test
   ```

   This is the **single source of truth** for what “CI” means. We’ll make GitHub Actions call `make ci`, and developers can do the same locally.

2. **Main CI workflow: `.github/workflows/ci.yml`**

   ```yaml
   name: CI

   on:
     push:
       branches: [main]
     pull_request:

   permissions:
     contents: read

   jobs:
     ci:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: ["3.10", "3.11", "3.12"]
       steps:
         - uses: actions/checkout@v4

         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: ${{ matrix.python-version }}
             cache: "pip"

         - name: Install dev dependencies
           run: |
             python -m pip install -U pip
             python -m pip install -r requirements-dev.txt

         - name: Run CI (lint + types + tests)
           run: make ci

         - name: Upload coverage.xml
           if: always()
           uses: actions/upload-artifact@v4
           with:
             name: coverage-${{ matrix.python-version }}
             path: coverage.xml

         - name: CI summary
           if: always()
           run: |
             echo "## Quality gates" >> $GITHUB_STEP_SUMMARY
             coverage report >> $GITHUB_STEP_SUMMARY
   ```

   This matches mainstream Python CI practices (matrix builds, pip caching, artifact upload). ([Real Python][1])

3. **Dependency Review workflow (lightweight)**

   Add `.github/workflows/dependency-review.yml` using GitHub’s `dependency-review-action`, which you’ve already standardized in RediAI.

   ```yaml
   name: Dependency Review

   on:
     pull_request:

   permissions:
     contents: read
     pull-requests: write

   jobs:
     dependency-review:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/dependency-review-action@v4
   ```

4. **Scorecard workflow (non-blocking, prep for later)**

   Add `.github/workflows/scorecard.yml` with **`continue-on-error: true`** so it reports but doesn’t fail builds yet. ([undefined][2])

   ```yaml
   name: Scorecard

   on:
     schedule:
       - cron: "0 8 * * 1"
     push:
       branches: [main]

   permissions:
     contents: read
     security-events: write
     id-token: write

   jobs:
     analysis:
       runs-on: ubuntu-latest
       continue-on-error: true
       steps:
         - uses: actions/checkout@v4
           with:
             persist-credentials: false

         - uses: ossf/scorecard-action@v2.3.3
           with:
             results_file: results.sarif
             results_format: sarif
             publish_results: true

         - uses: github/codeql-action/upload-sarif@v3
           with:
             sarif_file: results.sarif
   ```

   This lays groundwork for the stronger SSDF/SLSA posture described in your enhancement docs without overloading an empty repo.

5. **Local CI instructions**

   In `README.md`, add a short “Local CI” section:

   ````md
   ## Local CI

   To run the same checks that GitHub Actions runs:

   ```bash
   python -m pip install -r requirements-dev.txt
   make ci
   ````

   ```
   ```

### Acceptance for M00-C

* GitHub Actions `CI` workflow **passes** on pushes/PRs against `main`.
* `make ci` passes locally on Python 3.10+ with only the stub code.
* `dependency-review` and `scorecard` workflows run successfully (Scorecard might warn, but must not fail builds).

---

## Phase M00-D – Security & QA Stubs (Lightweight)

**Objective:** Plant light but explicit seeds for **security posture** and **QA evidence**, aligned with NIST SSDF / OWASP ASVS and your RediAI registry work, while keeping M00 small.

### Tasks for Cursor

1. **`SECURITY.md`**

   Very small doc that:

   * States that UNGAR will align with **NIST SSDF SP 800-218** and OWASP ASVS L2 over time. ([GitHub Docs][3])
   * Mentions that OpenSSF Scorecard is already wired in warn-only mode and will be made blocking later. ([undefined][2])
   * Describes how to report vulnerabilities (e.g., GitHub security advisories or email).

   You can also include a tiny table mapping current M00 practices (pre-commit, CI, dependency review) to SSDF categories, as your enhancement brief suggests. 

2. **`docs/index.md` + `docs/qa.md`**

   * `docs/index.md`: minimal “Welcome to UNGAR” page that links to:

     * `VISION.md`
     * `docs/qa.md`
   * `docs/qa.md`: a short “QA & Evidence” stub that:

     * States that quality gates currently enforced are: Ruff, mypy, pydocstyle, pytest+coverage (85%).
     * States that CI artifacts currently include `coverage.xml` only, with plans to add more (radon, bandit, pip-audit, SBOM, etc.) in a later milestone.

3. **Update `README.md` “Roadmap” / “Next milestones”**

   Add a small bullet list anchoring M01+:

   * **M01** – UNGAR core card-physics abstraction + 4×13×n tensor encoder stub.
   * **M02** – Game definitions for 1–2 simple games (e.g., High Card Duel + simple trick-taker) + more tests.
   * **M03** – Security & supply-chain hardening: Bandit, pip-audit, cyclonedx SBOM, basic SLSA provenance for artifacts.
   * **M04** – RediAI bridge package (separate repo) and Workflow Registry integration.

   This keeps Cursor (and future you) oriented without locking in too many details.

### Acceptance for M00-D

* `SECURITY.md` exists and clearly signals future SSDF/ASVS alignment and current practices.
* `docs/index.md` and `docs/qa.md` exist and build successfully in Markdown (no Sphinx yet).
* `README.md` has a small “Roadmap / milestones” section referencing M01+.

---

## How This Sets Up “Full Local CI” Later

* **Single source of truth:** `Makefile` `ci` target = “full CI run”. GitHub Actions calls `make ci`, and developers do the same locally.
* **Easy future expansion:** When you add new workflows (security scan, SBOM, docs, etc.), you can either:

  * Extend `make ci` (so local runs grow with CI), or
  * Add a `make ci-full` and keep `make ci` as a fast subset.
* **Scorecard & supply chain:** Scorecard is already in place in non-blocking mode; when you later add SBOMs and SLSA attestations (as in your enhancement docs), they’ll fit naturally into the `.github/workflows` area and can be summarized on `docs/qa.md`.

---

## Handoff Summary for Cursor

When you paste this into Cursor, you can frame M00 as:

> **Milestone M00 – UNGAR Repo & CI Bootstrap**
>
> Implement phases M00-A through M00-D exactly as described:
>
> * Create package skeleton (`src/ungar`, `tests`) and docs (`README.md`, `VISION.md`, `SECURITY.md`, `docs/qa.md`).
> * Add `pyproject.toml`, `requirements-dev.in` / `requirements-dev.txt`, `.pydocstyle`, `.coveragerc`, `.pre-commit-config.yaml`.
> * Add `Makefile` with `ci` target and `.github/workflows/ci.yml` (Python 3.10–3.12 matrix, calls `make ci`).
> * Add `dependency-review.yml` and non-blocking `scorecard.yml`.
> * Ensure `make ci` passes locally and the `CI` workflow is green on GitHub.

Once M00 is complete and green, we’ll be ready to use M01 for the **actual UNGAR core API and 4×13×n tensor substrate**, knowing the scaffolding and CI are already rock solid.

[1]: https://realpython.com/github-actions-python/?utm_source=chatgpt.com "Continuous Integration and Deployment for Python With ..."
[2]: https://scorecard.dev/?utm_source=chatgpt.com "OpenSSF Scorecard"
[3]: https://docs.github.com/en/actions/reference/security/secure-use?utm_source=chatgpt.com "Secure use reference"
