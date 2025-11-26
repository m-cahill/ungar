Here are concrete answers for each M00 clarification. 

---

### 1. Python Version Compatibility

> Should we maintain compatibility for 3.10–3.12 or strictly target 3.11+?

**Recommendation:**

* **Support `3.10`, `3.11`, and `3.12` in CI** for UNGAR as a standalone library.
* Treat **3.11 as the “primary dev version”** (matches RediAI), but keep 3.10 & 3.12 in the matrix.

**Why:**

* All three versions are still supported upstream. The Python devguide lists 3.11 and 3.12 as active branches with end-of-life dates in 2027 and 2028 respectively, and 3.10 still receiving security updates until around 2026. ([Python Developer's Guide][1])
* UNGAR is meant to be a **reusable core library** beyond RediAI; broader version support makes it friendlier to external users and other tools.
* We’re not relying on any 3.11+–only language features in M00, so there’s no cost to keeping 3.10 for now.

**Concrete instruction for Cursor:**

* Keep the CI matrix as:

  ```yaml
  strategy:
    matrix:
      python-version: ["3.10", "3.11", "3.12"]
  ```

* In `pyproject.toml`, set:

  ```toml
  requires-python = ">=3.10"
  ```

We can revisit and drop 3.10 **later**, e.g. when we want 3.11+–only features or when 3.10 is very close to EOL.

---

### 2. License & Copyright

> Who should be listed in the LICENSE file?

**Recommendation:**

* Use **MIT License**.
* Set the copyright as:

  ```text
  Copyright (c) 2025 Michael Cahill
  ```

If you later form an entity (LLC/corp) and want the library under that name, we can update future releases and/or add a dual notice. For M00, **“Michael Cahill”** is the right holder string to use.

---

### 3. Dependency Strategy (`requirements-dev.txt`)

> Pin to latest stable today, or align with RediAI’s pins?

**Recommendation:**

* For M00, **pin to the latest stable versions of the dev tools** (Ruff, mypy, pytest, etc.) at the time Cursor generates `requirements-dev.txt`.
* **No need to align with RediAI’s exact pins yet.**

**Why:**

* These are **pure dev tooling**, not runtime dependencies. There’s minimal risk if their versions differ from RediAI’s current pins.
* Keeping them current is beneficial; newer pytest/mypy/ruff generally bring better correctness and performance. ([TechRadar][2])
* When UNGAR is integrated more tightly into RediAI, we can either:

  * Adopt whichever tool versions are newer/stricter, or
  * Introduce a shared `constraints.txt` / central tooling policy if needed.

**Concrete instruction for Cursor:**

* Use a `requirements-dev.in` like:

  ```text
  ruff
  mypy
  pytest
  pytest-cov
  coverage
  pydocstyle
  hypothesis
  ```

* Then generate `requirements-dev.txt` with `pip-compile` and **commit the pinned output**.

Alignment with RediAI can be tackled as a separate “tooling unification” milestone later, once UNGAR’s core stabilizes.

---

### 4. Directory Structure: `docs/` vs `Project Files/`

> Do we want technical documentation in `docs/` at repo root, separate from `Project Files/` where project management docs live?

**Recommendation:**

* **Yes, keep them distinct:**

  * `docs/` at repo root → **technical / user / developer documentation** that ships with the project and can later be built by Sphinx/MkDocs or similar.
  * `Project Files/` → **planning, audits, milestone specs, prompts**, and other meta-docs that describe how we’re building UNGAR rather than how to use it.

This matches the pattern in your other projects where in-repo docs are cleanly separated from higher-level management artifacts.

**Concrete instruction for Cursor:**

* Create and maintain:

  ```text
  docs/
    index.md
    qa.md
  ```

* Leave `Project Files/` (or whatever you end up using) as the home for:

  * Milestone briefs (M00, M01, …),
  * Audit prompts,
  * Enhancement plans, etc.

`README.md` and `VISION.md` can link into both worlds (e.g., link to `docs/index.md` for user docs and optionally to a project roadmap in `Project Files/` if you want).

---

If you’d like, next step after these answers is to have me tighten the M00 brief text to incorporate these decisions explicitly so you can paste a single, conflict-free spec into Cursor.

[1]: https://devguide.python.org/versions/?utm_source=chatgpt.com "Status of Python versions"
[2]: https://www.techradar.com/pro/python-isnt-dead-despite-funding-cuts-programming-language-powers-on?utm_source=chatgpt.com "Python isn't dead - despite funding cuts, programming language powers on"
