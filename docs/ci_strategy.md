# UNGAR CI Strategy (M18)

**Status:** Stabilized (M18)
**Date:** 2025-11-30

This document describes the Continuous Integration strategy for UNGAR, covering test tiers, security hardening, and coverage policies.

## 1. Test Architecture

We use a **two-tier** testing strategy to ensure fast feedback while maintaining rigorous quality.

### Tier 1: Smoke Tests (Blocking, <20s)

*   **Marker:** `@pytest.mark.smoke`
*   **Goal:** Verify core contracts, environment integrity, and CLI entry points.
*   **Scope:**
    *   Tensor operations & invariants.
    *   Environment `reset()` and `step()`.
    *   CLI `list-runs --format json` (contract check).
    *   Schema validation on minimal artifacts.
*   **Execution:** Runs in the `smoke` CI job. Must pass before full tests run.

### Tier 2: Full Tests (Blocking, <5m)

*   **Marker:** Everything else (implicit).
*   **Goal:** Verify agent convergence, long-running games, and integration flows.
*   **Scope:**
    *   DQN/PPO training loops.
    *   Full game simulations.
    *   Integration tests (e.g., `test_analytics_integration.py`).
*   **Execution:** Runs in the `tests-full` CI job.

## 2. Security Policy

We enforce security hygiene via automated scanners.

### Tools

1.  **Bandit:** Static analysis for Python security issues.
    *   **Command:** `bandit -r src -lll -f json`
    *   **Policy:** Fail on **any HIGH severity** finding.
2.  **pip-audit:** Dependency vulnerability scanner.
    *   **Command:** `pip-audit -r requirements-dev.txt`
    *   **Policy:** Fail on **any known vulnerability**.

### Handling False Positives

*   **Bandit:** Use `# nosec` comments with a clear justification.
*   **pip-audit:** Vulnerabilities must be patched by upgrading dependencies. If a fix is unavailable and the risk is acceptable, use `--ignore-vuln` with a comment in the CI workflow/script explaining why.

## 3. Coverage Policy

*   **Target:** 70% (M18 Baseline).
*   **Margin:** The CI gate is set to `fail-under=70`.
*   **Evolution:** We plan to raise this to 80% in M20 once agent research stabilizes.
*   **Note:** Coverage is measured across `src/ungar` and `bridge/src/ungar_bridge`.

## 4. CLI Import Hygiene

The `ungar` CLI must remain responsive and runnable in environments without heavy visualization libraries (e.g., headless servers).

*   **Rule:** Do not import `matplotlib` or heavy analysis tools at the top level of `ungar/cli.py`.
*   **Pattern:** Import them lazily inside the specific command function (e.g., `cmd_plot_curves`).
*   **Test:** A smoke test ensures `ungar --help` runs successfully without these dependencies.

### XAI & Performance

*   **Default:** XAI overlays are **disabled** in standard CI runs to keep tests fast.
*   **Exceptions:** Specific integration tests (in the `Full` tier) may enable XAI on toy problems to verify the pipeline.

## 5. Developer Workflow

1.  **Local Dev:**
    *   Run smoke tests often: `pytest -m smoke`
    *   Run full tests before push: `pytest`
2.  **PR:**
    *   CI runs Smoke → Full + Security in parallel.
    *   Coverage report is posted to the summary.

