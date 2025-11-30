# Frontend Status: Frozen v1 (Mock-Only)

**Date:** 2025-11-30 (M17)
**Status:** Frozen / Mock-Complete

## Executive Summary

The UNGAR frontend is currently in a **frozen** state. It is functionally complete as a **v1 mock-up**, demonstrating the intended UX for:
*   Game board visualization (High Card Duel, etc.)
*   Card overlay grids (4x14)
*   Reward tables
*   Analytics dashboards (using mock data)

**No further frontend development is planned** until the backend agents and bridge infrastructure are fully mature.

## What Exists

*   **Routes:**
    *   `/ungar`: Landing page.
    *   `/ungar/demo`: Interactive game demo (mock backend).
    *   `/ungar/analytics`: Analytics dashboard (mock data).
*   **Components:**
    *   `CardOverlayGrid`: Visualizes the 4x14 tensor overlays.
    *   `RewardTable`: Shows episodic rewards.
    *   `Recharts` integration for learning curves.

## Integration Contract (Future Work)

When frontend work resumes, it must integrate with the backend via the **Analytics Schema v1**.

### 1. Artifact Consumption
The frontend should **not** rely on live backend execution during page load. Instead, it should consume exported run artifacts:
*   **Source:** `manifest.json`, `metrics.csv`, `overlays/*.json`
*   **Contract:** See [docs/analytics_schema.md](analytics_schema.md)

### 2. Run Discovery
The frontend should discover runs using the CLI contract:
*   **Command:** `ungar list-runs --format json`
*   **Output:** JSON list of available runs and their paths.

### 3. Run Export
To move data to the frontend hosting environment (if separate), use:
*   **Command:** `ungar export-run --run-id <ID> --out-dir <WEB_ROOT>/data/<ID>`

## Why Frozen?

Freezing the frontend allows the core team to focus 100% on:
1.  **Agent Performance:** Improving DQN/PPO logic.
2.  **Bridge Reliability:** hardening RediAI integration.
3.  **XAI Accuracy:** Ensuring overlays represent true model internals.

Once these backend pillars are solid, the frontend can be "thawed" to consume the real data they produce.

