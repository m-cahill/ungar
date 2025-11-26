# M00 Clarifying Questions

1. **Python Version Compatibility:** 
   - The M0 plan specifies a CI matrix of `["3.10", "3.11", "3.12"]`. 
   - RediAI documentation mentions standardizing on Python 3.11. 
   - **Question:** Should we maintain library compatibility for 3.10-3.12 (recommended for a standalone library/core), or strictly target 3.11+ to match RediAI's current state?

2. **License & Copyright:**
   - The plan suggests an MIT license.
   - **Question:** Please confirm the copyright holder name for the LICENSE file (e.g., "Michael Cahill" or an organization name?).

3. **Dependency Strategy:**
   - **Question:** For `requirements-dev.txt`, should I simply pin to the latest stable versions available today, or do you require specific alignment with RediAI's pinned versions immediately?

4. **Directory Structure:**
   - The plan specifies a standard `docs/` directory at the repo root.
   - **Question:** Please confirm you want the technical documentation in `docs/` at the root, distinct from the existing `Project Files/` folder where project management docs live.

