# M01 Clarifying Questions

1.  **Dependency Management (`pip-compile`):**
    *   In M00, `pip-compile` failed due to a local environment issue with `pip-tools`, so `requirements-dev.txt` was manually pinned.
    *   **Question:** Should I attempt to fix `pip-compile` execution in this environment (e.g., by upgrading `pip`/`pip-tools` in the shell), or should I continue manually updating `requirements-dev.txt` with pinned versions for new dependencies like `numpy`?

2.  **Plane Naming Conventions:**
    *   The plan uses snake_case examples (`my_hand`, `opponent_hand`).
    *   **Question:** Should we enforce snake_case for plane names in `CardTensorSpec`, or allow any string?

3.  **NumPy Version:**
    *   Plan specifies `numpy>=1.26,<3.0`.
    *   **Question:** Are there any specific known incompatibilities with NumPy 2.x we should be aware of for future RediAI integration, or is 2.x safe to target now? (Assuming safe as per plan, but verifying).

