# Security Policy

## Supported Versions

| Version | Supported          |
| :------ | :----------------- |
| 0.x     | :white_check_mark: |

## Reporting a Vulnerability

Please report vulnerabilities via GitHub Issues or email the maintainer directly.

## Security Posture (M03)

UNGAR enforces strict security hygiene throughout its lifecycle:

### 1. Static Analysis
*   **Tool:** [Bandit](https://github.com/PyCQA/bandit)
*   **Scope:** All source code (`src/ungar`) is scanned for common Python security issues (e.g., weak crypto, hardcoded secrets, unsafe deserialization).
*   **Gate:** CI fails if any High/Medium severity issues are found (unless explicitly suppressed with `# nosec`).

### 2. Dependency Auditing
*   **Tool:** [pip-audit](https://pypi.org/project/pip-audit/)
*   **Scope:** All installed dependencies are checked against the PyPA Advisory Database.
*   **Gate:** CI fails if any known vulnerabilities are detected.

### 3. Supply Chain & SBOM
*   **SBOM:** A Software Bill of Materials (CycloneDX JSON) is generated for every build and release.
*   **Provenance:** Release artifacts are built in a controlled CI environment. (Full SLSA attestation planned for future milestones).

### 4. Development Guidelines
*   **Secrets:** Never commit secrets to git.
*   **Randomness:** Use `secrets` for crypto, `random` only for simulations.
*   **Input Validation:** Always validate inputs at the boundary of public APIs.
