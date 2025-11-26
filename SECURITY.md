# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report vulnerabilities via email to `security@ungar.ai` (placeholder) or via GitHub Security Advisories.

## Security Posture

UNGAR aligns with **NIST SSDF SP 800-218** and **OWASP ASVS Level 2** practices.

Current security controls (M00):
* **Static Analysis:** `ruff` (lint), `mypy` (types).
* **Dependency Review:** GitHub Action runs on PRs.
* **Scorecard:** OpenSSF Scorecard runs weekly (warn-only).
* **Branch Protection:** Enforced via CI (planned).

