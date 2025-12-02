# Changelog

All notable changes to UNGAR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2025-12-02

### Added - v1 Core Baseline

**XAI & Explainability (M19-M22):**
- Explainable AI overlay engine with 4 methods: heuristic, random, policy_grad, value_grad
- Batch overlay engine for 5-10× faster gradient computation (M22)
- CardOverlay 4×14 importance maps with JSON serialization
- CLI support for XAI configuration (`--xai-enabled`, `--xai-methods`, `--xai-batch-size`)
- M22 full-cycle demo script validating batched XAI end-to-end

**Analytics & CLI (M15-M17):**
- Analytics schema v1 (manifest.json, metrics.csv, overlays/)
- Unified CLI: `train`, `list-runs`, `show-run`, `summarize-overlays`, `compare-overlays`
- Overlay aggregation and visualization tools
- Run directory structure with versioned manifests

**Training & Agents (M12-M14):**
- PPO-Lite and DQN-Lite agents for card games
- Device abstraction (CPU, CUDA, MPS)
- Configurable training loops with logging
- Metrics tracking (rewards, losses, values)

**Games (M02, M10-M11):**
- High Card Duel (simple 2-player)
- Mini Spades (trick-taking)
- Gin Rummy (complex meld-based)

**Documentation:**
- Comprehensive API contracts (docs/api_contracts_v1.md)
- Architecture Decision Records (ADRs 001-004)
- Quickstart guide (docs/quickstart_v1.md)
- XAI overlay guide (docs/xai_overlays.md)
- CLI reference and training guides

**Testing & CI (M18):**
- 153 tests with 84%+ coverage
- Contract tests for v1 API stability
- GitHub Actions CI with smoke and full test tiers
- Security scanning (Bandit, pip-audit)
- SBOM generation

### Fixed

- Ruff formatting compliance across codebase
- MyPy type checking for gradient XAI methods
- Numerical equivalence between batch and sequential overlays

### Contract Guarantees

**v1.x Stability Promise:**
- 4×14×n tensor representation (no changes)
- CardOverlay structure and shape (no changes)
- Analytics schema v1 (no breaking changes)
- CLI commands and core flags (no removals)

---

## [Unreleased]

Future work (M23+):
- GPU-specific optimizations
- DQN batch overlay support
- Advanced XAI methods (Integrated Gradients, attention)
- Cross-game validation suite

---

**Version Guide:**
- **MAJOR** (X.0.0): Breaking changes to v1 contracts
- **MINOR** (0.X.0): New features, backward-compatible
- **PATCH** (0.0.X): Bug fixes only

**Current Stable:** 0.1.0  
**Schema Version:** 1

