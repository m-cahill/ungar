# UNGAR v1 Release Checklist

**Purpose:** Pre-release validation for UNGAR v1.x versions

**Usage:** Check off all items before tagging a release

---

## Pre-Release Validation

### Code Quality

- [ ] All tests passing (`pytest`)
  ```bash
  python -m pytest
  ```
- [ ] Coverage ≥ 84% (`pytest --cov`)
- [ ] No linting errors (`ruff check .`)
- [ ] No formatting issues (`ruff format --check .`)
- [ ] Type checking clean (`mypy src/ungar`)

### CI Status

- [ ] All GitHub Actions workflows green
  - [ ] Smoke tests (3.10, 3.11, 3.12)
  - [ ] Full test suite (3.10, 3.11, 3.12)
  - [ ] Security scans (Bandit, pip-audit)
  - [ ] SBOM generation

- [ ] M22 demo passes (`python scripts/demo_m22_full_cycle.py --clean`)

### Contract Validation

- [ ] Contract tests passing (`pytest tests/test_*_contract.py`)
- [ ] Analytics schema version unchanged (`analytics_schema_version=1`)
- [ ] No breaking changes to v1 contracts:
  - [ ] 4×14×n tensor shape
  - [ ] CardOverlay structure
  - [ ] CLI commands and flags
  - [ ] manifest.json required fields

### Documentation

- [ ] CHANGELOG.md updated with release notes
- [ ] README.md current
- [ ] API contracts document current (`docs/api_contracts_v1.md`)
- [ ] All code examples in docs tested and working

### Manual Validation

- [ ] Run M22 demo locally
- [ ] Visually inspect heatmap PNG
- [ ] Review `demo_m22_summary.json` output
- [ ] Test CLI commands manually:
  ```bash
  ungar train --game high_card_duel --algo ppo --episodes 10 --xai-enabled
  ungar list-runs
  ungar show-run {run_id}
  ungar summarize-overlays --run {run_dir} --out-dir analysis/
  ```

### Version Management

- [ ] Version number updated in `src/ungar/version.py`
- [ ] `ungar --version` shows correct version
- [ ] Git tag created: `v{version}`
  ```bash
  git tag -a v0.1.0 -m "UNGAR v0.1.0 - Core Baseline"
  ```

---

## Release Process

### 1. Update Version

```bash
# Edit src/ungar/version.py
__version__ = "0.1.0"

# Verify
python -c "from ungar import __version__; print(__version__)"
```

### 2. Update CHANGELOG

Add release section to CHANGELOG.md:
```markdown
## [0.1.0] - YYYY-MM-DD
### Added
- Feature 1
- Feature 2
...
```

### 3. Run Full Validation

```bash
# Run all checks
pytest
python scripts/demo_m22_full_cycle.py --clean
ruff check .
mypy src/ungar
```

### 4. Commit and Tag

```bash
# Commit version bump
git add src/ungar/version.py CHANGELOG.md
git commit -m "chore(release): bump version to 0.1.0"

# Create annotated tag
git tag -a v0.1.0 -m "UNGAR v0.1.0 - Core Baseline

- XAI batch overlay engine (M22)
- v1 API contracts stabilized
- 153 tests, 84% coverage
- Analytics schema v1
"

# Push with tags
git push origin main --tags
```

### 5. Create GitHub Release

- Go to https://github.com/m-cahill/ungar/releases/new
- Select tag: `v0.1.0`
- Title: `UNGAR v0.1.0 - Core Baseline`
- Description: Copy from CHANGELOG.md
- Attach artifacts (optional):
  - `demo_m22_summary.json`
  - `mean_heatmap.png`

---

## Post-Release

- [ ] Verify GitHub release published
- [ ] Verify CI runs clean on tag
- [ ] Update project board/milestones
- [ ] Announce in project channels (if applicable)

---

## Rollback Plan

If critical issues found after release:

1. **Patch Release (0.1.1):**
   - Fix bugs
   - Follow checklist again
   - Tag `v0.1.1`

2. **Revert Tag (Emergency):**
   ```bash
   git tag -d v0.1.0
   git push origin :refs/tags/v0.1.0
   ```

---

**Last Updated:** 2025-12-02  
**For:** UNGAR v1.x releases

