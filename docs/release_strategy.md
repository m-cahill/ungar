# Release Strategy

## Versioning

UNGAR follows a split versioning strategy:

*   **Core (`ungar`):** Follows Semantic Versioning (Major.Minor.Patch). Changes to `CardTensor` or core logic trigger bumps.
*   **Bridge (`ungar-bridge`):** Follows its own Semantic Versioning. It may move faster than Core as new adapters are added.

## Distribution

Both packages are distributed via PyPI.

### Installing Core Only
```bash
pip install ungar
```

### Installing Core + Bridge
```bash
pip install ungar-bridge
```
(This automatically pulls in the compatible `ungar` core version).

## Release Process

1.  **Tagging:** Create a tag `vX.Y.Z` for core releases.
2.  **Automation:** The `release.yml` workflow builds and publishes the package.
3.  **Bridge Releases:** Currently, bridge releases are manually triggered or tagged separately (e.g., `bridge-vX.Y.Z`), though monorepo tooling (like `release-please`) could automate this in the future. For M04, we assume manual PyPI upload or shared tagging.

