"""Generate CI summary for GitHub Actions (M23).

Outputs structured summary for $GITHUB_STEP_SUMMARY.
"""

import json
from pathlib import Path

from ungar import __version__


def get_analytics_schema_version() -> int:
    """Get current analytics schema version from code."""
    # Hard-coded for v1.x
    return 1


def check_demo_status() -> str:
    """Check if M22 demo completed successfully."""
    # Look for most recent demo summary
    runs_dir = Path("runs")
    if not runs_dir.exists():
        return "SKIP"

    demo_summaries = list(runs_dir.glob("**/demo_m22_summary.json"))
    if not demo_summaries:
        return "NOT_RUN"

    # Check most recent
    latest = max(demo_summaries, key=lambda p: p.stat().st_mtime)
    try:
        with open(latest, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data.get("numerical_equivalence_passed"):
                return "PASS"
            else:
                return "FAIL"
    except Exception:
        return "ERROR"


def get_commit_hash() -> str:
    """Get short commit hash."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def get_test_coverage() -> str:
    """Get test coverage from coverage.xml if available."""
    coverage_file = Path("coverage.xml")
    if not coverage_file.exists():
        return "N/A"

    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(coverage_file)
        root = tree.getroot()
        coverage_elem = root.find(".//coverage")
        if coverage_elem is not None:
            line_rate = float(coverage_elem.get("line-rate", 0))
            return f"{line_rate * 100:.1f}%"
    except Exception:
        pass

    return "N/A"


def main() -> None:
    """Generate and print CI summary."""
    commit = get_commit_hash()
    schema_version = get_analytics_schema_version()
    demo_status = check_demo_status()
    coverage = get_test_coverage()

    # Structured bullet format (from user answers)
    print(f"✅ UNGAR v{__version__} (commit {commit})")
    print(f"✅ Analytics Schema: v{schema_version}")
    print(f"✅ M22 Demo: {demo_status}")
    print(f"✅ Coverage: {coverage}")


if __name__ == "__main__":
    main()
