import subprocess

COMMANDS = [
    ["ruff", "check", "."],
    ["ruff", "format", "--check", "."],
    ["mypy", "."],
    [
        "pytest",
        "src",
        "tests",
        "bridge/tests",
        "--cov=src/ungar",
        "--cov=bridge/src/ungar_bridge",
        "--cov-report=term-missing",
    ],
]


def main() -> int:
    for cmd in COMMANDS:
        print(f"==> Running: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
