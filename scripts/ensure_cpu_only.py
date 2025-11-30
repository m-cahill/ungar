"""Ensure that the installed torch build is CPU-only in CI."""

from __future__ import annotations

import torch


def main() -> None:
    """Exit non-zero if CUDA is available."""
    if torch.cuda.is_available():
        raise SystemExit("CUDA is available; expected CPU-only torch build in CI.")

    print(f"SUCCESS: Torch {torch.__version__} is CPU-only as expected.")


if __name__ == "__main__":
    main()
