"""Device configuration and management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass
class DeviceConfig:
    """Configuration for computation device."""

    device: Literal["cpu", "cuda", "mps", "auto"] = "auto"
    use_compile: bool = False  # torch.compile (2.0+)
    use_amp: bool = False  # Automatic Mixed Precision


def get_device(config: DeviceConfig | None = None) -> torch.device:
    """Select the best available device based on config and hardware."""
    if config is None:
        requested = "auto"
    else:
        requested = config.device

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        # Fallback with warning? Or fail? Fail is safer for explicit config.
        # But for M14 we'll just return CPU if requested CUDA isn't there to be safe in CI?
        # No, explicit config should probably be respected or fail.
        # However, `train_dqn` defaults might request auto.
        # Let's just return what requested and let torch fail if invalid.
        return torch.device("cuda")

    return torch.device(requested)
