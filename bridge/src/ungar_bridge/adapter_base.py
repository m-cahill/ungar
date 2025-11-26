"""Base interface for UNGAR bridge adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ungar.game import Move


class BridgeAdapter(ABC):
    """Abstract base class for bridging UNGAR core to external frameworks."""

    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the adapter with configuration."""

    @abstractmethod
    def state_to_external(self, state: Any) -> Any:
        """Convert a UNGAR GameState into the external framework's format."""

    @abstractmethod
    def external_to_move(self, external_action: Any) -> Move:
        """Convert an external framework action into a UNGAR Move."""
