"""Base interface for UNGAR bridge adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Protocol, Tuple

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


class UngarRLAdapter(Protocol):
    """Protocol for RL environments in the UNGAR bridge (Gymnasium-style)."""

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[Any, dict[str, Any]]:
        """Reset the environment."""
        ...

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute a step in the environment."""
        ...

    def legal_actions(self) -> List[int]:
        """Return a list of legal action indices."""
        ...

    @property
    def current_player(self) -> int:
        """Return the ID of the current player."""
        ...

    @property
    def num_players(self) -> int:
        """Return the total number of players."""
        ...
