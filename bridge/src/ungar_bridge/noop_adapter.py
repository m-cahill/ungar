"""Dummy adapter implementation for testing and template purposes."""

from __future__ import annotations

from typing import Any

from ungar.game import Move

from .adapter_base import BridgeAdapter


class NoOpAdapter(BridgeAdapter):
    """A passthrough adapter that does no transformation."""

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize the adapter (no-op)."""
        pass

    def state_to_external(self, state: Any) -> Any:
        """Return the state as-is."""
        return state

    def external_to_move(self, external_action: Any) -> Move:
        """Assume external action is already a Move and return it."""
        if not isinstance(external_action, Move):
            raise TypeError(f"Expected Move, got {type(external_action)}")
        return external_action
