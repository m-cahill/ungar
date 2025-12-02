from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from ungar.enums import RANK_COUNT, SUIT_COUNT
from ungar.xai import CardOverlay, zero_overlay
from ungar.xai_grad import compute_policy_grad_importance, compute_value_grad_importance


@runtime_checkable
class OverlayMethod(Protocol):
    """Protocol for XAI overlay generation methods."""

    label: str

    def compute(
        self,
        obs: np.ndarray,
        action: int,
        *,
        step: int,
        run_id: str,
        meta: dict | None = None,
    ) -> CardOverlay:
        """Compute an overlay for a given observation and action."""
        ...


class RandomOverlayMethod:
    """Generates random importance values."""

    label = "random"

    def compute(
        self,
        obs: np.ndarray,
        action: int,
        *,
        step: int,
        run_id: str,
        meta: dict | None = None,
    ) -> CardOverlay:
        """Generate a random 4x14 overlay."""
        importance = np.random.rand(SUIT_COUNT, RANK_COUNT)
        # Normalize to sum to 1
        importance = importance / importance.sum()

        return CardOverlay(
            run_id=run_id,
            label=self.label,
            agg="none",
            step=step,
            importance=importance,
            meta=meta or {},
        )


class HandHighlightMethod:
    """Highlights cards in the agent's hand (heuristic)."""

    label = "heuristic"

    def compute(
        self,
        obs: np.ndarray,
        action: int,
        *,
        step: int,
        run_id: str,
        meta: dict | None = None,
    ) -> CardOverlay:
        """Generate an overlay highlighting held cards."""
        # This method is tightly coupled to tensor layout.
        # For now, we try to infer or assume N based on size.

        size = obs.size
        # 4 * 14 = 56
        tensor_plane_size = SUIT_COUNT * RANK_COUNT

        if size % tensor_plane_size != 0:
            # Fallback if unknown shape
            return zero_overlay(self.label, meta)

        n_planes = size // tensor_plane_size
        # Reshape to (4, 14, n)
        # NOTE: Check flatten order. Usually 'C' (row-major).
        tensor = obs.reshape((SUIT_COUNT, RANK_COUNT, n_planes))

        # Plane 0 = My Hand (convention in high_card_duel and others)
        hand_plane = tensor[:, :, 0]

        # Normalize
        count = np.sum(hand_plane)
        if count > 0:
            importance = hand_plane.astype(float) / count
        else:
            importance = np.zeros((SUIT_COUNT, RANK_COUNT), dtype=float)

        return CardOverlay(
            run_id=run_id,
            label=self.label,
            agg="none",
            step=step,
            importance=importance,
            meta=meta or {},
        )


class PolicyGradOverlayMethod:
    """Gradient-based importance using policy output gradients."""

    label = "policy_grad"

    def __init__(self, model: nn.Module, game_name: str) -> None:
        self.model = model
        self.game_name = game_name

    def compute(
        self,
        obs: np.ndarray,
        action: int,
        *,
        step: int,
        run_id: str,
        meta: dict[str, Any] | None = None,
    ) -> CardOverlay:
        """Compute policy gradient overlay."""
        # Convert obs to tensor
        obs_tensor = torch.from_numpy(obs).float()

        # Compute importance
        importance = compute_policy_grad_importance(
            self.model, obs_tensor, action_index=action
        )

        return CardOverlay(
            run_id=run_id,
            label=self.label,
            agg="none",
            step=step,
            importance=importance,
            meta={
                **(meta or {}),
                "game": self.game_name,
                "method": "policy_grad",
                "target_type": "logit_or_q",  # Generic for now
            },
        )


class ValueGradOverlayMethod:
    """Gradient-based importance using value/critic output gradients.

    For PPO actor-critic agents, this computes the gradient of the state-value
    function V(s) with respect to the input observation, revealing which cards
    the critic considers most important for state valuation.

    Note: Currently only supported for PPO-style actor-critic agents.
    """

    label = "value_grad"

    def __init__(self, model: nn.Module, game_name: str, algo: str = "ppo") -> None:
        """Initialize value gradient overlay method.

        Args:
            model: The critic/value network (e.g., PPOLiteAgent.actor for ActorCritic).
                   Should have a get_value() method or return value output on forward().
            game_name: Name of the game being played.
            algo: Algorithm name (e.g., "ppo"). Used for metadata.
        """
        self.model = model
        self.game_name = game_name
        self.algo = algo

    def compute(
        self,
        obs: np.ndarray,
        action: int,
        *,
        step: int,
        run_id: str,
        meta: dict[str, Any] | None = None,
    ) -> CardOverlay:
        """Compute value gradient overlay.

        Note: The 'action' parameter is ignored for value gradients since we're
        computing V(s), not Q(s, a). It's kept in the signature for protocol compatibility.

        Args:
            obs: Observation array (flattened card tensor).
            action: Action index (ignored for value gradients).
            step: Training step number.
            run_id: Unique run identifier.
            meta: Optional additional metadata.

        Returns:
            CardOverlay with value gradient importance map.
        """
        # Convert obs to tensor
        obs_tensor = torch.from_numpy(obs).float()

        # Compute importance via value gradients
        importance = compute_value_grad_importance(self.model, obs_tensor)

        return CardOverlay(
            run_id=run_id,
            label=self.label,
            agg="none",
            step=step,
            importance=importance,
            meta={
                **(meta or {}),
                "game": self.game_name,
                "method": "value_grad",
                "target_type": "state_value",
                "algo": self.algo,
            },
        )
