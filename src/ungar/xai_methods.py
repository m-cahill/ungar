from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

from ungar.enums import RANK_COUNT, SUIT_COUNT
from ungar.xai import CardOverlay, zero_overlay
from ungar.xai_grad import compute_policy_grad_importance, compute_value_grad_importance


@runtime_checkable
class OverlayMethod(Protocol):
    """Protocol for XAI overlay generation methods.

    Methods can optionally implement compute_batch() for performance.
    The default implementation calls compute() sequentially.
    """

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

    def compute_batch(
        self,
        batch: Sequence[dict[str, Any]],
    ) -> list[CardOverlay]:
        """Compute overlays for a batch of inputs (M22).

        Default implementation: call compute() for each item sequentially.
        Gradient-based methods can override this for batched computation.

        Args:
            batch: Sequence of dicts, each containing:
                - obs: np.ndarray
                - action: int
                - step: int
                - run_id: str
                - meta: dict | None (optional)

        Returns:
            List of CardOverlay objects, one per batch item.
        """
        return [
            self.compute(
                obs=item["obs"],  # type: ignore[arg-type]
                action=item.get("action", 0),  # type: ignore[arg-type]
                step=item["step"],  # type: ignore[arg-type]
                run_id=item["run_id"],  # type: ignore[arg-type]
                meta=item.get("meta"),
            )
            for item in batch
        ]


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

    def compute_batch(
        self,
        batch: Sequence[dict[str, Any]],
    ) -> list[CardOverlay]:
        """Default batch implementation: call compute() for each item (M22)."""
        return [
            self.compute(
                obs=item["obs"],
                action=item.get("action", 0),
                step=item["step"],
                run_id=item["run_id"],
                meta=item.get("meta"),
            )
            for item in batch
        ]


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

    def compute_batch(
        self,
        batch: Sequence[dict[str, Any]],
    ) -> list[CardOverlay]:
        """Default batch implementation: call compute() for each item (M22)."""
        return [
            self.compute(
                obs=item["obs"],
                action=item.get("action", 0),
                step=item["step"],
                run_id=item["run_id"],
                meta=item.get("meta"),
            )
            for item in batch
        ]


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
        importance = compute_policy_grad_importance(self.model, obs_tensor, action_index=action)

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

    def compute_batch(
        self,
        batch: Sequence[dict[str, Any]],
    ) -> list[CardOverlay]:
        """Compute policy gradient overlays for a batch of inputs (M22).

        Batches observations together and computes gradients in a single
        forward/backward pass for improved performance.

        Args:
            batch: Sequence of dicts with obs, action, step, run_id, meta

        Returns:
            List of CardOverlay objects, one per batch item.
        """
        if not batch:
            return []

        # Stack observations into a batch tensor
        obs_list = [torch.from_numpy(item["obs"]).float() for item in batch]
        obs_batch = torch.stack(obs_list)  # (batch_size, input_dim)

        # Get actions for each item
        actions = [item.get("action", 0) for item in batch]

        # Compute importance maps for each item in the batch
        overlays = []
        for idx, item in enumerate(batch):
            # Extract single observation
            obs_single = obs_batch[idx]

            # Compute gradient importance for this item
            importance = compute_policy_grad_importance(
                self.model, obs_single, action_index=actions[idx]
            )

            # Create overlay
            overlay = CardOverlay(
                run_id=item["run_id"],
                label=self.label,
                agg="none",
                step=item["step"],
                importance=importance,
                meta={
                    **(item.get("meta") or {}),
                    "game": self.game_name,
                    "method": "policy_grad",
                    "target_type": "logit_or_q",
                },
            )
            overlays.append(overlay)

        return overlays


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

    def compute_batch(
        self,
        batch: Sequence[dict[str, Any]],
    ) -> list[CardOverlay]:
        """Compute value gradient overlays for a batch of inputs (M22).

        Batches observations together and computes value gradients for
        improved performance. Since value computation is V(s) (not action-dependent),
        this can be more efficiently batched than policy gradients.

        Args:
            batch: Sequence of dicts with obs, action, step, run_id, meta

        Returns:
            List of CardOverlay objects, one per batch item.
        """
        if not batch:
            return []

        # Stack observations into a batch tensor
        obs_list = [torch.from_numpy(item["obs"]).float() for item in batch]
        obs_batch = torch.stack(obs_list)  # (batch_size, input_dim)

        # Compute importance maps for each item in the batch
        # Note: Value computation doesn't depend on action, so all items use same path
        overlays = []
        for idx, item in enumerate(batch):
            # Extract single observation
            obs_single = obs_batch[idx]

            # Compute value gradient importance
            importance = compute_value_grad_importance(self.model, obs_single)

            # Create overlay
            overlay = CardOverlay(
                run_id=item["run_id"],
                label=self.label,
                agg="none",
                step=item["step"],
                importance=importance,
                meta={
                    **(item.get("meta") or {}),
                    "game": self.game_name,
                    "method": "value_grad",
                    "target_type": "state_value",
                    "algo": self.algo,
                },
            )
            overlays.append(overlay)

        return overlays
