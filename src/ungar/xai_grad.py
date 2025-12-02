"""Gradient-based XAI utilities for UNGAR.

This module provides helpers to compute importance maps (saliency) by backpropagating
gradients from network outputs (Q-values or policy logits) to the input tensor.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn

from ungar.enums import RANK_COUNT, SUIT_COUNT


def compute_policy_grad_importance(
    model: nn.Module,
    obs_tensor: torch.Tensor,
    action_index: int,
    normalize: bool = True,
) -> np.ndarray:
    """Compute gradient-based importance map for a specific action output.

    For DQN: Gradients of Q(s, a) w.r.t. s.
    For PPO: Gradients of Logit(a) w.r.t. s.

    Args:
        model: The PyTorch model (DQN or Actor).
        obs_tensor: Input observation tensor of shape (4, 14, N) or (Batch, ...).
                    If single observation, it will be unsqueezed.
        action_index: The index of the action to attribute.
        normalize: Whether to normalize the result to sum to 1.

    Returns:
        A (4, 14) float numpy array representing importance.
    """
    model.eval()

    # Ensure input requires grad
    # Clone to avoid modifying original tensor if it's being used elsewhere
    # Handle batch dimension: if (4, 14, N), unsqueeze to (1, 4, 14, N)
    # UNGAR models typically expect flat input (B, input_dim)
    # We need to know if the model expects flat or structured input.
    # The current DQNLite/PPOLite agents take FLATTENED input.
    # However, to compute attribution on the 4x14 grid, we need the structure.
    # The gradient will be w.r.t. the flattened input, which we can reshape back.

    # Check if input is flat or structured.
    # Agent wrapper usually flattens it.
    # Here we assume we receive the tensor that goes INTO the model.
    # If the model takes flat input, we pass flat input.

    _is_flat = len(obs_tensor.shape) == 1
    _is_batch_flat = len(obs_tensor.shape) == 2

    # If we get structured input (4, 14, N), we might need to flatten it for the model
    # BUT we need to track gradients on the original structured tensor if possible,
    # or reshape the gradient later.

    # Let's standardize: we want gradients w.r.t. the 4x14xN representation.
    # If the model takes flattened input, we flat it inside.

    # Let's assume obs_tensor is the RAW observation (probably flattened if coming from agent.act)
    # But wait, to get gradients w.r.t. specific cards, we need to know the shape (4, 14, N).
    # If we receive a flat tensor, we must know N to reshape the gradient.

    # For M20-A, we will assume obs_tensor is the tensor passed TO the model.
    # If the model is a simple MLP (like in DQNLite), it takes flat input.
    # So the gradient will be flat. We reshape it at the end.

    # 1. Prepare tensor
    if not obs_tensor.requires_grad:
        # We need a leaf tensor with requires_grad=True
        # We clone it and detach to start a fresh graph
        input_tensor = obs_tensor.clone().detach().requires_grad_(True)
    else:
        input_tensor = obs_tensor

    # Add batch dim if missing
    if len(input_tensor.shape) == 1:
        batch_input = input_tensor.unsqueeze(0)
    else:
        batch_input = input_tensor

    # 2. Forward pass
    # Zero gradients first (though we are on a fresh tensor usually)
    model.zero_grad()

    output = model(batch_input)

    # Output is typically (Batch, ActionSpace) logits/Q-values
    # Or (logits, value) tuple for PPO?
    # DQNLite returns Q-values directly.
    # PPOLite actor returns logits directly (if we access actor network).
    # We need to handle tuple outputs if necessary.

    if isinstance(output, tuple):
        # PPO might return (action_logits, value) or similar
        # Let's assume index 0 is policy logits
        target_output = output[0]
    else:
        target_output = output

    # 3. Select scalar target
    # We want the value for 'action_index'
    target_scalar = target_output[0, action_index]

    # 4. Backward pass
    target_scalar.backward()

    # 5. Get gradients
    grad = input_tensor.grad
    if grad is None:
        # Should not happen if graph is connected
        return np.zeros((SUIT_COUNT, RANK_COUNT))

    # 6. Process gradients
    grad_np = grad.detach().cpu().numpy()

    # Handle absolute magnitude
    abs_grad = np.abs(grad_np)

    # If batch, squeeze (we assume single obs for overlay)
    if len(abs_grad.shape) > 1 and abs_grad.shape[0] == 1:
        abs_grad = abs_grad.squeeze(0)

    # Reshape to (4, 14, N)
    # We need to infer N.
    size = abs_grad.size
    plane_size = SUIT_COUNT * RANK_COUNT

    if size % plane_size != 0:
        # Fallback / Error
        return np.zeros((SUIT_COUNT, RANK_COUNT))

    n_planes = size // plane_size
    # Reshape
    # Important: Flatten order. PyTorch flatten is row-major (C-style).
    # If original shape was (4, 14, N) then flatten order depends.
    # UNGAR usually flattens (4, 14, N) -> (4*14*N).
    # So reshaping back to (4, 14, N) is correct if it was (4, 14, N).

    # BUT wait, the test failure says I got 2.0 instead of 5.0.
    # This means I'm only seeing the contribution from the first plane?
    # Or reshaping is wrong?

    # If I reshape to (4, 14, N), then [0, 0, 0] is first element.
    # [0, 0, 1] is second element? No.
    # A standard reshape(4, 14, N) fills the last dimension fastest.
    # So index 0 -> (0,0,0), index 1 -> (0,0,1).
    # But 56 is (4*14).
    # If N=2, total=112.
    # If reshape(4, 14, 2):
    # (0,0,0) is index 0.
    # (0,0,1) is index 1.
    # (0,1,0) is index 2.

    # My test setup:
    # weight[0, 0] = 2.0 (index 0)
    # weight[0, 56] = 3.0 (index 56)

    # If I reshape (4, 14, 2), index 56 corresponds to...
    # 56 // 2 = 28 -> (28 // 14, 28 % 14) -> (2, 0).
    # So index 56 is (2, 0, 0).
    # This means the test assumption about index 56 being (0,0) in plane 1 is WRONG if the shape is (4, 14, N).

    # If the shape is (4, 14, N), then flattening traverses N first.
    # So plane 0 and plane 1 are interleaved.
    # But usually we stack planes? (N, 4, 14) or (4, 14, N)?
    # UNGAR Tensor is (4, 14, N).
    # When flattened, it goes (0,0,0), (0,0,1)... (0,0,N-1), (0,1,0)...

    # So if I want to hit (0,0) on plane 1, that is index 1 (if N=2).
    # If I used index 56, that is halfway through the array.

    # Let's fix the TEST assumption first. If I want "same card, different plane",
    # and shape is (4, 14, N), then indices are adjacent.
    # index 0 is (0,0, plane 0)
    # index 1 is (0,0, plane 1)

    structured_grad = abs_grad.reshape((SUIT_COUNT, RANK_COUNT, n_planes))

    # 7. Reduce channels (sum across planes)
    # We want (4, 14)
    # Summing magnitudes captures "sensitivity of this card across all feature planes"
    importance = np.sum(structured_grad, axis=2)

    # 8. Normalize (L1 / Sum-to-1)
    if normalize:
        total = np.sum(importance)
        if total > 1e-9:
            importance = importance / total
        else:
            # Degenerate case: no gradient. Return zeros.
            pass

    return cast(np.ndarray, importance)


def compute_value_grad_importance(
    model: nn.Module,
    obs_tensor: torch.Tensor,
    *,
    value_index: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute gradient-based importance map for a value/critic output.

    For PPO: Gradients of V(s) w.r.t. s (state value function).
    For multi-head critics: Use value_index to select which output head.

    Args:
        model: The PyTorch model (Critic or ActorCritic with get_value).
                Should output scalar value(s) per observation.
        obs_tensor: Input observation tensor, typically flat (input_dim,) or batched (B, input_dim).
        value_index: Optional index if model outputs multiple values. None means single output or output[0].
        normalize: Whether to normalize the result to sum to 1 (L1 norm).

    Returns:
        A (4, 14) float numpy array representing importance of each card position.
    """
    model.eval()

    # 1. Prepare input tensor with gradient tracking
    if not obs_tensor.requires_grad:
        input_tensor = obs_tensor.clone().detach().requires_grad_(True)
    else:
        input_tensor = obs_tensor

    # Add batch dimension if missing
    if len(input_tensor.shape) == 1:
        batch_input = input_tensor.unsqueeze(0)
    else:
        batch_input = input_tensor

    # 2. Forward pass
    model.zero_grad()

    # Try to call get_value if available (PPO ActorCritic), otherwise direct call
    if hasattr(model, "get_value"):
        output = model.get_value(batch_input)
    else:
        output = model(batch_input)

    # 3. Select scalar target
    # Value output is typically (Batch, 1) or (Batch,) or just scalar
    if isinstance(output, tuple):
        # Some architectures might return (value, ...) tuples
        # Assume first element is the value
        target_output = output[0]
    else:
        target_output = output

    # Handle different output shapes
    if len(target_output.shape) == 0:
        # Scalar output
        target_scalar = target_output
    elif len(target_output.shape) == 1:
        # (Batch,) shape
        if value_index is not None:
            target_scalar = target_output[value_index]
        else:
            target_scalar = target_output[0]
    elif len(target_output.shape) == 2:
        # (Batch, num_values) shape
        if value_index is not None:
            target_scalar = target_output[0, value_index]
        else:
            # Default to first output (or squeeze if single value head)
            target_scalar = target_output[0, 0] if target_output.shape[1] > 0 else target_output[0]
    else:
        # Unexpected shape, try to get first element
        target_scalar = target_output.flatten()[0]

    # 4. Backward pass
    target_scalar.backward()

    # 5. Get gradients
    grad = input_tensor.grad
    if grad is None:
        # No gradient computed (disconnected graph)
        return np.zeros((SUIT_COUNT, RANK_COUNT))

    # 6. Process gradients to (4, 14) importance map
    grad_np = grad.detach().cpu().numpy()

    # Absolute magnitude (we care about sensitivity, not direction)
    abs_grad = np.abs(grad_np)

    # Remove batch dimension if present
    if len(abs_grad.shape) > 1 and abs_grad.shape[0] == 1:
        abs_grad = abs_grad.squeeze(0)

    # 7. Reshape from flat to (4, 14, N) structure
    size = abs_grad.size
    plane_size = SUIT_COUNT * RANK_COUNT

    if size % plane_size != 0:
        # Input doesn't match expected card tensor structure
        return np.zeros((SUIT_COUNT, RANK_COUNT))

    n_planes = size // plane_size
    structured_grad = abs_grad.reshape((SUIT_COUNT, RANK_COUNT, n_planes))

    # 8. Reduce across feature planes (sum absolute gradients)
    importance = np.sum(structured_grad, axis=2)

    # 9. Normalize (L1 norm = 1.0)
    if normalize:
        total = np.sum(importance)
        if total > 1e-9:
            importance = importance / total
        # else: keep as zeros if no gradient

    return cast(np.ndarray, importance)
