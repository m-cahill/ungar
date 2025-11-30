"""PPO-Lite Implementation.

A minimal, dependency-free PyTorch PPO implementation for UNGAR.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from ungar.training.config import PPOConfig

from .unified_agent import Transition


@dataclass
class PPOTransition:
    """Transition data needed for PPO updates."""

    obs: np.ndarray
    action: int
    logprob: float
    reward: float
    done: bool
    value: float
    legal_moves_mask: np.ndarray


class ActorCritic(nn.Module):
    """Shared backbone Actor-Critic network."""

    def __init__(self, input_dim: int, action_space_size: int) -> None:
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_space_size)
        self.critic = nn.Linear(64, 1)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.feature_net(x))

    def get_action_and_value(
        self, x: torch.Tensor, legal_masks: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.feature_net(x)
        logits = self.actor(features)

        # Mask illegal moves with large negative number
        # legal_masks is 1 for legal, 0 for illegal.
        # We want to add -inf where illegal (mask=0).
        inf_mask = (1.0 - legal_masks) * -1e8
        logits = logits + inf_mask

        probs = Categorical(logits=logits)
        if action is None:
            action_tensor = probs.sample()
        else:
            # Normalize any incoming type to a proper LongTensor on the same device
            action_tensor = torch.as_tensor(action, dtype=torch.long, device=logits.device)

        return action_tensor, probs.log_prob(action_tensor), probs.entropy(), self.critic(features)


class PPOLiteAgent:
    """Lightweight PPO Agent implementing UnifiedAgent protocol."""

    def __init__(
        self,
        input_dim: int,
        action_space_size: int,
        config: PPOConfig,
    ) -> None:
        if config.seed is not None:
            random.seed(config.seed)
            np.random.seed(config.seed)
            torch.manual_seed(config.seed)

        self.config = config
        self.action_space_size = action_space_size

        self.network = ActorCritic(input_dim, action_space_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)

        # Buffer for current rollout
        self.buffer: List[PPOTransition] = []

    def act(self, obs: np.ndarray, legal_moves: List[int]) -> int:
        """Select action using policy network."""
        # Create full mask for legal moves
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        mask[legal_moves] = 1.0

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            mask_t = torch.FloatTensor(mask).unsqueeze(0)
            action, logprob, _, value = self.network.get_action_and_value(obs_t, mask_t)

        action_idx = int(action.item())

        # Store transition data temporarily (waiting for reward/done)
        # We handle storage in `train_step`, but act needs to pass info?
        # Actually, UnifiedAgent protocol act() returns int.
        # We need a way to store logprob/value for the training step.
        # We'll store it in a temporary variable "last_act_info".
        self._last_act_info = {"logprob": logprob.item(), "value": value.item(), "mask": mask}

        return action_idx

    def train_step(self, transition: Transition) -> None:
        """Store transition in buffer. Real update happens when buffer full."""
        # Reconstruct full info using stored last act info
        if not hasattr(self, "_last_act_info"):
            return  # Should not happen in standard loop

        info = self._last_act_info

        ppo_trans = PPOTransition(
            obs=transition.obs,
            action=transition.action,
            logprob=float(info["logprob"]),
            reward=transition.reward,
            done=transition.done,
            value=float(info["value"]),
            legal_moves_mask=np.array(info["mask"]),
        )
        self.buffer.append(ppo_trans)

    def update(self) -> dict[str, float]:
        """Perform PPO update using collected buffer."""
        if not self.buffer:
            return {}

        # Convert buffer to tensors
        obs = torch.FloatTensor(np.array([t.obs for t in self.buffer]))
        actions = torch.LongTensor(np.array([t.action for t in self.buffer]))
        logprobs = torch.FloatTensor(np.array([t.logprob for t in self.buffer]))
        rewards = torch.FloatTensor(np.array([t.reward for t in self.buffer]))
        dones = torch.FloatTensor(np.array([float(t.done) for t in self.buffer]))
        values = torch.FloatTensor(np.array([t.value for t in self.buffer]))
        masks = torch.FloatTensor(np.array([t.legal_moves_mask for t in self.buffer]))

        # GAE Calculation
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0.0

        # For simplicity in Lite version, assume buffer is one continuous episode
        # or handle boundaries with dones.
        # Ideally we compute GAE backwards.
        # We need next_value for the very last step if not done.
        # For this implementation, assume last step is terminal or ignore bootstrap
        # for simplicity (or use 0). Proper way needs next_val.
        # Let's assume 0 for last step next val to keep it lite.

        for t in reversed(range(len(self.buffer))):
            if t == len(self.buffer) - 1:
                nextnonterminal = 1.0 - float(dones[t])
                nextvalues_val = 0.0  # Simplification
            else:
                nextnonterminal = 1.0 - float(dones[t])
                nextvalues_val = float(values[t + 1])

            delta = (
                float(rewards[t])
                + self.config.gamma * nextvalues_val * nextnonterminal
                - float(values[t])
            )
            advantages[t] = lastgaelam = (
                delta + self.config.gamma * self.config.gae_lambda * nextnonterminal * lastgaelam
            )

        returns = advantages + values

        # Update Loop
        b_inds = np.arange(len(self.buffer))

        for _ in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(self.buffer), self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(
                    obs[mb_inds], masks[mb_inds], actions[mb_inds]
                )

                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = advantages[mb_inds]
                # Normalize advantage
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                v_loss = 0.5 * ((newvalue.view(-1) - returns[mb_inds]) ** 2).mean()

                # Entropy Loss
                entropy_loss = entropy.mean()

                loss = (
                    pg_loss
                    - self.config.entropy_coef * entropy_loss
                    + self.config.value_coef * v_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Clear buffer
        self.buffer = []

        return {"loss": float(loss.item())}

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path))
