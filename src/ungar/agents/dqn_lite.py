"""DQN-Lite Implementation.

A minimal, dependency-free PyTorch DQN implementation for UNGAR.
"""

from __future__ import annotations

import collections
import random
from typing import Deque, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .unified_agent import Transition


class QNetwork(nn.Module):
    """Simple MLP for Q-learning."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = collections.deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNLiteAgent:
    """Lightweight DQN Agent implementing UnifiedAgent protocol."""

    def __init__(
        self,
        input_dim: int,
        action_space_size: int,
        lr: float = 0.0005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        epsilon_decay: int = 1000,
        buffer_size: int = 5000,
        batch_size: int = 32,
        target_update_tau: float = 0.1,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.action_space_size = action_space_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_tau = target_update_tau

        # Epsilon scheduling
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps = 0

        # Networks
        self.policy_net = QNetwork(input_dim, action_space_size)  # Renamed from q_net to standard
        self.target_net = QNetwork(input_dim, action_space_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Buffer
        self.memory = ReplayBuffer(buffer_size)

    def act(self, obs: np.ndarray, legal_moves: List[int]) -> int:
        """Select action using epsilon-greedy policy with legal move masking."""
        self.steps += 1

        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon -= (1.0 - self.epsilon_end) / self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_end)

        if not legal_moves:
            raise ValueError("No legal moves available.")

        # Exploration
        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        # Exploitation
        with torch.no_grad():
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            q_values = self.policy_net(state_t)

            # Mask illegal moves with -inf
            mask = torch.full_like(q_values, float("-inf"))
            mask[0, legal_moves] = 0
            masked_q = q_values + mask

            return int(masked_q.argmax(dim=1).item())

    def train_step(self, transition: Transition) -> None:
        """Store transition and update networks."""
        self.memory.push(transition)

        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)

        # Prepare batch
        states = torch.FloatTensor(np.array([t.obs for t in batch]))
        actions = torch.LongTensor([t.action for t in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([t.next_obs for t in batch]))
        dones = torch.FloatTensor([float(t.done) for t in batch]).unsqueeze(1)

        # Q(s, a)
        curr_q = self.policy_net(states).gather(1, actions)

        # max Q(s', a') masked by legal moves
        with torch.no_grad():
            next_q_raw = self.target_net(next_states)

            # We need to mask next_q values based on legal moves for each sample
            # Since legal moves differ per state, we iterate or use advanced indexing.
            # For simplicity in this Lite version, we iterate to build the max tensor.
            # Optimization: could vectorize if legal masks were part of transition.

            next_max_q = []
            for i, t in enumerate(batch):
                if t.done:
                    next_max_q.append(0.0)
                else:
                    legal = t.legal_moves_next
                    if not legal:
                        # Should not happen if done=False
                        next_max_q.append(0.0)
                    else:
                        q_vals = next_q_raw[i]
                        # Filter only legal indices
                        legal_vals = q_vals[legal]
                        next_max_q.append(legal_vals.max().item())

            next_max_q_t = torch.FloatTensor(next_max_q).unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_max_q_t

        # Loss and optimize
        loss = nn.MSELoss()(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target net
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.target_update_tau * param.data
                + (1.0 - self.target_update_tau) * target_param.data
            )

    def save(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
