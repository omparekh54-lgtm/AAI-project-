from __future__ import annotations
import random
import numpy as np
import torch
from torch import nn, optim
from .replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)


class SharedDQNAgent:
    """Parameter-shared DQN used by the homogeneous cab agents."""
    def __init__(self, state_size=16, action_size=6, gamma=0.95, lr=1e-3,
                 batch_size=64, seed=42, target_update=100):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update = target_update
        self.policy = QNetwork(state_size, action_size)
        self.target = QNetwork(state_size, action_size)
        self.target.load_state_dict(self.policy.state_dict())
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.learn_steps = 0

    def act(self, state, valid_actions=None, explore=True):
        valid_actions = list(range(self.action_size)) if valid_actions is None else list(valid_actions)
        if explore and random.random() < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            q = self.policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[0].numpy()
        return max(valid_actions, key=lambda a: q[a])

    def remember(self, *transition):
        self.memory.add(*transition)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample(self.batch_size)
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)
        q = self.policy(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target(next_states).max(1).values
            target = rewards + self.gamma * next_q * (1 - dones)
        loss = nn.functional.smooth_l1_loss(q, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optim.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())
