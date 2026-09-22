from __future__ import annotations
import random
import numpy as np
import torch
from torch import nn, optim
from .dqn import QNetwork


class VDNAgent:
    """Lightweight Value Decomposition Network for cooperative cab dispatch.

    Each cab has a local Q-value and the joint team value is their sum. Training
    is centralised over a joint transition while execution remains local.
    """
    def __init__(self, n_agents=8, state_size=16, action_size=6, gamma=0.95,
                 lr=1e-3, batch_size=32, seed=42):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.n_agents = n_agents
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.policy = QNetwork(state_size, action_size)
        self.target = QNetwork(state_size, action_size)
        self.target.load_state_dict(self.policy.state_dict())
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []
        self.learn_steps = 0

    def act(self, states, valid_actions):
        actions = []
        for i, state in enumerate(states):
            valid = list(valid_actions[i])
            if random.random() < self.epsilon:
                actions.append(random.choice(valid))
            else:
                with torch.no_grad():
                    q = self.policy(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[0]
                actions.append(max(valid, key=lambda a: float(q[a])))
        return actions

    def remember(self, states, actions, rewards, next_states, next_valid_actions, done):
        self.memory.append((np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.int64),
                            np.asarray(rewards, dtype=np.float32), np.asarray(next_states, dtype=np.float32),
                            next_valid_actions, float(done)))
        if len(self.memory) > 20000:
            self.memory.pop(0)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
        actions = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.long)
        rewards = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
        next_states = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b[5] for b in batch], dtype=torch.float32)

        bsz, nag, st = states.shape
        q_all = self.policy(states.reshape(bsz * nag, st)).reshape(bsz, nag, self.action_size)
        chosen = q_all.gather(2, actions.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        with torch.no_grad():
            next_all = self.target(next_states.reshape(bsz * nag, st)).reshape(bsz, nag, self.action_size)
            next_joint = next_all.max(dim=2).values.sum(dim=1)
            team_reward = rewards.sum(dim=1)
            target = team_reward + self.gamma * next_joint * (1 - dones)
        loss = nn.functional.smooth_l1_loss(chosen, target)
        self.optim.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0); self.optim.step()
        self.learn_steps += 1
        if self.learn_steps % 100 == 0:
            self.target.load_state_dict(self.policy.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())
