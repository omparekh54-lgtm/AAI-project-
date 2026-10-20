from __future__ import annotations
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
    def add(self, state, action, reward, next_state, done, next_valid_actions, discount=None):
        # discount: gamma**k for a transition lasting k minutes (semi-MDP); None means one step.
        self.buffer.append((state, action, reward, next_state, done, tuple(next_valid_actions), discount))
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)
