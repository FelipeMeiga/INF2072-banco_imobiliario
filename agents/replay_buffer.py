from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Transition:
    state_action: np.ndarray
    acting_player: int
    reward: float
    next_state: np.ndarray
    next_actions: List[np.ndarray]
    next_acting_player: int
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000, seed: int | None = None):
        self.memory = deque(maxlen=capacity)
        self.random = random.Random(seed)

    def push(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size: int):
        return self.random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
