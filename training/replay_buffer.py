import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, next_legal_actions, done):
        transition = (state, action, reward, next_state, next_legal_actions, done)
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, next_legal_actions_batch, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            list(next_legal_actions_batch),
            np.array(dones, dtype=np.float32),
        )

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "buffer": list(self.buffer),
        }

    def load_state_dict(self, state_dict):
        self.capacity = state_dict["capacity"]
        self.buffer = deque(state_dict["buffer"], maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)