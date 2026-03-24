import random

import numpy as np
import torch

from config import COLS


class DoubleDQNAgent:
    def __init__(self, player, q_network, epsilon=0.1, device="cpu"):
        self.player = player
        self.q_network = q_network
        self.epsilon = epsilon
        self.device = device

    def select_action(self, env):
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        state_tensor = self._get_state_tensor(env)

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)

        masked_q_values = self._mask_illegal_actions(q_values, legal_actions)
        return int(torch.argmax(masked_q_values).item())

    def _get_state_tensor(self, env):
        board = env.get_board().astype(np.float32).flatten()
        board = board * self.player
        state_tensor = torch.tensor(board, dtype=torch.float32, device=self.device)
        return state_tensor.unsqueeze(0)

    def _mask_illegal_actions(self, q_values, legal_actions):
        masked_q_values = q_values.clone()

        illegal_actions = [col for col in range(COLS) if col not in legal_actions]
        for action in illegal_actions:
            masked_q_values[action] = float("-inf")

        return masked_q_values