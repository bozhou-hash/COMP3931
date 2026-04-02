import copy
import math

import numpy as np
import torch

from config import PLAYER_ONE, PLAYER_TWO, COLS
from utils.heuristics import evaluate_board


class HybridAgent:
    def __init__(
        self,
        player,
        q_network,
        device="cpu",
        depth=4,
        neural_weight=0.7,
        heuristic_weight=0.3,
    ):
        self.player = player
        self.q_network = q_network
        self.device = device
        self.depth = depth
        self.neural_weight = neural_weight
        self.heuristic_weight = heuristic_weight

    def select_action(self, env):
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return None

        _, best_action = self._minimax(
            env=env,
            depth=self.depth,
            alpha=-math.inf,
            beta=math.inf,
            maximizing_player=True,
        )
        return best_action

    def _minimax(self, env, depth, alpha, beta, maximizing_player):
        legal_actions = env.get_legal_actions()
        result = env.get_result()

        if depth == 0 or result is not None or not legal_actions:
            return self._evaluate_state(env, result, depth), None

        if maximizing_player:
            best_score = -math.inf
            best_action = legal_actions[0]

            for action in self._order_actions(legal_actions):
                next_env = copy.deepcopy(env)
                next_env.drop_piece(action)

                score, _ = self._minimax(
                    env=next_env,
                    depth=depth - 1,
                    alpha=alpha,
                    beta=beta,
                    maximizing_player=False,
                )

                if score > best_score:
                    best_score = score
                    best_action = action

                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break

            return best_score, best_action

        best_score = math.inf
        best_action = legal_actions[0]

        for action in self._order_actions(legal_actions):
            next_env = copy.deepcopy(env)
            next_env.drop_piece(action)

            score, _ = self._minimax(
                env=next_env,
                depth=depth - 1,
                alpha=alpha,
                beta=beta,
                maximizing_player=True,
            )

            if score < best_score:
                best_score = score
                best_action = action

            beta = min(beta, best_score)
            if alpha >= beta:
                break

        return best_score, best_action

    def _evaluate_state(self, env, result, depth):
        if result == self.player:
            return 1000000 + depth

        if result == self._get_opponent():
            return -1000000 - depth

        if result == 0:
            return 0

        heuristic_score = evaluate_board(env.get_board(), self.player)
        neural_score = self._evaluate_with_network(env)

        return (
            self.neural_weight * neural_score
            + self.heuristic_weight * heuristic_score
        )

    def _evaluate_with_network(self, env):
        legal_actions = env.get_legal_actions()
        if not legal_actions:
            return 0.0

        state = self._get_state(env, self.player)
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)

        masked_q_values = self._mask_illegal_actions(q_values, legal_actions)

        # Use the best legal action's Q-value as a state score
        best_q_value = torch.max(masked_q_values).item()
        return float(best_q_value)

    def _get_state(self, env, player):
        board = env.get_board().astype(np.float32).flatten()
        return board * player

    def _mask_illegal_actions(self, q_values, legal_actions):
        masked_q_values = q_values.clone()

        illegal_actions = [col for col in range(COLS) if col not in legal_actions]
        for action in illegal_actions:
            masked_q_values[action] = float("-inf")

        return masked_q_values

    def _get_opponent(self):
        return PLAYER_TWO if self.player == PLAYER_ONE else PLAYER_ONE

    def _order_actions(self, legal_actions):
        center_col = COLS // 2
        return sorted(legal_actions, key=lambda col: abs(col - center_col))