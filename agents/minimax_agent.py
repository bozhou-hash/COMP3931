import copy
import math

from config import PLAYER_ONE, PLAYER_TWO, COLS
from utils.heuristics import evaluate_board


class MinimaxAgent:
    def __init__(self, player, depth=4):
        self.player = player
        self.depth = depth

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
            return self._evaluate_terminal_state(env, result, depth), None

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

    def _evaluate_terminal_state(self, env, result, depth):
        if result == self.player:
            return 1000000 + depth

        if result == self._get_opponent():
            return -1000000 - depth

        if result == 0:
            return 0

        return evaluate_board(env.get_board(), self.player)

    def _get_opponent(self):
        return PLAYER_TWO if self.player == PLAYER_ONE else PLAYER_ONE

    def _order_actions(self, legal_actions):
        center_col = COLS // 2
        return sorted(legal_actions, key=lambda col: abs(col - center_col))