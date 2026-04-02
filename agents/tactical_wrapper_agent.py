from config import PLAYER_ONE, PLAYER_TWO


class TacticalWrapperAgent:
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.player = base_agent.player

    def select_action(self, env):
        legal_actions = env.get_legal_actions()
        opponent = PLAYER_TWO if self.player == PLAYER_ONE else PLAYER_ONE

        # 1. If we can win immediately, do it
        for action in legal_actions:
            env_copy = env.clone()
            env_copy.current_player = self.player
            env_copy.drop_piece(action)

            if env_copy.get_result() == self.player:
                return action

        # 2. If opponent can win immediately, block that move
        opponent_winning_actions = []
        for action in legal_actions:
            env_copy = env.clone()
            env_copy.current_player = opponent
            env_copy.drop_piece(action)

            if env_copy.get_result() == opponent:
                opponent_winning_actions.append(action)

        if opponent_winning_actions:
            # Block the opponent's immediate winning column
            return opponent_winning_actions[0]

        # 3. Otherwise fall back to the wrapped agent
        return self.base_agent.select_action(env)