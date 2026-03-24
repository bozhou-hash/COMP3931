from env.connect4_env import Connect4Env
from ui.game_ui import GameUI
from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from config import PLAYER_ONE, PLAYER_TWO


def main():
    env = Connect4Env()
    ui = GameUI(env)

    # Run UI for human vs human
    # ui.run_human_vs_human()

    # Run UI for human vs random agent
    # random_agent = RandomAgent(player=PLAYER_TWO)
    # ui.run_human_vs_agent(agent=random_agent)

    # Run UI for human vs minimax agent
    minimax_agent = MinimaxAgent(player=PLAYER_TWO, depth=4)
    ui.run_human_vs_agent(minimax_agent, human_player=PLAYER_ONE)


if __name__ == "__main__":
    main()