from env.connect4_env import Connect4Env
from ui.game_ui import GameUI
from agents.random_agent import RandomAgent
from config import PLAYER_TWO


def main():
    env = Connect4Env()
    ui = GameUI(env)

    random_agent = RandomAgent(player=PLAYER_TWO)
    ui.run_human_vs_agent(agent=random_agent)


if __name__ == "__main__":
    main()