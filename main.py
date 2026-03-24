from env.connect4_env import Connect4Env
from ui.game_ui import GameUI


def main():
    env = Connect4Env()
    ui = GameUI(env)
    ui.run_human_vs_human()


if __name__ == "__main__":
    main()