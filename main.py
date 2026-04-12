import pygame

from env.connect4_env import Connect4Env
from ui.game_ui import GameUI
from config import GAME_TITLE


def main():
    pygame.init()
    pygame.display.set_caption(GAME_TITLE)

    env = Connect4Env()
    ui = GameUI(env)
    ui.run()


if __name__ == "__main__":
    main()