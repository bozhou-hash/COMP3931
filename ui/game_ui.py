import os
import sys
import pygame
import torch

from agents.random_agent import RandomAgent
from agents.minimax_agent import MinimaxAgent
from agents.dqn_agent import DQNAgent
from agents.double_dqn_agent import DoubleDQNAgent
from agents.hybrid_agent import HybridAgent
from models.q_network import QNetwork

from config import (
    ROWS,
    COLS,
    CELL_SIZE,
    TOP_BAR_HEIGHT,
    PREVIEW_ROW_HEIGHT,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    TOTAL_SCREEN_HEIGHT,
    FPS,
    GAME_TITLE,
    PLAYER_ONE,
    PLAYER_TWO,
    BACKGROUND_COLOR,
    TOP_BAR_COLOR,
    BOARD_BLUE_DARK,
    BOARD_BLUE_LIGHT,
    BOARD_SHADOW,
    HOLE_OUTER,
    HOLE_INNER,
    PLAYER_ONE_COLOR,
    PLAYER_TWO_COLOR,
    TEXT_COLOR,
    OVERLAY_COLOR,
    BUTTON_COLOR,
    BUTTON_HOVER,
    BUTTON_TEXT,
    SEPARATOR_COLOR,
    DISC_SHADOW_COLOR,
    PREVIEW_ALPHA,
    TITLE_FONT_NAME,
    BODY_FONT_NAME,
    BUTTON_FONT_NAME,
    TITLE_FONT_SIZE,
    BODY_FONT_SIZE,
    BUTTON_FONT_SIZE,
    TITLE_FONT_BOLD,
    BODY_FONT_BOLD,
    BUTTON_FONT_BOLD,
)


class GameUI:
    def __init__(self, env):
        self.env = env
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, TOTAL_SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(TITLE_FONT_NAME, TITLE_FONT_SIZE, bold=TITLE_FONT_BOLD)
        self.small_font = pygame.font.SysFont(BODY_FONT_NAME, BODY_FONT_SIZE, bold=BODY_FONT_BOLD)
        self.button_font = pygame.font.SysFont(BUTTON_FONT_NAME, BUTTON_FONT_SIZE, bold=BUTTON_FONT_BOLD)

        self.player_one_wins = 0
        self.player_two_wins = 0
        self.draws = 0
        self._result_recorded = False

        self.restart_button_rect = None
        self.menu_button_rect = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.selected_mode_name = None
        self.current_agent = None
        self.current_human_player = PLAYER_ONE

    # =========================
    # Model loading utilities
    # =========================
    @staticmethod
    def get_project_root():
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def get_model_path(self, folder_name, filename):
        return os.path.join(self.get_project_root(), "training", folder_name, filename)

    def load_q_network(self, model_path):
        q_network = QNetwork().to(self.device)
        loaded_object = torch.load(model_path, map_location=self.device, weights_only=False)

        if isinstance(loaded_object, dict) and "q_network_state_dict" in loaded_object:
            q_network.load_state_dict(loaded_object["q_network_state_dict"])
        elif isinstance(loaded_object, dict):
            q_network.load_state_dict(loaded_object)
        else:
            raise ValueError(f"Unsupported model format: {model_path}")

        q_network.eval()
        return q_network

    def load_dqn_agent(self, model_folder, model_filename, player):
        model_path = self.get_model_path(model_folder, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DQN model file not found: {model_path}")

        q_network = self.load_q_network(model_path)

        return DQNAgent(
            player=player,
            q_network=q_network,
            epsilon=0.0,
            device=self.device,
        )

    def load_double_dqn_agent(self, model_folder, model_filename, player):
        model_path = self.get_model_path(model_folder, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Double DQN model file not found: {model_path}")

        q_network = self.load_q_network(model_path)

        return DoubleDQNAgent(
            player=player,
            q_network=q_network,
            epsilon=0.0,
            device=self.device,
        )

    def load_hybrid_agent(self, model_folder, model_filename, player, depth=4):
        model_path = self.get_model_path(model_folder, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Hybrid model file not found: {model_path}")

        q_network = self.load_q_network(model_path)

        return HybridAgent(
            player=player,
            q_network=q_network,
            device=self.device,
            depth=depth,
            neural_weight=0.7,
            heuristic_weight=0.3,
        )

    # =========================
    # General helpers
    # =========================
    def reset_match(self):
        self.env.reset()
        self._result_recorded = False
        self.restart_button_rect = None
        self.menu_button_rect = None

    def _update_record_if_needed(self):
        if not self.env.game_over or self._result_recorded:
            return

        if self.env.winner == PLAYER_ONE:
            self.player_one_wins += 1
        elif self.env.winner == PLAYER_TWO:
            self.player_two_wins += 1
        else:
            self.draws += 1

        self._result_recorded = True

    @staticmethod
    def get_clicked_column(mouse_x):
        col = mouse_x // CELL_SIZE
        if 0 <= col < COLS:
            return col
        return None

    def _draw_button(self, rect, text):
        mouse_pos = pygame.mouse.get_pos()
        color = BUTTON_HOVER if rect.collidepoint(mouse_pos) else BUTTON_COLOR
        pygame.draw.rect(self.screen, color, rect, border_radius=12)

        text_surface = self.button_font.render(text, True, BUTTON_TEXT)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)

    # =========================
    # Menu
    # =========================
    def draw_menu(self):
        self.screen.fill(BACKGROUND_COLOR)

        title_surface = self.font.render("Connect Four 6x5", True, TEXT_COLOR)
        title_rect = title_surface.get_rect(center=(SCREEN_WIDTH // 2, 70))
        self.screen.blit(title_surface, title_rect)

        subtitle_surface = self.small_font.render("Choose an opponent to begin", True, TEXT_COLOR)
        subtitle_rect = subtitle_surface.get_rect(center=(SCREEN_WIDTH // 2, 110))
        self.screen.blit(subtitle_surface, subtitle_rect)

        button_width = 320
        button_height = 55
        start_y = 170
        gap = 18
        x = (SCREEN_WIDTH - button_width) // 2

        buttons = [
            ("Human vs Human", pygame.Rect(x, start_y + 0 * (button_height + gap), button_width, button_height)),
            ("Human vs Random", pygame.Rect(x, start_y + 1 * (button_height + gap), button_width, button_height)),
            ("Human vs Minimax", pygame.Rect(x, start_y + 2 * (button_height + gap), button_width, button_height)),
            ("Human vs DQN", pygame.Rect(x, start_y + 3 * (button_height + gap), button_width, button_height)),
            ("Human vs Double DQN", pygame.Rect(x, start_y + 4 * (button_height + gap), button_width, button_height)),
            ("Human vs Hybrid", pygame.Rect(x, start_y + 5 * (button_height + gap), button_width, button_height)),
        ]

        for label, rect in buttons:
            self._draw_button(rect, label)

        pygame.display.flip()
        return buttons

    def create_agent_for_mode(self, mode_name):
        if mode_name == "Human vs Human":
            return None, PLAYER_ONE

        if mode_name == "Human vs Random":
            return RandomAgent(player=PLAYER_TWO), PLAYER_ONE

        if mode_name == "Human vs Minimax":
            return MinimaxAgent(player=PLAYER_TWO, depth=4), PLAYER_ONE

        if mode_name == "Human vs DQN":
            agent = self.load_dqn_agent(
                model_folder="checkpoints",
                model_filename="dqn_model_final.pth",
                player=PLAYER_TWO,
            )
            return agent, PLAYER_ONE

        if mode_name == "Human vs Double DQN":
            agent = self.load_double_dqn_agent(
                model_folder="double_dqn_checkpoints",
                model_filename="double_dqn_model_final.pth",
                player=PLAYER_TWO,
            )
            return agent, PLAYER_ONE

        if mode_name == "Human vs Hybrid":
            agent = self.load_hybrid_agent(
                model_folder="imitation_checkpoints",
                model_filename="dqn_imitation_tactical_depth6_model_final.pth",
                player=PLAYER_TWO,
                depth=4,
            )
            return agent, PLAYER_ONE

        raise ValueError(f"Unknown mode selected: {mode_name}")

    def run_menu(self):
        while True:
            self.clock.tick(FPS)
            buttons = self.draw_menu()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for label, rect in buttons:
                        if rect.collidepoint(event.pos):
                            self.selected_mode_name = label
                            self.current_agent, self.current_human_player = self.create_agent_for_mode(label)
                            self.reset_match()
                            return

    # =========================
    # Game drawing
    # =========================
    def draw(self):
        self._update_record_if_needed()
        self.screen.fill(BACKGROUND_COLOR)
        self._draw_top_bar()
        self._draw_board()
        self._draw_hover_preview()

        if self.env.game_over:
            self._draw_game_over_overlay()

        pygame.display.flip()

    def _draw_top_bar(self):
        pygame.draw.rect(
            self.screen,
            TOP_BAR_COLOR,
            (0, 0, SCREEN_WIDTH, TOP_BAR_HEIGHT)
        )

        if self.env.game_over:
            if self.env.winner == PLAYER_ONE:
                text = "Player 1 Wins!"
                color = PLAYER_ONE_COLOR
            elif self.env.winner == PLAYER_TWO:
                text = "Player 2 Wins!"
                color = PLAYER_TWO_COLOR
            else:
                text = "Draw!"
                color = TEXT_COLOR
        else:
            current_player = self.env.get_current_player()
            if current_player == PLAYER_ONE:
                text = "Player 1's Turn"
                color = PLAYER_ONE_COLOR
            else:
                text = "Player 2's Turn"
                color = PLAYER_TWO_COLOR

        title_surface = self.font.render(text, True, color)
        title_rect = title_surface.get_rect(center=(SCREEN_WIDTH // 2, 25))
        self.screen.blit(title_surface, title_rect)

        mode_text = self.selected_mode_name if self.selected_mode_name else "Menu"
        mode_surface = self.small_font.render(mode_text, True, TEXT_COLOR)
        mode_rect = mode_surface.get_rect(center=(SCREEN_WIDTH // 2, 55))
        self.screen.blit(mode_surface, mode_rect)

        record_text = f"P1: {self.player_one_wins}   P2: {self.player_two_wins}   Draws: {self.draws}"
        record_surface = self.small_font.render(record_text, True, TEXT_COLOR)
        record_rect = record_surface.get_rect(center=(SCREEN_WIDTH // 2, TOP_BAR_HEIGHT - 20))
        self.screen.blit(record_surface, record_rect)

        pygame.draw.line(
            self.screen,
            SEPARATOR_COLOR,
            (20, TOP_BAR_HEIGHT),
            (SCREEN_WIDTH - 20, TOP_BAR_HEIGHT),
            2
        )

    def _draw_board(self):
        board_top = TOP_BAR_HEIGHT + PREVIEW_ROW_HEIGHT
        board = self.env.get_board()

        for row in range(ROWS):
            for col in range(COLS):
                x = col * CELL_SIZE
                y = board_top + row * CELL_SIZE

                tile_color = BOARD_BLUE_DARK if (row + col) % 2 == 0 else BOARD_BLUE_LIGHT

                pygame.draw.rect(
                    self.screen,
                    BOARD_SHADOW,
                    (x + 4, y + 4, CELL_SIZE, CELL_SIZE),
                    border_radius=10
                )

                pygame.draw.rect(
                    self.screen,
                    tile_color,
                    (x, y, CELL_SIZE, CELL_SIZE),
                    border_radius=10
                )

                center = (x + CELL_SIZE // 2, y + CELL_SIZE // 2)

                pygame.draw.circle(
                    self.screen,
                    HOLE_OUTER,
                    (center[0] + 2, center[1] + 2),
                    CELL_SIZE // 2 - 10
                )
                pygame.draw.circle(
                    self.screen,
                    HOLE_INNER,
                    center,
                    CELL_SIZE // 2 - 12
                )

                piece = board[row, col]
                if piece == PLAYER_ONE:
                    self._draw_disc(PLAYER_ONE_COLOR, center)
                elif piece == PLAYER_TWO:
                    self._draw_disc(PLAYER_TWO_COLOR, center)

    def _draw_disc(self, color, center):
        x, y = center
        radius = CELL_SIZE // 2 - 12

        pygame.draw.circle(self.screen, DISC_SHADOW_COLOR, (x + 4, y + 4), radius)
        pygame.draw.circle(self.screen, color, (x, y), radius)

        light = (
            min(color[0] + 50, 255),
            min(color[1] + 50, 255),
            min(color[2] + 50, 255),
        )
        dark = (
            max(color[0] - 50, 0),
            max(color[1] - 50, 0),
            max(color[2] - 50, 0),
        )

        rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)
        pygame.draw.arc(self.screen, light, rect, 3.14, 4.71, 5)
        pygame.draw.arc(self.screen, dark, rect, 0, 3.14, 5)

    def _draw_hover_preview(self):
        if self.env.game_over:
            return

        if self.current_agent is not None and self.env.get_current_player() != self.current_human_player:
            return

        mouse_x, _ = pygame.mouse.get_pos()
        col = mouse_x // CELL_SIZE

        if not (0 <= col < COLS):
            return

        if not self.env.is_valid_action(col):
            return

        current_player = self.env.get_current_player()
        color = PLAYER_ONE_COLOR if current_player == PLAYER_ONE else PLAYER_TWO_COLOR

        preview_surface = pygame.Surface(
            (SCREEN_WIDTH, TOP_BAR_HEIGHT + PREVIEW_ROW_HEIGHT),
            pygame.SRCALPHA
        )
        preview_color = (*color, PREVIEW_ALPHA)

        center = (
            col * CELL_SIZE + CELL_SIZE // 2,
            TOP_BAR_HEIGHT + PREVIEW_ROW_HEIGHT // 2
        )

        pygame.draw.circle(
            preview_surface,
            preview_color,
            center,
            CELL_SIZE // 2 - 14
        )

        self.screen.blit(preview_surface, (0, 0))

    def _draw_game_over_overlay(self):
        overlay = pygame.Surface((SCREEN_WIDTH, TOTAL_SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill(OVERLAY_COLOR)
        self.screen.blit(overlay, (0, 0))

        if self.env.winner == PLAYER_ONE:
            message = "Player 1 Wins!"
            message_color = PLAYER_ONE_COLOR
        elif self.env.winner == PLAYER_TWO:
            message = "Player 2 Wins!"
            message_color = PLAYER_TWO_COLOR
        else:
            message = "It's a Draw!"
            message_color = TEXT_COLOR

        text_surface = self.font.render(message, True, message_color)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, TOTAL_SCREEN_HEIGHT // 2 - 70))
        self.screen.blit(text_surface, text_rect)

        stats_text = f"P1: {self.player_one_wins}   P2: {self.player_two_wins}   Draws: {self.draws}"
        stats_surface = self.small_font.render(stats_text, True, TEXT_COLOR)
        stats_rect = stats_surface.get_rect(center=(SCREEN_WIDTH // 2, TOTAL_SCREEN_HEIGHT // 2 - 30))
        self.screen.blit(stats_surface, stats_rect)

        restart_rect = pygame.Rect(
            SCREEN_WIDTH // 2 - 110,
            TOTAL_SCREEN_HEIGHT // 2 + 10,
            220,
            55
        )
        menu_rect = pygame.Rect(
            SCREEN_WIDTH // 2 - 110,
            TOTAL_SCREEN_HEIGHT // 2 + 80,
            220,
            55
        )

        self._draw_button(restart_rect, "Restart Game")
        self._draw_button(menu_rect, "Back to Menu")

        self.restart_button_rect = restart_rect
        self.menu_button_rect = menu_rect

    # =========================
    # Game loops
    # =========================
    def run_human_vs_human(self):
        running = True
        self.restart_button_rect = None
        self.menu_button_rect = None

        while running:
            self.clock.tick(FPS)
            self.draw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_r:
                        self.reset_match()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.env.game_over:
                        if self.restart_button_rect and self.restart_button_rect.collidepoint(event.pos):
                            self.reset_match()
                        elif self.menu_button_rect and self.menu_button_rect.collidepoint(event.pos):
                            return
                    else:
                        mouse_x, _ = event.pos
                        col = self.get_clicked_column(mouse_x)
                        if col is not None:
                            self.env.drop_piece(col)

    def run_human_vs_agent(self, agent, human_player=PLAYER_ONE):
        running = True
        self.restart_button_rect = None
        self.menu_button_rect = None

        agent_delay = 500
        agent_move_time = None
        pending_agent_action = None

        while running:
            self.clock.tick(FPS)
            self.draw()

            if not self.env.game_over and self.env.get_current_player() == agent.player:
                if pending_agent_action is None:
                    pending_agent_action = agent.select_action(self.env)
                    agent_move_time = pygame.time.get_ticks()

                if (
                    pending_agent_action is not None
                    and agent_move_time is not None
                    and pygame.time.get_ticks() - agent_move_time >= agent_delay
                ):
                    self.env.drop_piece(pending_agent_action)
                    pending_agent_action = None
                    agent_move_time = None
            else:
                pending_agent_action = None
                agent_move_time = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    elif event.key == pygame.K_r:
                        self.reset_match()
                        pending_agent_action = None
                        agent_move_time = None

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.env.game_over:
                        if self.restart_button_rect and self.restart_button_rect.collidepoint(event.pos):
                            self.reset_match()
                            pending_agent_action = None
                            agent_move_time = None
                        elif self.menu_button_rect and self.menu_button_rect.collidepoint(event.pos):
                            return
                    else:
                        if self.env.get_current_player() == human_player:
                            mouse_x, _ = event.pos
                            col = self.get_clicked_column(mouse_x)
                            if col is not None:
                                self.env.drop_piece(col)

    def run(self):
        while True:
            self.run_menu()

            if self.current_agent is None:
                self.run_human_vs_human()
            else:
                self.run_human_vs_agent(
                    agent=self.current_agent,
                    human_player=self.current_human_player,
                )