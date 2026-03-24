import sys
import pygame

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
        pygame.init()
        pygame.display.set_caption(GAME_TITLE)

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

        help_surface = self.small_font.render(
            "Click a column to drop a piece   |   R = restart   |   ESC = quit",
            True,
            TEXT_COLOR
        )
        help_rect = help_surface.get_rect(center=(SCREEN_WIDTH // 2, 55))
        self.screen.blit(help_surface, help_rect)

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
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, TOTAL_SCREEN_HEIGHT // 2 - 50))
        self.screen.blit(text_surface, text_rect)

        stats_text = f"P1: {self.player_one_wins}   P2: {self.player_two_wins}   Draws: {self.draws}"
        stats_surface = self.small_font.render(stats_text, True, TEXT_COLOR)
        stats_rect = stats_surface.get_rect(center=(SCREEN_WIDTH // 2, TOTAL_SCREEN_HEIGHT // 2 - 10))
        self.screen.blit(stats_surface, stats_rect)

        button_rect = pygame.Rect(
            SCREEN_WIDTH // 2 - 110,
            TOTAL_SCREEN_HEIGHT // 2 + 25,
            220,
            55
        )

        mouse_pos = pygame.mouse.get_pos()
        button_color = BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else BUTTON_COLOR

        pygame.draw.rect(self.screen, button_color, button_rect, border_radius=12)

        button_text = self.button_font.render("Restart Game", True, BUTTON_TEXT)
        button_text_rect = button_text.get_rect(center=button_rect.center)
        self.screen.blit(button_text, button_text_rect)

        self.restart_button_rect = button_rect

    @staticmethod
    def get_clicked_column(mouse_x):
        col = mouse_x // CELL_SIZE
        if 0 <= col < COLS:
            return col
        return None

    def run_human_vs_human(self):
        running = True
        self.restart_button_rect = None

        while running:
            self.clock.tick(FPS)
            self.draw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.env.reset()
                        self._result_recorded = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.env.game_over:
                        if self.restart_button_rect and self.restart_button_rect.collidepoint(event.pos):
                            self.env.reset()
                            self._result_recorded = False
                    else:
                        mouse_x, _ = event.pos
                        col = self.get_clicked_column(mouse_x)
                        if col is not None:
                            self.env.drop_piece(col)

        pygame.quit()
        sys.exit()

    def run_human_vs_agent(self, agent, human_player=PLAYER_ONE):
        running = True
        self.restart_button_rect = None

        agent_delay = 500  # milliseconds
        agent_move_time = None
        pending_agent_action = None

        while running:
            self.clock.tick(FPS)
            self.draw()

            # Agent move with delay
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
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.env.reset()
                        self._result_recorded = False
                        pending_agent_action = None
                        agent_move_time = None

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.env.game_over:
                        if self.restart_button_rect and self.restart_button_rect.collidepoint(event.pos):
                            self.env.reset()
                            self._result_recorded = False
                            pending_agent_action = None
                            agent_move_time = None
                    else:
                        if self.env.get_current_player() == human_player:
                            mouse_x, _ = event.pos
                            col = self.get_clicked_column(mouse_x)
                            if col is not None:
                                self.env.drop_piece(col)

            pygame.display.flip()

        pygame.quit()
        sys.exit()