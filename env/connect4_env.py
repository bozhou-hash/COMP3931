import numpy as np

from config import ROWS, COLS, CONNECT_N, EMPTY, PLAYER_ONE


class Connect4Env:
    def __init__(self):
        self.starting_player = -PLAYER_ONE
        self.board = None
        self.current_player = None
        self.winner = None
        self.game_over = False
        self.reset()

    def reset(self):
        """
        Reset the board for a new game.
        Alternate the starting player each time reset() is called.
        """
        self.board = np.zeros((ROWS, COLS), dtype=int)
        self.starting_player *= -1
        self.current_player = self.starting_player
        self.winner = None
        self.game_over = False
        return self.board.copy()

    def get_board(self):
        return self.board.copy()

    def get_current_player(self):
        return self.current_player

    def get_legal_actions(self):
        return [col for col in range(COLS) if self.board[0, col] == EMPTY]

    def is_valid_action(self, col):
        return 0 <= col < COLS and self.board[0, col] == EMPTY

    def drop_piece(self, col):
        """
        Drop a piece for the current player into the given column.
        Returns (row, col) if successful, otherwise None.
        """
        if not self.is_valid_action(col) or self.game_over:
            return None

        for row in range(ROWS - 1, -1, -1):
            if self.board[row, col] == EMPTY:
                self.board[row, col] = self.current_player

                if self.check_winner(row, col):
                    self.winner = self.current_player
                    self.game_over = True
                elif self.is_draw():
                    self.winner = 0
                    self.game_over = True
                else:
                    self.current_player *= -1

                return row, col

        return None

    def is_draw(self):
        return np.all(self.board[0] != EMPTY)

    def check_winner(self, row, col):
        """
        Check whether the piece placed at (row, col) creates CONNECT_N in a row.
        """
        player = self.board[row, col]
        if player == EMPTY:
            return False

        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal down-right
            (1, -1),  # diagonal down-left
        ]

        for dr, dc in directions:
            count = 1
            count += self._count_direction(row, col, dr, dc, player)
            count += self._count_direction(row, col, -dr, -dc, player)

            if count >= CONNECT_N:
                return True

        return False

    def _count_direction(self, row, col, dr, dc, player):
        count = 0
        r, c = row + dr, col + dc

        while 0 <= r < ROWS and 0 <= c < COLS and self.board[r, c] == player:
            count += 1
            r += dr
            c += dc

        return count

    def get_result(self):
        """
        Returns:
            1  -> PLAYER_ONE wins
           -1  -> PLAYER_TWO wins
            0  -> draw
          None -> game still ongoing
        """
        if not self.game_over:
            return None
        return self.winner

    def clone(self):
        cloned_env = Connect4Env()
        cloned_env.starting_player = self.starting_player
        cloned_env.board = self.board.copy()
        cloned_env.current_player = self.current_player
        cloned_env.winner = self.winner
        cloned_env.game_over = self.game_over
        return cloned_env