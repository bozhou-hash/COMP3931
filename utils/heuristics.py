from config import (
    ROWS,
    COLS,
    CONNECT_N,
    EMPTY,
    PLAYER_ONE,
    PLAYER_TWO,
)


def get_opponent(player):
    return PLAYER_TWO if player == PLAYER_ONE else PLAYER_ONE


def score_window(window, player):
    opponent = get_opponent(player)

    player_count = window.count(player)
    opponent_count = window.count(opponent)
    empty_count = window.count(EMPTY)

    score = 0

    if player_count == CONNECT_N:
        score += 100000
    elif player_count == CONNECT_N - 1 and empty_count == 1:
        score += 100
    elif player_count == CONNECT_N - 2 and empty_count == 2:
        score += 10

    if opponent_count == CONNECT_N:
        score -= 100000
    elif opponent_count == CONNECT_N - 1 and empty_count == 1:
        score -= 120
    elif opponent_count == CONNECT_N - 2 and empty_count == 2:
        score -= 12

    return score


def evaluate_board(board, player):
    score = 0

    # Center column preference
    center_col = COLS // 2
    center_values = board[:, center_col].tolist()
    score += center_values.count(player) * 3

    # Horizontal
    for row in range(ROWS):
        for col in range(COLS - CONNECT_N + 1):
            window = board[row, col:col + CONNECT_N].tolist()
            score += score_window(window, player)

    # Vertical
    for row in range(ROWS - CONNECT_N + 1):
        for col in range(COLS):
            window = board[row:row + CONNECT_N, col].tolist()
            score += score_window(window, player)

    # Positive diagonal
    for row in range(ROWS - CONNECT_N + 1):
        for col in range(COLS - CONNECT_N + 1):
            window = [board[row + offset, col + offset] for offset in range(CONNECT_N)]
            score += score_window(window, player)

    # Negative diagonal
    for row in range(CONNECT_N - 1, ROWS):
        for col in range(COLS - CONNECT_N + 1):
            window = [board[row - offset, col + offset] for offset in range(CONNECT_N)]
            score += score_window(window, player)

    return score