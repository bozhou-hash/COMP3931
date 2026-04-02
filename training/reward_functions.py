from config import ROWS, COLS, CONNECT_N, EMPTY


def basic_reward(result, player):
    if result == player:
        return 1.0
    if result == 0:
        return 0.0
    if result is None:
        return 0.0
    return -1.0


def shaped_reward(env, player, last_move_col, result):
    """
    More informative reward:
    - Win: +1
    - Loss: -1
    - Draw: 0
    - Intermediate:
        + center control
        + create threats (2-in-a-row, 3-in-a-row)
        + block opponent
        - allow opponent immediate win
    """

    if result == player:
        return 1.0
    if result == 0:
        return 0.0
    if result is not None:
        return -1.0

    board = env.get_board()
    reward = 0.0

    # center column bonus
    center_col = COLS // 2
    center_count = sum(1 for r in range(ROWS) if board[r, center_col] == player)
    reward += 0.03 * center_count

    # count patterns
    reward += evaluate_board(board, player)

    # penalize opponent threats
    reward -= evaluate_board(board, -player) * 0.8

    # immediate blunder penalty
    if gives_opponent_win(env, player):
        reward -= 0.5

    return reward


def evaluate_board(board, player):
    score = 0.0

    # check all directions
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for r in range(ROWS):
        for c in range(COLS):
            if board[r, c] != player:
                continue

            for dr, dc in directions:
                count = 1
                empty = 0

                for i in range(1, CONNECT_N):
                    rr = r + dr * i
                    cc = c + dc * i

                    if not (0 <= rr < ROWS and 0 <= cc < COLS):
                        break

                    if board[rr, cc] == player:
                        count += 1
                    elif board[rr, cc] == EMPTY:
                        empty += 1
                        break
                    else:
                        break

                # scoring
                if count == 2:
                    score += 0.02
                elif count == 3:
                    score += 0.1

    return score


def gives_opponent_win(env, player):
    """
    Check if opponent can win in one move
    """
    opponent = -player
    legal_actions = env.get_legal_actions()

    for action in legal_actions:
        temp_env = clone_env(env)
        temp_env.drop_piece(action)

        if temp_env.get_result() == opponent:
            return True

    return False


def clone_env(env):
    from env.connect4_env import Connect4Env

    new_env = Connect4Env()
    new_env.board = env.get_board().copy()
    new_env.current_player = env.get_current_player()
    new_env.game_over = env.game_over
    new_env.winner = env.winner

    return new_env